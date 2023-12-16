from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing
import sys
from tempfile import gettempdir
from pydub import AudioSegment
import random
import os
import torch
from glob import glob
import pathlib
import re
import speech_recognition as sr
from os import path

"""add random synthetic noise using SSML tags"""
def addSSML(input_text,display=False):
    prosody_tags={
        "volume" : ["x-soft", "soft", "medium", "loud", "x-loud"],
        "rate" : ["x-slow", "slow", "medium", "fast", "x-fast"],
        "pitch" : ["x-low", "low", "medium", "high", "x-high"]
    }
    tag = random.sample(prosody_tags.keys(),1)[0]
    value = random.sample(prosody_tags[tag],1)[0]
    result="<speak><prosody "+ tag +"='"+value+"'>"+ input_text +"</prosody></speak>"
    if display:
        print(result)
    return result

"""Text-TO-Speech"""
# Need to configure first according to the official website: https://docs.aws.amazon.com/polly/latest/dg/getting-started.html
def TTS(input_text,display=False):
    # # Create a client using the credentials and region defined in the [adminuser]
    # # section of the AWS credentials file (~/.aws/credentials).
    session = Session(profile_name="default")
    polly = session.client("polly",region_name='us-east-2')

    try:
        # Request speech synthesis
        response = polly.synthesize_speech(TextType='ssml',Text=input_text, OutputFormat="mp3",
                                        VoiceId="Joanna")
    except (BotoCoreError, ClientError) as error:
        # The service returned an error, exit gracefully
        print(error)
        sys.exit(-1)

    # Access the audio stream from the response
    if "AudioStream" in response:
        # Note: Closing the stream is important because the service throttles on the
        # number of parallel connections. Here we are using contextlib.closing to
        # ensure the close method of the stream object will be called automatically
        # at the end of the with statement's scope.
        with closing(response["AudioStream"]) as stream:
            output = os.path.join(gettempdir(), "speech.mp3")
            if display:
                output ="audio_display/speech.mp3"

            try:
            # Open a file for writing the output as a binary stream
                with open(output, "wb") as file:
                   file.write(stream.read())
            except IOError as error:
              # Could not write to file, exit gracefully
              print(error)
              sys.exit(-1)

    else:
        # The response didn't contain audio data, exit gracefully
        print("Could not stream audio")
        sys.exit(-1)

    return output


"""combine a random noise and speech"""
def addNoise(speech,display=False):

    #load noise files into list
    path = 'noise\\'
    list = os.listdir(path)
    n = random.randint(0, len(list))
    noise_file = list[n]

    sound1 = AudioSegment.from_mp3('noise\\'+ noise_file)
    sound2 = AudioSegment.from_mp3(speech)

    # mix sound2 with sound1, starting at random ms into sound1)
    duration = sound2.duration_seconds * 1000
    pos = random.uniform(0, duration)
    output = sound2.overlay(sound1,position=pos)

    # save the result
    mixed_speech = os.path.join(gettempdir(), "mixed_speech.wav")
    if display:
        mixed_speech = "audio_display/mixed_speech.wav"
    output.export(mixed_speech, format="wav")
    return mixed_speech

"""Speech-To-Text"""
# model: https://pytorch.org/hub/snakers4_silero-models_stt/
def STT(speech):

    AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), speech)
    # AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "french.aiff")
    # AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "chinese.flac")

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Google Speech Recognition
    return r.recognize_google(audio)

def WER(ref, hyp ,debug=False):
    r = ref.lower().split()
    h = hyp.lower().split()
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    DEL_PENALTY=1 # Tact
    INS_PENALTY=1 # Tact
    SUB_PENALTY=1 # Tact
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("Ncor " + str(numCor))
        print("Nsub " + str(numSub))
        print("Ndel " + str(numDel))
        print("Nins " + str(numIns))
    wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
    return wer_result
    # return {'WER':wer_result, 'Cor':numCor, 'Sub':numSub, 'Ins':numIns, 'Del':numDel}

class PronunciationDictionary(object):
    def __init__(self):
        self.word2phone = {}
        self.phone2word = {}

        with open("cmudict_SPHINX_40") as rf:
            for line in rf:
                content = line.strip()
                content = re.split('" "|\t', content)
                word = content[0].lower()
                phone = " ".join(content[1:])
                if word.find("(1)") != -1 or word.find("(2)") != -1 or word.find("(3)") != -1 or word.find("(4)") != -1:
                    self.word2phone[f"{word[:-3]}"].append(phone)
                else:
                    self.word2phone[f"{word}"] = [phone]
                    self.phone2word[f"{phone}"] = word


    def generate_phone_sequence(self, sentence):
        # words = word_tokenize(sentence)
        words = sentence.lower().strip().split(" ")
        phonewords = []
        for word in words:
            if word not in self.word2phone:
                phone = word
            else:
                phone = self.word2phone[f"{word}"][0]
            phonewords.append(phone)
        return " ".join(phonewords)


if __name__ == "__main__":
    random.seed(1)
    reference = "Can you make an event for Friday?"

    input_text=addSSML(reference)
    speech=TTS(input_text,True)
    mixed_speech=addNoise(speech,True)
    transcription=STT(mixed_speech)

    print("Reference:\t\t"+reference)
    print("Transcription:\t"+transcription)
    print("WER: " + str(WER(reference, transcription)))
    print("PER: " + str(WER(PronunciationDictionary().generate_phone_sequence(reference), PronunciationDictionary().generate_phone_sequence(transcription))))