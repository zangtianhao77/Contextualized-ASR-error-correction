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
import json
import string
import speech_recognition as sr
from os import path
from huggingsound import SpeechRecognitionModel
"""load an ASR model"""
model = SpeechRecognitionModel("facebook/wav2vec2-base-960h")

"""add random synthetic noise using SSML tags"""
def addSSML(input_text,display=False):
    prosody_tags={
        "volume" : ["x-soft", "soft", "medium", "loud", "x-loud"],
        "rate" : ["slow", "medium", "fast"],
        "pitch" : ["x-low", "low", "medium", "high", "x-high"]
    }
    tag = random.sample(prosody_tags.keys(),1)[0]
    value = random.sample(prosody_tags[tag],1)[0]
    input_text=input_text.replace("&","and").replace("<","less than").replace(">","greater than").replace("'","").replace('"',"")
    result="<speak><prosody "+ tag +"='"+value+"'>"+ input_text +"</prosody></speak>"
    if display:
        print(result)
    return result

"""Text-TO-Speech"""
def TTS(input_text,display=False):
    # # Create a client using the credentials and region defined in the [adminuser]
    # # section of the AWS credentials file (~/.aws/credentials).
    session = Session(profile_name="default")
    polly = session.client("polly")

    try:
        # Request speech synthesis
        # VoiceId=random.sample(["Nicole", "Hannah", "Kevin", "Enrique", "Tatyana", "Russell", "Olivia", "Lotte", "Geraint", "Carmen", "Ayanda", "Mads", "Penelope", "Mia",
        #  "Joanna", "Matthew", "Brian", "Seoyeon", "Ruben", "Ricardo", "Maxim", "Lea", "Giorgio", "Carla", "Aria", "Naja", "Arlet", "Maja", "Astrid",
        #  "Ivy", "Kimberly", "Chantal", "Amy", "Vicki", "Marlene", "Ewa", "Conchita", "Camila", "Karl", "Zeina", "Miguel", "Mathieu", "Justin",
        #  "Lucia", "Jacek", "Bianca", "Takumi", "Ines", "Gwyneth", "Cristiano", "Mizuki", "Celine", "Zhiyu", "Jan", "Liv", "Joey", "Raveena", "Filiz",
        #  "Dora", "Salli", "Aditi", "Vitoria", "Emma", "Lupe", "Hans", "Kendra", "Gabrielle"],1)[0]
        VoiceId=random.sample([ "Salli", "Joanna", "Ivy", "Kendra", "Kimberly", "Matthew", "Justin", "Joey"],1)[0]
        response = polly.synthesize_speech(TextType='ssml',Text=input_text, OutputFormat="mp3",
                                        VoiceId=VoiceId)
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
    path = 'noise/'
    list = os.listdir(path)
    n = random.randint(0, len(list)-1)
    noise_file = list[n]
    final_path = path + noise_file
    # assert os.path.isfile(final_path)
    # with open(final_path, "r") as f:
    #     pass

    sound1 = AudioSegment.from_mp3(final_path)
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
def STT(speech):

    # audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]
    audio_paths = [speech]

    transcriptions = model.transcribe(audio_paths)

    return transcriptions[0]['transcription']
# def STT(speech):

#     AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), speech)
#     # AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "french.aiff")
#     # AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "chinese.flac")

#     # use the audio file as the audio source
#     r = sr.Recognizer()
#     with sr.AudioFile(AUDIO_FILE) as source:
#         audio = r.record(source)  # read the entire audio file

#     # recognize speech using Google Speech Recognition
#     return r.recognize_google(audio)
# def STT(speech):
#     device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU

#     model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
#                                        model='silero_stt',
#                                        language='en', # also available 'de', 'es'
#                                        device=device)
#     (read_batch, split_into_batches,
#     read_audio, prepare_model_input) = utils  # see function signature for details

#     # download a single file, any format compatible with TorchAudio (soundfile backend)
#     # torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
#     #                                dst ='speech_orig.wav', progress=True)
#     # test_files = glob('speech_orig.wav')
#     ##add file path
#     file_name = os.path.abspath(speech)
#     # convert a filename to a url
#     url_name = pathlib.Path(file_name).as_uri()
#     torch.hub.download_url_to_file(url_name,
#                                 dst =speech, progress=True)
#     test_files = glob(speech)
#     batches = split_into_batches(test_files, batch_size=10)
#     input = prepare_model_input(read_batch(batches[0]),
#                                 device=device)
#     output= model(input)
#     for example in output:
#         return decoder(example.cpu())

def WER(ref, hyp ,debug=False):
    r = ref.lower().translate(str.maketrans('', '', string.punctuation)).split()
    h = hyp.lower().translate(str.maketrans('', '', string.punctuation)).split()
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
    # return wer_result
    return {'WER':wer_result, 'Cor':numCor, 'Sub':numSub, 'Ins':numIns, 'Del':numDel}

class PronunciationDictionary(object):
    def __init__(self):
        self.word2phone = {}
        # self.phones = set()
        self.phone2word = {}
        # self.homophones = {}

        # with open("ASR/cmudict-0.7b") as rf:
        with open("cmudict_SPHINX_40") as rf:
            last_word = ""
            for line in rf:
                # content = line.strip().split("  ")
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
        words = sentence.lower().translate(str.maketrans('', '', string.punctuation)).strip().split(" ")
        phonewords = []
        for word in words:
            if word not in self.word2phone:
                phone = word
            else:
                phone = self.word2phone[f"{word}"][0]
            phonewords.append(phone)
        return " ".join(phonewords)


if __name__ == "__main__":
    inputs = []
    avg_WER_list = []

    # with open("train.dataflow_dialogues.jsonl", 'r', encoding='UTF-8') as f:
    #     json_list = [json.loads(line) for line in f]
    # for i in range(50):
    #     for j in json_list[i]["turns"]:
    #         inputs.append(j["user_utterance"]['original_text'])
    list = os.listdir('wikihow_corpus\js_files_en/')
    for i in range(200):
        file = list[i]
        with open('wikihow_corpus\js_files_en/'+file, 'r',encoding='UTF-8') as load_f:
            json_file = json.load(load_f)
            if not json_file["title"] is None:
                inputs.append(json_file["title"])
            for j in json_file["QAs"]:
                inputs.append(j[0])
    results_json = []
    for _ in inputs:
        reference=_
        input_text=addSSML(reference)
        speech=TTS(input_text)
        mixed_speech=addNoise(speech)
        transcription=STT(mixed_speech)

        print("Reference:\t\t"+reference)
        print("Transcription:\t"+transcription)
        wer=WER(reference, transcription)
        errors=[]
        if wer['Ins']>0:
            errors.append("Insertion")
        if wer['Del']>0:
            errors.append("Deletion")
        if wer['Sub']>0:
            errors.append("Substitution")
        print("WER: " + str(WER(reference, transcription)['WER']))
        avg_WER_list.append(WER(reference, transcription)['WER'])
        print("PER: " + str(WER(PronunciationDictionary().generate_phone_sequence(reference), PronunciationDictionary().generate_phone_sequence(transcription))['WER']))
        print(errors)
        results_json.append({
            'errors':errors,
            'reference': reference,
            'transcription': transcription})

    #avg WER calculation
    avg_WER = sum(avg_WER_list) / len(avg_WER_list)
    print("AVG_WER:" + round(avg_WER,3))

    with open('output/WikiHow-jon.json', 'w') as f:
        f.write(json.dumps(results_json, indent=4))
