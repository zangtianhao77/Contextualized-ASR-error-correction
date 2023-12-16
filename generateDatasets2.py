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
from huggingsound import SpeechRecognitionModel
import librosa
from dtw import dtw
from numpy.linalg import norm
"""add random synthetic noise using SSML tags"""
# https://docs.aws.amazon.com/polly/latest/dg/supportedtags.html
def addSSML(input_text,display=False):
    prosody_tags={
        "volume" : ["soft","medium","loud"],
        "rate" : ["slow","medium","fast"],
        "pitch" : ["low","medium" ,"high"]
    }
    volume=random.sample(prosody_tags["volume"],1)[0]
    rate = random.sample(prosody_tags["rate"], 1)[0]
    pitch = random.sample(prosody_tags["pitch"], 1)[0]
    input_text=input_text.replace("&","and").replace("<","less than").replace(">","greater than").replace("'","").replace('"',"")
    result = "<speak><prosody volume ='"+volume +"'"+ " rate ='"+rate +"'"+" pitch ='"+pitch +"'>"+ input_text +"</prosody></speak>"
    if display:
        print(result)

    return ({'volume':volume,'rate': rate,'pitch': pitch},result)

def testSSML(input_text,n=3,display=False):
    output=[]
    prosody_tags={
        "volume" : ["soft","medium","loud"],
        "rate" : ["slow","medium","fast"],
        "pitch" : ["low","medium" ,"high"]
    }
    input_text=input_text.replace("&","and").replace("<","less than").replace(">","greater than").replace("'","").replace('"',"")
    for volume in prosody_tags["volume"]:
        for rate in prosody_tags["rate"]:
            for pitch in prosody_tags["pitch"]:
                result = "<speak><prosody volume ='"+volume +"'"+ " rate ='"+rate +"'"+" pitch ='"+pitch +"'>"+ input_text +"</prosody></speak>"
                output.append([{'volume':volume,'rate': rate,'pitch': pitch},result])

    if display:
        print(output)
    return output


"""Text-TO-Speech"""
def TTS(input_text,VoiceId,display=False):
    # # Create a client using the credentials and region defined in the [adminuser]
    # # section of the AWS credentials file (~/.aws/credentials).
    session = Session(profile_name="default")
    polly = session.client("polly")

    try:
        # Request speech synthesis
        # https://docs.aws.amazon.com/polly/latest/dg/voicelist.html
        # VoiceId=random.sample(["Nicole", "Hannah", "Kevin", "Enrique", "Tatyana", "Russell", "Olivia", "Lotte", "Geraint", "Carmen", "Ayanda", "Mads", "Penelope", "Mia",
        #  "Joanna", "Matthew", "Brian", "Seoyeon", "Ruben", "Ricardo", "Maxim", "Lea", "Giorgio", "Carla", "Aria", "Naja", "Arlet", "Maja", "Astrid",
        #  "Ivy", "Kimberly", "Chantal", "Amy", "Vicki", "Marlene", "Ewa", "Conchita", "Camila", "Karl", "Zeina", "Miguel", "Mathieu", "Justin",
        #  "Lucia", "Jacek", "Bianca", "Takumi", "Ines", "Gwyneth", "Cristiano", "Mizuki", "Celine", "Zhiyu", "Jan", "Liv", "Joey", "Raveena", "Filiz",
        #  "Dora", "Salli", "Aditi", "Vitoria", "Emma", "Lupe", "Hans", "Kendra", "Gabrielle"],1)[0]
        # VoiceId=random.sample(["Salli", "Joanna", "Ivy", "Kendra", "Kimberly", "Matthew", "Justin", "Joey"],1)[0]
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
# https://github.com/jiaaro/pydub/
def addNoise(speech,display=False):

    # load noise files into list
    path = 'noise/'
    list = os.listdir(path)
    n = random.randint(0, len(list)-1)
    noise_file = list[n]
    silence = AudioSegment.silent(duration=random.uniform(0, 3000))
    noise = AudioSegment.from_mp3(path + noise_file) - 5
    speech = AudioSegment.from_mp3(speech)

    # mix sound2 with sound1, starting at random ms into sound1)
    duration = speech.duration_seconds * 1000
    pos = random.uniform(0, duration)
    speech=speech+silence
    output = speech.overlay(noise,position=pos)
    # save the result
    mixed_speech = os.path.join(gettempdir(), "mixed_speech.wav")
    if display:
        mixed_speech = "audio_display/mixed_speech.wav"
    output.export(mixed_speech, format="wav")
    return mixed_speech

"""Speech-To-Text"""
# https://github.com/Uberi/speech_recognition
def STT(speech):
    device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU

    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
    (read_batch, split_into_batches,
    read_audio, prepare_model_input) = utils  # see function signature for details

    # download a single file, any format compatible with TorchAudio (soundfile backend)
    # torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
    #                                dst ='speech_orig.wav', progress=True)
    # test_files = glob('speech_orig.wav')
    ##add file path
    file_name = os.path.abspath(speech)
    # convert a filename to a url
    url_name = pathlib.Path(file_name).as_uri()
    torch.hub.download_url_to_file(url_name,
                                dst =speech, progress=True)
    test_files = glob(speech)
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]),
                                device=device)
    output= model(input)
    for example in output:
        return decoder(example.cpu())

# https://huggingface.co/facebook/wav2vec2-large-960h
def STT_facebook(speech):
    model = SpeechRecognitionModel("facebook/wav2vec2-large-960h")
    # audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]
    audio_paths = [speech]
    # facebook large
    transcriptions = model.transcribe(audio_paths)

    return transcriptions[0]['transcription'].lower()

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

    # with open("dstc8-schema-guided-dialogue-master/train\dialogues_001.json", 'r') as load_f:
    #     input = json.load(load_f)
    # for i in range(50):
    #     for j in input[i]["turns"]:
    #         inputs.append(j["utterance"])

    # with open("task_oriented_dialogue_as_dataflow_synthesis-master\datasets\SMCalFlow 2.0/train.dataflow_dialogues.jsonl", 'r', encoding='UTF-8') as f:
    #     json_list = [json.loads(line) for line in f]
    # for i in range(50):
    #     for j in json_list[i]["turns"]:
    #         inputs.append(j["user_utterance"]['original_text'])

    # voices ={
    #     "Australian" : ["Nicole", "Olivia", "Russell"],
    #     "British" : ["Amy", "Emma", "Brian"],
    #     "Indian" : ["Aditi", "Raveena"],
    #     "New Zealand": ["Aria"],
    #     "South African": ["Ayanda"],
    #     "US": ["Salli", "Joanna", "Ivy", "Kendra", "Kimberly", "Matthew", "Justin", "Joey"]
    # }

    speakers ={
        "Nicole" : ["en-AU", "Female", "Adult"],
        "Russell" : ["en-AU", "Male", "Adult"],
        "Amy" : ["en-GB", "Female", "Adult"],
        "Emma" : ["en-GB", "Female", "Adult"],
        "Brian" : ["en-GB", "Male", "Adult"],
        "Aditi": ["en-IN", "Female", "Adult"],
        "Raveena": ["en-IN", "Female", "Adult"],
        "Ivy": ["en-US", "Female", "child"],
        "Joanna": ["en-US", "Female", "Adult"],
        "Kendra": ["en-US", "Female", "Adult"],
        "Kimberly": ["en-US", "Female", "Adult"],
        "Salli": ["en-US", "Female", "Adult"],
        "Joey": ["en-US", "Male", "Adult"],
        "Justin": ["en-US", "Male", "child"],
        "Matthew": ["en-US", "Male", "Adult"]
    }


    list = os.listdir('wikihow_corpus\js_files_en/')
    for i in range(len(list)):
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
        transcriptions = []
        print("Reference:\t\t"+reference)
        for j in range(2):
            input_SSML = addSSML(reference)
            voiceId=random.choice(tuple(speakers.keys()))
            speech = TTS(input_SSML[1],voiceId)
            mixed_speech = addNoise(speech)
            transcription = STT_facebook(mixed_speech)
            print("Transcription:\t"+transcription)
            # wer=WER(reference, transcription)
            # errors=[]
            # if wer['Ins']>0:
            #     errors.append("Insertion")
            # if wer['Del']>0:
            #     errors.append("Deletion")
            # if wer['Sub']>0:
            #     errors.append("Substitution")
            # print("WER: " + str(WER(reference, transcription)['WER']))
            # print("PER: " + str(WER(PronunciationDictionary().generate_phone_sequence(reference), PronunciationDictionary().generate_phone_sequence(transcription))['WER']))
            transcriptions.append({
                'speaker':voiceId,
                "prosody": input_SSML[0],
                # "errors":errors,
                # 'WER':wer["WER"],
                'transcription': transcription})
        results_json.append({'reference': reference,'transcriptions':transcriptions})

    with open('output/WikiHow-trans-wav2vec2.json', 'w') as f:
        f.write(json.dumps(results_json, indent=4))
    # voice={}
    # speaker={}
    # with open("output/WikiHow.json", 'r') as load_f:
    #     input = json.load(load_f)
    # for i in input:
    #     reference=i["reference"]
    #     for j in i["transcriptions"]:
    #         for k in j["speaker"][1]:
    #             if k in voice:
    #                 voice[k].append(j["WER"])
    #             else:
    #                 voice[k]=[]
    #         if str(j["speaker"]) in speaker:
    #             speaker[str(j["speaker"])].append(j["WER"])
    #         else:
    #             speaker[str(j["speaker"])]=[]
    # for _ in voice:
    #     voice[_]=sum(voice[_])/len(voice[_])
    # for _ in speaker:
    #     speaker[_]=sum(speaker[_])/len(speaker[_])
    # with open('output/voice.json', 'w') as f:
    #     f.write(json.dumps(voice, indent=4))
    # with open('output/speaker.json', 'w') as f:
    #     f.write(json.dumps(speaker, indent=4))


    # with open("output/WikiHow-SSML.json", 'r') as load_f:
    #     input = json.load(load_f)
    # combine={}
    # value = {}
    # for i in input:
    #     reference=i["reference"]
    #     for j in i["transcriptions"]:
    #         if str(j["prosody"]) in combine:
    #             combine[str(j["prosody"])].append(j["WER"])
    #         else:
    #             combine[str(j["prosody"])]=[]
    #         for k in j["prosody"]:
    #             if k+":"+j["prosody"][k] in value:
    #                 value[k+":"+j["prosody"][k]].append(j["WER"])
    #             else:
    #                 value[k+":"+j["prosody"][k]] = []
    # for _ in combine:
    #     combine[_]=sum(combine[_])/len(combine[_])
    # for _ in value:
    #     value[_]=sum(value[_])/len(value[_])
    # with open('output/combine.json', 'w') as f:
    #     f.write(json.dumps(combine, indent=4))
    # with open('output/value.json', 'w') as f:
    #     f.write(json.dumps(value, indent=4))
            # mixed_speech = addNoise(speech)
    #         transcription = STT_facebook(speech)
    #         print("Transcription:\t"+transcription)
    #         print("WER: " + str(WER(reference, transcription)['WER']))
    #         wer = WER(reference, transcription)
    #         print("PER: " + str(WER(PronunciationDictionary().generate_phone_sequence(reference), PronunciationDictionary().generate_phone_sequence(transcription))['WER']))
    #         transcriptions.append({
    #             "prosody":j[0],
    #             'WER':wer["WER"],
    #             'transcription': transcription})
    #     results_json.append({'reference': reference,'transcriptions':transcriptions})
    #
    #
    # with open('output/WikiHow.json', 'w') as f:
    #     f.write(json.dumps(results_json, indent=4))
    #
    # with open("output/WikiHow.json", 'r') as load_f:
    #     input = json.load(load_f)
    # combine={}
    # value = {}
    # for i in input:
    #     reference=i["reference"]
    #     for j in i["transcriptions"]:
    #         if str(j["prosody"]) in combine:
    #             combine[str(j["prosody"])].append(j["WER"])
    #         else:
    #             combine[str(j["prosody"])]=[]
    #         for k in j["prosody"]:
    #             if k+":"+j["prosody"][k] in value:
    #                 value[k+":"+j["prosody"][k]].append(j["WER"])
    #             else:
    #                 value[k+":"+j["prosody"][k]] = []
    # for _ in combine:
    #     combine[_]=sum(combine[_])/len(combine[_])
    # for _ in value:
    #     value[_]=sum(value[_])/len(value[_])
    # with open('output/combine.json', 'w') as f:
    #     f.write(json.dumps(combine, indent=4))
    # with open('output/value.json', 'w') as f:
    #     f.write(json.dumps(value, indent=4))
    #
    # results_json = []
    # a=0
    # b=0
    # c=0
    # for _ in input:
    #     wer=WER(_["reference"], _["transcription"])["WER"]
    #     if wer<0.05:
    #         a=a+1
    #     if wer>=0.05 and wer<=0.4:
    #         results_json.append({
    #             'errors':_["errors"],'reference': _["reference"],
    #                              'transcription': _["transcription"]})
    #         b=b+1
    #     if wer>0.4:
    #         c=c+1
    # with open('output/WikiHow-wer0.05-0.40.json', 'w') as f:
    #     f.write(json.dumps(results_json, indent=4))
    # print("WER<0.05: "+str(a/len(input)))
    # print("0.05<=WER<0.4: "+str(b/len(input)))
    # print("0.4<WER: "+str(c/len(input)))
    #
    # with open("output/WikiHow-wer0.05-0.40.json", 'r') as load_f:
    #     input = json.load(load_f)
    #
    # results_json = []
    # I=[]
    # D=[]
    # S=[]
    # ID=[]
    # IS=[]
    # DS =[]
    # IDS=[]
    # num=0
    # for _ in input:
    #     if len(_["errors"])>1:
    #         num=num+1
    #
    #     if _["errors"]==["Insertion"]:
    #         I.append(_)
    #     if _["errors"]==["Deletion"]:
    #         D.append(_)
    #     if _["errors"]==["Substitution"]:
    #         S.append(_)
    #     if _["errors"]==["Insertion","Deletion"]:
    #         ID.append(_)
    #     if _["errors"]==["Deletion","Substitution"]:
    #         DS.append(_)
    #     if _["errors"]==["Insertion","Substitution"]:
    #         IS.append(_)
    #     if _["errors"]==["Insertion","Deletion","Substitution"]:
    #         IDS.append(_)
    #
    # results_json=I+D+S+ID+DS+IS+IDS
    #
    # with open('output/WikiHow-ASR.json', 'w') as f:
    #     f.write(json.dumps(results_json, indent=4))
    # print("With multiple types of errors: "+str(num/len(input)))
    #
    # reference=[]
    # transcription=[]
    # with open("output/WikiHow-ASR.json", 'r') as load_f:
    #     input = json.load(load_f)
    # s_wer=0
    # s_words=0
    # s_num=0
    # for _ in input:
    #     wer=WER(_["reference"], _["transcription"])
    #     s_wer+=wer["WER"]
    #     s_words+=wer['Sub']+wer['Ins']+wer['Del']
    # print("Average WER: "+str(s_wer/len(input)))
    # print("Average errors: "+str(s_words / len(input)))
    #
    # reference=[]
    # transcription=[]
    # with open("output/WikiHow-ASR.json", 'r') as load_f:
    #     input = json.load(load_f)
    # f1= open("output/WikiHow-reference.txt", 'w')
    # f2= open("output/WikiHow-transcription.txt", 'w')
    # for _ in input:
    #     f1.writelines(_["reference"].lower().translate(str.maketrans('', '', string.punctuation)))
    #     f1.writelines("\n")
    #     f2.writelines(_["transcription"])
    #     f2.writelines("\n")

    # VoiceId = random.sample(list(speakers.keys()), 1)[0]
    # random.seed(1)
    # reference="how to do your homework without being bored"
    # input_text=addSSML(reference)
    # speech=TTS(input_text,VoiceId,True)
    # mixed_speech=addNoise(speech,True)
    # transcription=STT_facebook(mixed_speech)
    # print(transcription)