# import json
# import string
# import re
# """calculate the WER & PER"""
# def WER(ref, hyp ,debug=False):
#     r = ref.lower().translate(str.maketrans('', '', string.punctuation)).split()
#     h = hyp.lower().translate(str.maketrans('', '', string.punctuation)).split()
#     #costs will holds the costs, like in the Levenshtein distance algorithm
#     costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
#     # backtrace will hold the operations we've done.
#     # so we could later backtrace, like the WER algorithm requires us to.
#     backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

#     OP_OK = 0
#     OP_SUB = 1
#     OP_INS = 2
#     OP_DEL = 3

#     DEL_PENALTY=1 # Tact
#     INS_PENALTY=1 # Tact
#     SUB_PENALTY=1 # Tact
#     # First column represents the case where we achieve zero
#     # hypothesis words by deleting all reference words.
#     for i in range(1, len(r)+1):
#         costs[i][0] = DEL_PENALTY*i
#         backtrace[i][0] = OP_DEL

#     # First row represents the case where we achieve the hypothesis
#     # by inserting all hypothesis words into a zero-length reference.
#     for j in range(1, len(h) + 1):
#         costs[0][j] = INS_PENALTY * j
#         backtrace[0][j] = OP_INS

#     # computation
#     for i in range(1, len(r)+1):
#         for j in range(1, len(h)+1):
#             if r[i-1] == h[j-1]:
#                 costs[i][j] = costs[i-1][j-1]
#                 backtrace[i][j] = OP_OK
#             else:
#                 substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
#                 insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
#                 deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

#                 costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
#                 if costs[i][j] == substitutionCost:
#                     backtrace[i][j] = OP_SUB
#                 elif costs[i][j] == insertionCost:
#                     backtrace[i][j] = OP_INS
#                 else:
#                     backtrace[i][j] = OP_DEL

#     # back trace though the best route:
#     i = len(r)
#     j = len(h)
#     numSub = 0
#     numDel = 0
#     numIns = 0
#     numCor = 0
#     if debug:
#         print("OP\tREF\tHYP")
#         lines = []
#     while i > 0 or j > 0:
#         if backtrace[i][j] == OP_OK:
#             numCor += 1
#             i-=1
#             j-=1
#             if debug:
#                 lines.append("OK\t" + r[i]+"\t"+h[j])
#         elif backtrace[i][j] == OP_SUB:
#             numSub +=1
#             i-=1
#             j-=1
#             if debug:
#                 lines.append("SUB\t" + r[i]+"\t"+h[j])
#         elif backtrace[i][j] == OP_INS:
#             numIns += 1
#             j-=1
#             if debug:
#                 lines.append("INS\t" + "****" + "\t" + h[j])
#         elif backtrace[i][j] == OP_DEL:
#             numDel += 1
#             i-=1
#             if debug:
#                 lines.append("DEL\t" + r[i]+"\t"+"****")
#     if debug:
#         lines = reversed(lines)
#         for line in lines:
#             print(line)
#         print("Ncor " + str(numCor))
#         print("Nsub " + str(numSub))
#         print("Ndel " + str(numDel))
#         print("Nins " + str(numIns))
#     wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
#     # return wer_result
#     return {'WER':wer_result, 'Cor':numCor, 'Sub':numSub, 'Ins':numIns, 'Del':numDel}

# """calculate the spelling errors rate with four models"""
# class PronunciationDictionary(object):
#     def __init__(self):
#         self.word2phone = {}
#         # self.phones = set()
#         self.phone2word = {}
#         # self.homophones = {}

#         # with open("ASR/cmudict-0.7b") as rf:
#         with open("cmudict_SPHINX_40") as rf:
#             last_word = ""
#             for line in rf:
#                 # content = line.strip().split("  ")
#                 content = line.strip()
#                 content = re.split('" "|\t', content)
#                 word = content[0].lower()
#                 phone = " ".join(content[1:])
#                 if word.find("(1)") != -1 or word.find("(2)") != -1 or word.find("(3)") != -1 or word.find("(4)") != -1:
#                     self.word2phone[f"{word[:-3]}"].append(phone)
#                 else:
#                     self.word2phone[f"{word}"] = [phone]
#                     self.phone2word[f"{phone}"] = word



#     def generate_phone_sequence(self, sentence):
#         # words = word_tokenize(sentence)
#         spell_error = 0
#         words = sentence.lower().translate(str.maketrans('', '', string.punctuation)).strip().split(" ")
#         phonewords = []
#         for word in words:
#             if word not in self.word2phone:
#                 phone = word
#                 spell_error+=1
#             else:
#                 phone = self.word2phone[f"{word}"][0]
#             phonewords.append(phone)
#         return spell_error
# with open ('output/WikiHow-facebook_large1.json', 'r') as load_f:
#     input1 = json.load(load_f)

# with open ('output/WikiHow-jon_large1.json', 'r') as load_f:
#     input2 = json.load(load_f)

# with open ('output/WikiHow-pytorch1.json', 'r') as load_f:
#     input3 = json.load(load_f)

# with open ('output/WikiHow-HuBERT1.json', 'r') as load_f:
#     input4 = json.load(load_f)

# #sum of spelling errors
# sum_error1 = 0
# sum_error2 = 0
# sum_error3 = 0
# sum_error4 = 0
# for i in range(len(input1)):
#     error = PronunciationDictionary().generate_phone_sequence(input1[i]["transcription"])
#     sum_error1 = sum_error1 + error

# for i in range(len(input2)):
#     error = PronunciationDictionary().generate_phone_sequence(input2[i]["transcription"])
#     sum_error2 = sum_error2 + error

# for i in range(len(input3)):
#     error = PronunciationDictionary().generate_phone_sequence(input3[i]["transcription"])
#     sum_error3 = sum_error3 + error

# for i in range(len(input4)):
#     error = PronunciationDictionary().generate_phone_sequence(input4[i]["transcription"])
#     sum_error4 = sum_error4 + error

# #sum of all of the words in all sentences
# sum_word1 = 0
# sum_word2 = 0
# sum_word3 = 0
# sum_word4 = 0

# for i in range(len(input1)):
#     trans_sentence = input1[i]["transcription"]
#     word = trans_sentence.split()
#     num_word = len(word)
#     sum_word1 = sum_word1 + num_word

# for i in range(len(input2)):
#     trans_sentence = input2[i]["transcription"]
#     word = trans_sentence.split()
#     num_word = len(word)
#     sum_word2 = sum_word2 + num_word

# for i in range(len(input3)):
#     trans_sentence = input3[i]["transcription"]
#     word = trans_sentence.split()
#     num_word = len(word)
#     sum_word3 = sum_word3 + num_word

# for i in range(len(input4)):
#     trans_sentence = input4[i]["transcription"]
#     word = trans_sentence.split()
#     num_word = len(word)
#     sum_word4 = sum_word4 + num_word





# #calculation of spelling error rate
# print("spelling error rate for model1_wav2vec2_large_960h:" + str(sum_error1 /(float)(sum_word1)))

# print("spelling error rate for model2_wav2cev_xlsr_large:" + str(sum_error2 /(float)(sum_word2)))

# print("spelling error rate for model3_SILERO:" + str(sum_error3 /(float)(sum_word3)))

# print("spelling error rate for model4_HuBERT_large:" + str(sum_error4 /(float)(sum_word4)))


# fo = open("output/sum_spelling_errors.txt", "w")
# fo.write("spelling error rate for model1_wav2vec2_large_960h:" + str(sum_error1))
# fo.write("spelling error rate for model2_wav2cev_xlsr_large:" + str(sum_error2))
# fo.write("spelling error rate for model3_SILERO:" + str(sum_error3))
# fo.write("spelling error rate for model4_HuBERT_large:" + str(sum_error4))

# fo.close()

# """wer"""
# # WER
# #model1 result
# avg_WER_list1 = []
# for i in range(len(input1)):
#     wer_score1 = WER(input1[i]["reference"], input1[i]["transcription"])['WER']
#     avg_WER_list1.append(wer_score1)

# # WER
# #model2 result
# avg_WER_list2 = []
# for i in range(len(input2)):
#     wer_score2 = WER(input2[i]["reference"], input2[i]["transcription"])['WER']
#     avg_WER_list2.append(wer_score2)

# # WER
# #model3 result
# avg_WER_list3 = []
# for i in range(len(input3)):
#     wer_score3 = WER(input3[i]["reference"], input3[i]["transcription"])['WER']
#     avg_WER_list3.append(wer_score3)

# # WER
# #model4 result
# avg_WER_list4 = []
# for i in range(len(input4)):
#     wer_score4 = WER(input4[i]["reference"], input4[i]["transcription"])['WER']
#     avg_WER_list4.append(wer_score4)


# #avg WER calculation
# print("AVG_WER_model1_wav2vec2_large_960h:" + str(sum(avg_WER_list1) / (float)(len(avg_WER_list1))))

# print("AVG_WER_model2_wav2cev_xlsr_large:" + str(sum(avg_WER_list2) / (float)(len(avg_WER_list2))))

# print("AVG_WER_model3_SILERO:" + str(sum(avg_WER_list3) / (float)(len(avg_WER_list3))))

# print("AVG_WER_model4_HuBERT_large:" + str(sum(avg_WER_list4) / (float)(len(avg_WER_list4))))


import matplotlib.pyplot as plt

AVG_WER_Y = [0.2535, 0.2343, 0.2711, 0.2032]
spell_error_Y = [0.0784, 0.0871, 0.0775, 0.0611]
AVG_WER_X = [1, 2, 3, 4]

plt.plot(AVG_WER_X, AVG_WER_Y, 'o', label = 'AVG_WER')
plt.plot(AVG_WER_X, spell_error_Y, 'r+', label = 'spell_error')

index_ls = ['wav2vec2_large', 'wav2cev_xlsr', 'SILERO', 'HuBERT_large']
plt.xticks(AVG_WER_X, index_ls)
plt.xlabel('Models')
plt.ylabel('AVG_WER & spelling error rate')
plt.legend()
plt.title('Average WERs & Spelling Error Rate of 4 models')
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# val_ls = [np.random.randint(100) + i*20 for i in range(7)]
# scale_ls = range(7)
# index_ls = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
# plt.bar(scale_ls, val_ls)
# _ = plt.xticks(scale_ls,index_ls) ## 可以设置坐标字
# plt.title('Average customer flows Number by Weekdays')
# plt.show()