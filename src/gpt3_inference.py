#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import openai
import numpy as np
import csv
import os
openai.api_key = "Your OpenAi Key"

start_sequence = "\nA: "
restart_sequence = "\nQ: "

# GPT-3による推論
object_name = "coffee"
PROMPT_DATA = "../data/prompt.txt"
f = open(PROMPT_DATA, 'r')
prompt_data = f.read()
prompt = prompt_data + "\nQ: {} is in the\nA:".format(object_name)
# print(prompt)


# prompt = '''
# Bath slipper is in the bathroom.
# Plate is in the kitchen.
# Car toy is in the living room.
# Sheep doll is in the bedroom.
#
# Q: Towel is in the
# A: bathroom
#
# Q: {} is in the
# A:'''.format(object_name)
#


response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=1,
    logprobs=16,
    stop="\n")

print(prompt+response['choices'][0]['text'])
# print(response)
data = response['choices'][0]['logprobs']['top_logprobs'][0]

llm_w_list = []
w_logprobs = []
w_probs = []
for v in data:
    print(v, np.exp(data[v]))
    llm_w_list.append(v)
    w_logprobs.append(data[v])
    w_probs.append(np.exp(data[v]))

print("GPT-3側の単語辞書:{}\n".format(llm_w_list))


w_probs = [float(i) / sum(w_probs) for i in w_probs]  # 正規化
print("GPT-3側の確率分布:{}\n".format(w_probs))
# print(sum(w_probs))


# SpCoSLAM側の単語の辞書に合わせる処理
with open('../data/W_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        pass
    place_names_spco = row
place_names_spco.pop(-1)
print("SpCo側の単語辞書:{}\n".format(place_names_spco))

# SpCo側に含まれない単語の抽出
diff = list(set(llm_w_list) - set(place_names_spco))

# GPT-3側とSpCo側で統合した単語辞書
integration_dict = place_names_spco + diff

llm_probs_r = [0 for i in range(len(integration_dict))]  # 場所概念の単語数に合わせた表現
for j in range(len(integration_dict)):
    # GPT-3側にある単語のとき
    if ((integration_dict[j] in llm_w_list) == True):
        a = llm_w_list.index(integration_dict[j])
        llm_probs_r[j] = w_probs[a]
    # GPT-3側にないとき
    else:
        llm_probs_r[j] = 0

print("統合した単語辞書:{}\n".format(integration_dict))
print("統合した確率分布:{}\n".format(llm_probs_r))



FilePath = path = "../data"
if not os.path.exists(FilePath):
    os.makedirs(FilePath)

# csvファイルで1行目に保存
with open(FilePath + "/gpt3_inference_result.csv", 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(llm_probs_r)

# 統合した単語辞書の保存
with open(FilePath + "/integration_W_list.csv", 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(integration_dict)

# 統合した単語辞書
with open(FilePath + '/integration_W_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        pass
    integrated_place_name_list = row

print(integrated_place_name_list)
integrated_place_name_list.pop(-1)
