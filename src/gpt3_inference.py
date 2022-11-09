#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import openai
import numpy as np
openai.api_key = "Your OpenAI API key"

start_sequence = "\nA: "
restart_sequence = "\nQ: "

prompt = '''
Bath slipper is in the bathroom.
Plate is in the kitchen.
Car toy is in the living room.
Sheep doll is in the bedroom.

Q: Towel is in the
A: bathroom

Q: coffee is in the
A:'''

response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=1,
    logprobs=16,
    stop="\n")

# print(prompt+response['choices'][0]['text'])
print(response)
data = response['choices'][0]['logprobs']['top_logprobs'][0]

w_list = []
w_logprobs = []
w_probs = []
for v in data:
    w_list.append(v)
    w_logprobs.append(data[v])
    w_probs.append(np.exp(data[v]))

print(w_list)
print(w_probs)

w_probs = [float(i) / sum(w_probs) for i in w_probs]  # 正規化
# print(w_probs)
# print(sum(w_probs))
