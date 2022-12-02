#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import openai
import numpy as np
import csv


# 統合した単語辞書
with open('../data/integration_W_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        pass
    integrated_place_name_list = row

theta_sw = []
with open('../data/W.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        del row[-1]
        theta_sw.append(np.array(row, dtype=np.float64))

# 場所の単語辞書
with open('../data/W_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        pass
    place_name_list = row

place_name_list.pop(-1)

# print(integrated_place_name_list)
# print(theta_sw)
# print(place_name_list)

################################
## ここでGPT-3で増えた要素分を0でパディングする
# 統合した辞書とSpCoの辞書での差分数を確認
diff = len(integrated_place_name_list) - len(place_name_list)
add_empty = np.zeros(diff)

# print(add_empty)

# print(len(integrated_place_name_list))

theta_sw_integrate = np.zeros((len(theta_sw), len(integrated_place_name_list)))
for l in range(len(theta_sw)):
    theta_sw_integrate[l] = np.insert(theta_sw[l], len(theta_sw[l]), add_empty)

print(theta_sw_integrate)