#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

input_array = list(map(int, input().split(' '))) # 入力 ' 1 2 3 4 5' を想定

index_array = []
for i in range(len(input_array)):
    if 3 <= input_array[i] and input_array[i] <=8:
        index_array.append(i)


input_array = np.array(input_array)

print(np.delete(input_array, index_array))
