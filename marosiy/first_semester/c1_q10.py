#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

input_array1 = list(map(int, input().split(' '))) # 入力 ' 1 2 3 4 5' を想定
input_array2 = list(map(int, input().split(' '))) # 入力 ' 1 2 3 4 5' を想定

input_array1 = np.array(input_array1)
input_array2 = np.array(input_array2)


if input_array1.all == input_array2.all:
    print('一緒だったよ！')
else:
    print('違ったよ！')