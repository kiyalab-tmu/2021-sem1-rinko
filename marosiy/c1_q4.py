#!/usr/bin/python3
# -*- coding: utf-8 -*-

input_array = list(map(int, input().split(' '))) # 入力 ' 1 2 3 4 5' を想定
k = input()


input_array.sort()

first_array = []
second_array = []


for first in range(len(input_array)-1):
    for second in range(first+1, len(input_array)):
        if input_array[first] + input_array[second] == k:
            first_array.append(input_array[first])
            second_array.append(input_array[second])
        elif input_array[first] + input_array[second] > k:
            break


for i in range(len(first_array)):
    print(str(first_array[i]) + ',' + str(second_array[i]))
