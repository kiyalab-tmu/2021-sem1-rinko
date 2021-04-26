#!/usr/bin/python3
# -*- coding: utf-8 -*-

input_array1 = list(map(int, input().split(' '))) # 1 2 3 4 5　を想定
input_array2 = list(map(int, input().split(' ')))


if len(input_array1) > len(input_array2):
    long_array = input_array1
    short_array = input_array2
else:
    long_array = input_array2
    short_array = input_array1




for i in range(len(long_array)):
    is_not_found = True
    for j in range(len(short_array)):
        if long_array[i] == short_array[j]:
            is_not_found = False

    if is_not_found:
        print(long_array[i])
        break
    
    
