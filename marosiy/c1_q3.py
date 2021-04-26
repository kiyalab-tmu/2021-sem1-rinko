#!/usr/bin/python3
# -*- coding: utf-8 -*-

input_array1 = list(map(int, input().split(' '))) # 入力 ' 1 2 3 4 5' を想定
input_array2 = list(map(int, input().split(' ')))


input_array1.sort()
input_array2.sort()


if len(input_array1) > len(input_array2):
    long_array = input_array1
    short_array = input_array2
else:
    long_array = input_array2
    short_array = input_array1




for i in range(len(short_array)):
    if short_array[i] != long_array[i]:
        print(long_array[i])
        break
    if i == len(short_array):
        print(long_array[i+1]) 
    
