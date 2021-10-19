#!/usr/bin/python3
# -*- coding: utf-8 -*-


for i in range(1, 11):
    print(str(i).ljust(3), end='')
    print('|', end='')
    for j in range(1, 11):
        print(str(i*j).rjust(4), end='')
    print()
    if i == 1:
        for i in range(44):
            print('#', end='')
        print()