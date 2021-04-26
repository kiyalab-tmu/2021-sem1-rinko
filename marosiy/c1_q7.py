#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

a = np.random.rand(5, 5)

min_num = a.min()
max_num = a.max()

# Min-Max Normalization
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        a[i, j] = (a[i, j] - min_num) / (max_num - min_num)

print('a.max:', a.max())
print('a.min:', a.min())
