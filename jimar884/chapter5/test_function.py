# a = {'a':1, 'b':2, 'c':4}
# for i, item in enumerate(a):
#     print(i, item)

# for i in a:
#     print(a)
# a = [1, 2, 3, 4, 5, 6, 7]
# print(a.index(1))

# a = "hello world   aiueo"
# a = ''.join(a)
# print(a)

# import torch
# a = torch.tensor(
#     [
#         [1, 2, 3],
#         [4, 5, 6]
#     ]
# )
# print(a)
# b = torch.cat((a, None), 1)
# print(b)

from math import floor

def apply_threshold(value, bitdepth):
    return floor(floor(value / 2**(8 - bitdepth)) / (2**bitdepth - 1) * 255)

a = [i for i in range(256)]
print(a)
print("--"*10)
a1 = []
for v in a:
    a1.append(apply_threshold(v, 1))
print(a1)
print("--"*10)
a2 = []
for v in a:
    a2.append(apply_threshold(v, 2))
print(a2)