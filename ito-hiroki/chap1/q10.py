import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 4, 3])

# 説明を聞いて思ったけど，shape(size)が同じかを確認しなきゃいけないね
# このままだとnumpyのbroadcastによって，a/bの条件によっては結果が違う可能性がある
print((a == b).all())
print((a == a).all())
