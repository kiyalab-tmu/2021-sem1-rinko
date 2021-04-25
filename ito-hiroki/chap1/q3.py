from collections import Counter

sample_one = [2, 3, 4, 5, 6, 7, 5, 8]
sample_two = [6, 8, 7, 4, 5, 2, 3]

print(list((Counter(sample_one) - Counter(sample_two)).keys())[0])