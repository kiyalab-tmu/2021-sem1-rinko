sample = [1, 3, 2, 2]
k = 4

sample = sorted(sample)
ans = set()
for i in range(len(sample)):
    for j in range(i + 1, len(sample)):
        if sample[i] + sample[j] == k:
            ans.add((sample[i], sample[j]))
print(ans)
