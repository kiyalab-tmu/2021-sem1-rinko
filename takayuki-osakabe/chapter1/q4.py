def pair_sum(list, k):
    list = sorted(list)
    res = []

    for i in range(len(list)):
        for j in range(i+1, len(list)):
            if list[i] + list[j] == k:
                res.append((list[i], list[j]))

    return res

if __name__ == '__main__':
    sample = [1,3,2,2]
    k = 4
    
    res = pair_sum(sample, k)
    print(res)
