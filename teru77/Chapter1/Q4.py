def Pair_Sum(arr,k):
    pairs = []
    cnt = 0
    for i in arr:       
        for j in arr[cnt+1:]:
            if i + j == k:
                pairs.append((i,j))
        cnt += 1
    return pairs
            
if __name__ == '__main__':
    sample_input = [1, 3, 2, 2]
    k = 4
    print(Pair_Sum(sample_input,k))