def Quick_Sort(arr):
    left = []
    right = []
    if len(arr) <= 1:
        return arr
    
    #配列の先頭を基準とする
    pivot = arr[0]
    cnt = 0

    for ele in arr:
        if ele < pivot:
            left.append(ele)
        elif ele > pivot:
            right.append(ele)
        else:
            cnt += 1
            
    left = Quick_Sort(left)
    right = Quick_Sort(right)
    
    return left + [pivot] * cnt + right

if __name__ == '__main__':
    sample_input = [9,8,7,5,6,3,1,2,4]
    print('Input=',sample_input)
    print('Output=',Quick_Sort(sample_input))   
    