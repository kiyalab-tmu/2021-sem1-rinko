import random

def quick_sort(data):
    left  = []
    right = []

    if len(data) <= 1:
        return data

    num = data[0]
    cnt = 0

    for li in data:
        if   li < num:
            left.append(li)
        elif li > num:
            right.append(li)
        else:
            cnt += 1
    left  = quick_sort(left)
    right = quick_sort(right)
    return left + [num] * cnt + right

if __name__ == '__main__':
    input_data = list(range(1,10))
    random.shuffle(input_data)
    sort_data = quick_sort(input_data)
    print('Before:', input_data)
    print('After: ', sort_data)