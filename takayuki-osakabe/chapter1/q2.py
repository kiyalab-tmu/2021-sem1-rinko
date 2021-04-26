import random

def quick_sort(list):
    if len(list) <= 1:
        return list

    left = []
    right = []

    pivot = list[0]
    count = 0

    for num in list:
        if num < pivot:
            left.append(num)
        elif num > pivot:
            right.append(num)
        else:
            count += 1

    left = quick_sort(left)
    right = quick_sort(right)

    return left + [pivot]*count + right

if __name__ == '__main__':
    list = [0,1,2,3,4,5,6,7,8,9,10]
    input = random.sample(list, len(list))
    print(input)

    output = quick_sort(input)

    print(output)
