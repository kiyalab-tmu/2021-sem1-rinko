def quick_sort(data):
    left  = []
    right = []

    num = data[0]
    cnt = 0

    for li in data:
        if   li < num:
            left.append(li)
        elif li > num:
            right.append(li)
        else:
            cnt += 1
    left = quick_sort(left)
    right = quick_sort(right)
    return left + [num] * cnt + righ

if __name__ == "__main__":
    data = list(range(1,10))