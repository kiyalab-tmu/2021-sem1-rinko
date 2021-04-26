import collections


def finder(arr1, arr2):

    arr1.sort()
    arr2.sort()

    for num1, num2 in zip(arr1, arr2):
        if num1 != num2:
            return num1

    return arr1[-1]


def finder2(arr1, arr2):
    # avoid key errors
    d = collections.defaultdict(int)

    for num in arr2:
        d[num] += 1

    for num in arr1:
        if d[num] == 0:
            return num
        else:
            d[num] -= 1


print(finder2([5, 5, 7, 7], [5, 7, 7]))

# Clever trick, sum(arr1) - sum(arr2)


def finder3(arr1, arr2):
    # The cleverest trick, linear time and constant space complexity
    result = 0

    for num in (arr1 + arr2):
        result ^= num

    return result
