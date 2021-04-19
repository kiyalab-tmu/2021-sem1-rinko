"""
クイックソートの実装
ちょっとチート: http://www.ics.kagoshima-u.ac.jp/~fuchida/edu/algorithm/sort-algorithm/quick-sort.html
pivotでデータの先頭の要素を軸要素とする時は，partitionの後にinsertか何かでpivotの値を間に入れる処理が必要
"""


def pivot(a, i, j):
    k = i + 1
    while k <= j and a[i] == a[k]:
        k += 1
    if k > j:
        return -1
    if a[i] >= a[k]:
        return i
    return k


def partition(a, i, j, x):
    left = i
    right = j

    while left <= right:
        while left <= j and a[left] < x:
            left += 1
        while right >= i and a[right] >= x:
            right -= 1

        if left > right:
            break
        a[left], a[right] = a[right], a[left]
        left += 1
        right -= 1
    return left


def quick_sort(a, i, j):
    if i == j:
        return
    p = pivot(a, i, j)
    if p != -1:
        k = partition(a, i, j, a[p])
        quick_sort(a, i, k - 1)
        quick_sort(a, k, j)


if __name__ == "__main__":
    a = [9, 8, 7, 5, 6, 3, 1, 2, 4]
    quick_sort(a, 0, len(a) - 1)
    print(a)
