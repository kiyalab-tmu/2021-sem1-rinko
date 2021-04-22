def quickSort(xs: [int], begin: int = 0, end: int = None):
    if end is None:
        end = len(xs) - 1

    def _quickSort(xs: [int], begin: int, end: int):
        if begin >= end:
            return
        pivot = partition(xs, begin, end)
        _quickSort(xs, begin, pivot - 1)
        _quickSort(xs, pivot + 1, end)

    return _quickSort(xs, begin, end)


def partition(xs: [list], begin: int, end: int) -> int:
    pivot = begin
    for i in range(begin + 1, end + 1):
        if xs[i] <= xs[begin]:
            pivot += 1
            xs[i], xs[pivot] = xs[pivot], xs[i]
    xs[pivot], xs[begin] = xs[begin], xs[pivot]
    return pivot


if __name__ == "__main__":
    xs = [12, 11, 3, 78, 2, 4, 6, 1]
    quickSort(xs, 0, len(xs) - 1)
    print(xs)