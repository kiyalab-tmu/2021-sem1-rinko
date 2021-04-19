def findPairSum(xs: [list], k: int):
    pairs = []
    i = 0
    while i < len(xs):
        if k - xs[i] in xs[i + 1 :]:
            pairs.append(((xs[i], k - xs[i]),))
        i += 1
    return pairs


if __name__ == "__main__":
    xs = [1, 3, 2, 2]
    k = 4
    print(findPairSum(xs, k))  # [(1, 3), (2, 2)]

    xs = [1, 2, 4, 5, 4]
    k = 1
    print(findPairSum(xs, k))  # []
