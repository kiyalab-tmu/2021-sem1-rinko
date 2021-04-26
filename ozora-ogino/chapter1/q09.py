def negate(xs: [int], min_val: int = 3, max_val: int = 8) -> [list]:
    for i in range(len(xs)):
        if xs[i] <= max_val and xs[i] >= min_val:
            xs[i] *= -1


if __name__ == "__main__":
    xs = [1, 3, 6, 7, 122, 11, 3, 44]
    negate(xs)
    print(xs)
