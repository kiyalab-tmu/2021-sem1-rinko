def is_equal(xs1: [int], xs2: [int]) -> bool:
    if len(xs1) != len(xs2):
        return False

    for x1, x2 in zip(xs1, xs2):
        if type(x1) is not int:
            if not is_equal(x1, x2):
                return False
        else:
            if x1 != x2:
                return False
    return True


if __name__ == "__main__":
    xs1, xs2 = [1, 2, 4, 5, 65], [1, 2, 4, 5, 65]
    print(is_equal(xs1, xs2))  # True

    xs1, xs2 = [2, 4, 5, 65], [1, 4, 5, 65]
    print(is_equal(xs1, xs2))  # False

    xs1, xs2 = [1], [1, 2, 4, 5, 65]
    print(is_equal(xs1, xs2))  # False

    xs1, xs2 = [], []
    print(is_equal(xs1, xs2))  # True