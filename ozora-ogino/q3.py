def findMissingValue(xs1: [list], xs2: [list]) -> int:
    if len(xs1) == len(xs2):
        raise Exception("Given two array is same length.")

    xs1, xs2 = (xs1, xs2) if len(xs1) > len(xs2) else (xs2, xs1)

    for n in xs2:
        if n in xs1:
            xs1.remove(n)
    return xs1


if __name__ == "__main__":
    xs1, xs2 = [2, 3, 4, 5, 6, 7, 5, 8], [6, 8, 7, 4, 5, 2, 3]
    print(findMissingValue(xs1, xs2))  # [5]

    xs1, xs2 = [2, 3, 4, 5, 6, 7, 5, 8], [6, 8, 4, 5, 2, 3]
    print(findMissingValue(xs1, xs2))  # [7, 5]

    xs1, xs2 = [1], [1]
    print(findMissingValue(xs1, xs2))  # Exeption
