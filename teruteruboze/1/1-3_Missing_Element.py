import random

def find_missing_el(A, B):
    out = []
    for a in A:
        isExist = False
        for b in B:
            if a == b:
                isExist = True
                break
        if not isExist:
            return a
        

if __name__ == '__main__':
    A = list(range(2,7))
    B = list(range(2,7))
    random.shuffle(B)
    B.pop(random.randint(0, len(B)-1))
    print(A,B)
    print(find_missing_el(A,B))