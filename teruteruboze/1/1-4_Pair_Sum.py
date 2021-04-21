import random

def pair_sum(data, k):
    for i in range(len(data)//2):
        for j in range(len(data)):
            if i == j:
                pass
            else:
                if data[i] + data[j] == k:
                    print('(', data[i], ',', data[j], ')')

if __name__ == '__main__':
    data = []
    for i in range(4):
        data.append(random.randint(1,10))
    k = random.randint(2,15)
    print(data, k)
    pair_sum(data, k)