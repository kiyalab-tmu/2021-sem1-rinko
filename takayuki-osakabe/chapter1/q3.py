import random

def missing(list1, list2):
    if len(list1) < len(list2):
        list1, list2 = list2, list1

    for i in list1:
        if not i in list2:
            return i

if __name__ == '__main__':
    sample = [0,1,2,3,4,5,6,7,8,9,10]

    list1 = random.sample(sample, len(sample))
    list2 = random.sample(sample, len(sample)-1)
    
    print(list1)
    print(list2)

    output = missing(list1, list2)

    print(output)
