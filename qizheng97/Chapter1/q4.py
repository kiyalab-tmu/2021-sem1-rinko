def pairsum(list,key):
    while list:
        if (key-list[0] in list):
            print(list[0],key-list[0])
            list.remove(key - list[0])
            list.remove(list[0])
        else:
            list.remove(list[0])
    return



