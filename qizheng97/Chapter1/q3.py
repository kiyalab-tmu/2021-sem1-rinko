def misselement(lista,listb):
    if (len(lista)<len(listb)):
        lista,listb=listb,lista
    for i in lista:
        if not (i in listb):
            print(i)
            return i
        else:
            listb.remove(i)
    return 0

