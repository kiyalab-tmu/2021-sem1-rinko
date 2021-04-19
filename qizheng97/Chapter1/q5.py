def mTable(max):
    for i in range(max):
        for j in range(max):
            if (j==0):
               print("%-2d%s" % ((i+1) , "  |   "), end='')
            print("%-5d" % ((i+1)*(j+1)), end='')
        print()
        if (i == 0):
            print("#####"*(max+1))



