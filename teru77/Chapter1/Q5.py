def Multiplication_Table(max):
    for i in range(1,max):
        print("{0:4d}   |".format(i),end='')
        for j in range(1,max):
            print('{0:4d}'.format(i*j),end='')
        print()
        
        if i == 1:
            print('   #'+'####'*max)
            
if __name__ == '__main__':
    Multiplication_Table(11)
