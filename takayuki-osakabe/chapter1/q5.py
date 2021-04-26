def multitable():
    for i in range(1,11):
        for j in range(1,11):
            if j == 1:
                print(str(i).ljust(3), end='')
                print('|', end='')
            if i == 1:
                print(str(j).rjust(4), end='')
            else:
                print(str(i*j).rjust(4), end='')
        if i == 1:
            print('\n')
            print('#'*44, end='')
        print('\n')

if __name__ == '__main__':
    multitable()
