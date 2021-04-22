if __name__ == '__main__':
    cells = []
    nrow, ncol = 20, 20
    for row_num in range(1,nrow+1):
        for col_num in range(1,ncol+1):
            cells.append(row_num * col_num)
    
    for row_num in range(nrow):
        if row_num == 1:
            li = '#' * len(li)
            print(li)
        
        li = str(row_num+1)
        if   len(li) == 2:
            li += ' |'
        elif len(li) == 1:
            li += '  |'

        for col_num in range(ncol):
            append_num = str(cells[(row_num * ncol) + col_num])
            if   len(append_num) == 3:
                li += ' '
            elif len(append_num) == 2:
                li += '  '
            elif len(append_num) == 1:
                li += '   '
            li += append_num

        print(li)