def quicksort(list,low,high):
    if (low<high):
       list,p=partation(list,low,high)
       print(list, low, high,p)
       list=quicksort(list,low,p-1)
       list=quicksort(list,p+1,high)
    return list
def partation(list,low,high):
    pivot=list[high]
    i=low
    for j in range(low,high+1):
        if (list[j]<pivot):
            list[i],list[j]=list[j],list[i]
            i=i+1
    list[i],list[high]=list[high],list[i]
    return list,i

