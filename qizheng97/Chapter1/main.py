import q1 as q1
import q2 as q2
import q3 as q3
import q4 as q4
import q5 as q5
import q7 as q7
import q8 as q8
import q9 as q9
import q10 as q10

import numpy as np
def main():
    #q1.fizzbuzz()

    #list=[3,7,8,5,2,1,9,5,4]
    #print(q2.quicksort(list,0,len(list)-1))


    #lista=[2,3,4,5,6,7,5,8]
    #listb=[6,8,7,4,5,2,3]
    #q3.misselement(lista,listb)

    #list=[1,3,2,2,5,8,6,-1]
    #q4.pairsum

    #q5.mTable(10)

    #q6.q6()

    #print(q7.q7())

    #m1=np.random.randint(10,size=(5,3))
    #m2=np.random.randint(10,size=(3,2))
    #m=q8.multiply(m1,m2)
    #print(m1)
    #print(m2)
    #print(m)
    #print(np.dot(m1,m2))

    #m=np.random.randint(10,size=(5,5))
    #print(q9.q9(m))

    m1=np.random.randint(2,size=(2,2))
    m2 = np.random.randint(2, size=(2, 2))
    print(m1,m2)
    print(q10.q10(m1,m2))

if __name__=="__main__":
   main()
