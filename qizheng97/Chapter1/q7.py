import numpy as np
def q7():
    n=np.random.rand(5,5)
    mean=np.mean(n)
    std=np.std(n)
    return (n-mean)/std