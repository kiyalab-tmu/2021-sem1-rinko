import numpy as np
def q9(m):
    m[(m>3) & (m<8)]*=-1
    return m