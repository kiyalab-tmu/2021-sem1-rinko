import numpy as np
np.set_printoptions(precision=2, floatmode='maxprec_equal')

if __name__ == '__main__':
    matrix = np.random.rand(5, 5)
    ans_minmax = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    ans_zscore = (matrix - matrix.mean()) / matrix.std()
    print('Input:\n',ans_minmax) 
    print('Min-max:\n',ans_minmax)   
    print('Z-score:\n',ans_zscore)