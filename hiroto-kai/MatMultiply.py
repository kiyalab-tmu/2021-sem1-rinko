from random import randint
import numpy as np

def genMatrix(n_row, n_col):
	mat = [[randint(0,10) for x in range(n_col)] for y in range(n_row)] 
	return mat
	
def MatMul(matrix1, matrix2):
	result = np.zeros((len(matrix1), len(matrix2[0])), dtype=int)
	for i in range(len(matrix1)):
		for j in range(len(matrix2[0])):
			for k in range(len(matrix2)):
				result[i][j] += matrix1[i][k] * matrix2[k][j]
	return result

if __name__ == "__main__":
	mat1 = genMatrix(3,3)
	mat2 = genMatrix(3,5)
	print(mat1, mat2)
	result = MatMul(mat1, mat2)
	result2 = np.matmul(mat1, mat2)
	print("{}\n{}".format(result, result2))
	
	
	
	