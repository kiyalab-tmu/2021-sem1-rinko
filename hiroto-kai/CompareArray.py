from random import randint
import numpy as np

def genMatrix(n_row, n_col):
	mat = [[randint(0,10) for x in range(n_col)] for y in range(n_row)] 
	return mat

def CompareMatrix(mat1, mat2):
	for i in range(len(mat1)):
		for j in range(len(mat1[0])):
			if mat1[i][j] != mat2[i][j]:
				return False
				break;
			break;
	return True;
	
if __name__ == "__main__":
	mat1 = genMatrix(2,5)
	mat2 = genMatrix(2,5)
	if CompareMatrix(mat1, mat1):
		print("Equal")
	else:
		print("Not Equal")
	