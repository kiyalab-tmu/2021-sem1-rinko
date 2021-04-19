def findMissingElement(A,B):
  if len(A) > len(B):
    Z_long, Z_short = A, B
  else:
    Z_long, Z_short = B, A
  
  for val in Z_short:
    Z_long.remove(val)
  
  return Z_long

sample_input1, sample_input2 = [2, 3, 4, 5, 6, 7, 5, 8], [6, 8, 7, 4, 5, 2, 3]
print(findMissingElement(sample_input1,sample_input2))