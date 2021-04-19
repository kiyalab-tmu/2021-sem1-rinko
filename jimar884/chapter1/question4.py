def findPairSum(A, k):
  A.sort()
  answers = []

  for i in range(0, len(sample_input)):
    for j in range(i + 1, len(sample_input)):
      if sample_input[i]+sample_input[j] == k and not([sample_input[i], sample_input[j]] in answers):
        answers.append([sample_input[i], sample_input[j]])
        break
    
  
  return  answers

sample_input = [1, 3, 2, 2]
k = 4

print(findPairSum(sample_input, k))