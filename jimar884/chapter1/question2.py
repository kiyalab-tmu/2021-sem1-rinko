def quicksort(A, lo, hi):
  if lo < hi:
    p = partition(A, lo, hi)
    quicksort(A, lo, p - 1)
    quicksort(A, p + 1, hi)

def partition(A, lo, hi):
  pivot = A[hi]
  i = lo
  for j in range(lo, hi):
    if A[j] < pivot:
      A[i], A[j] = A[j], A[i]
      i += 1
  A[i], A[hi] = A[hi], A[i]
  return i

sample_input = [9, 8, 7, 5, 6, 3, 1, 2, 4]
print(sample_input)
quicksort(sample_input, 0, len(sample_input) - 1)
print(sample_input)