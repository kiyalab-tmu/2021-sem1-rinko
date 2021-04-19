def write_num(n):
  print(str(n).ljust(3) + "|", end='')
  for i in range(1,10+1):
    print("{0:4d}".format(n*i), end='')
  print("")

write_num(1)
print("#"*44)
for i in range(2, 10+1):
  write_num(i)