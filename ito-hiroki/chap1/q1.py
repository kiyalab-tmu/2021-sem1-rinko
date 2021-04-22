"""
1~100までのFizzBuzz
"""

for i in range(1, 101):
    ans = ""
    if i % 3 == 0:
        ans += "Fizz"
    if i % 5 == 0:
        ans += "Buzz"
    print(i if ans == "" else ans)
