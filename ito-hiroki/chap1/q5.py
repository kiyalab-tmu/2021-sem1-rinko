ans = "1  |"
for i in range(1, 11):
    ans += f"{i:4}"
print(ans)
print("#" * len(ans))

for i in range(2, 11):
    ans = ""
    for j in range(11):
        if j == 0:
            ans = f"{i:<3}|"
        else:
            ans += f"{i*j:4}"
    print(ans)
