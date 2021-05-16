def fizz_buzz():
    for i in range(0, 101):
        s = ""
        if i % 3 == 0:
            s += "Fizz"
        if i % 5 == 0:
            s += "Buzz"
        print(s) if len(s) else print(i)


fizz_buzz()


# python -c "print '\n'.join(['Fizz'*(x % 3 == 2) + 'Buzz'*(x % 5 == 4) or str(x + 1) for x in range(100)])"
# python -c "print '\n'.join(['Fizz'*(x % 3 == 0) + 'Buzz'*(x % 5 == 0) or str(x) for x in range(1, 101)])"
