def fizzbuzz():
    for i in range(100):
        k=i+1
        if (k % 3 == 0) and (k % 5 == 0):
            print("FizzBuzz")
            continue
        if (k%3==0):
            print("Fizz")
            continue
        if (k%5==0):
            print("Buzz")
            continue
        print(k)

