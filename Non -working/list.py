def square(x):
    print(x)
    print('Calculating square of', x)
    return x * x

lst = [1, 2, 3]
list(map(square, lst))
# res = map(square, lst)
# res