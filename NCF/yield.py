import itertools


def get():
    ls = ([1,2,3], [4,5,6], [7,8])
    print('h1')
    for x in itertools.product(*ls):
        print('h2')
        yield x
        print('h3')


for x in get():
    print(x)
