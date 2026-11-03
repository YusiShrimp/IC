"""Small snippet to raise an IndexError."""


def index_error(x):
    lst = list('foobar')
    print(lst[len(lst)])


index_error(0)
