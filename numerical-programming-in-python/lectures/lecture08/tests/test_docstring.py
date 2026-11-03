def mean(x):
    """Mean of a list of numbers.

    >>> mean([1, 5, 9])
    5.0

    >>> X = [1, 2, 3]
    >>> mean(X)
    2.0

    """
    return sum(x) / (len(x))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
