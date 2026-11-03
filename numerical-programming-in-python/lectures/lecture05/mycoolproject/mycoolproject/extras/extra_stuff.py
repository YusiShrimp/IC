"""This is a module in the utils package. It contains utility functions."""
import numpy as np

__all__ = ['extra_func1', 'extra_func2']

def extra_func1():
    """This is a utility function in utils1. It uses a numpy function."""
    print('util_func1 in utils1', np.random.rand())
    return 1


def extra_func2():
    """This is another utility function in utils1."""
    print('util_func2 in utils1')
    return 2
