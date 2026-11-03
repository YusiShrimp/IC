"""
This is the mod1 module.

We use the special variable __all__ to specify which functions are imported
when we use the following import statement in the package __init__.py file:

from .mod1 import *

Note that this does not import the secret_func function, because it is not
included in __all__.
"""

__all__ = ['func1']

def func1():
    """This is func1 in mod1."""
    print('func1 in mod1')
    return 1


def secret_func():
    """
    This is a secret function in mod1, it is not imported when
    import * is used.
    """
    print('secret_func in mod1')
    return 1
