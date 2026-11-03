"""
This is the package docstring.

We import everything from mod1 and utils, and only func2 from mod2
Note, however, than because we used __all__ in mod1, there are
some functions that are not imported here.

Note that I use the noqa comment to avoid flake8 warnings about
* and unused imports.
"""

from .cool_module import *
from .another_cool_module import *
from .extras import *
