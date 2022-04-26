# For mypy typing
from typing import List, Callable

# Imports from this package
import pyfrbus.constants as constants

# Import modules so functions can be loaded from them
import numpy  # noqa: F401
import pyfrbus  # noqa: F401

# This module is technically where the model is executed, in fun_form
# so we have to import these functions for when the equations are eval'd
for declaration in constants.CONST_SUPPORTED_FUNCTIONS_EX_DEC:
    exec(declaration)


# Returns string in "functional form" with eval
# args a list of names for arguments:
# Usually x, the current guess,
# DataFrame.values argument data, to reference lags/exos,
# and z, a solution vector with length len(eqs)
def fun_form(xsub: List[str], args: List[str]) -> Callable:
    return eval("lambda " + ", ".join(args) + ": [" + ", ".join(xsub) + "]")
