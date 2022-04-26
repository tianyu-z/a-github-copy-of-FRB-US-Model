import symengine
import numpy

# For mypy typing
from typing import List, Callable, Tuple
from typing_extensions import Final


# Defined here, to be used in symbolic equations
# via CONST_SUPPORTED_FUNCTIONS_DICT
def ind_ltezero_symb(x):
    return symengine.Piecewise((0, x > 0), (1, True))


# This is the runnable version of the <= 0 indicator
def ind_ltezero(x):  # noqa: 401
    return 0 if x > 0 else 1


def varargs_max(*args):
    return numpy.max(list(args))


def varargs_min(*args):
    return numpy.min(list(args))


# The ONLY function keywords allowed in model equations
# Used in symbolic equation solving and differentiation, and model evaluation
# Defines the mapping of numpy name, sympy name, numpy function, symbolic constructor
# Any other identifier in the model is parsed as a variable name
CONST_SUPPORTED_FUNCTIONS_TUP: Final[List[Tuple[str, str, Callable, Callable]]] = [
    ("log", "log", numpy.log, symengine.log),
    ("exp", "exp", numpy.exp, symengine.exp),
    ("max", "Max", varargs_max, symengine.Max),
    ("min", "Min", varargs_min, symengine.Min),
    ("abs", "Abs", numpy.abs, symengine.Abs),
    ("ind_ltezero", "ind_ltezero_symb", ind_ltezero, ind_ltezero_symb),
]


CONST_SUPPORTED_FUNCTIONS_EX: Final[List[str]] = list(
    name for (name, _, _, _) in CONST_SUPPORTED_FUNCTIONS_TUP
)
CONST_SUPPORTED_FUNCTIONS_SYMB: Final[List[str]] = list(
    name for (_, name, _, _) in CONST_SUPPORTED_FUNCTIONS_TUP
)

# Declarations of supported functions
# `exec`d to be able to run Symengine/Sympy symbolics in the local environment
# Note: symbolic functions are given the same name as numpy functions here
# so that those tokens are recognized in model equations during symbolic procedures
CONST_SUPPORTED_FUNCTIONS_SYMB_DEC: Final[List[str]] = [
    # Special handling as symengine.lib is not accessible except via import
    f"{fun} = symengine.{call.__name__}"
    if "symengine.lib" in call.__module__
    else f"{fun} = {call.__module__}.{call.__name__}"
    for (fun, _, _, call) in CONST_SUPPORTED_FUNCTIONS_TUP
]

# Corresponding version for executable numeric functions
CONST_SUPPORTED_FUNCTIONS_EX_DEC: Final[List[str]] = [
    f"{fun} = {call.__module__}.{call.__name__}"
    if hasattr(call, "__module__")
    else f"{fun} = {call.__class__.__module__}.{call.__name__}"
    for (fun, _, call, _) in CONST_SUPPORTED_FUNCTIONS_TUP
]

# Verson where symbolic names are given to executable numeric functions
# For jacobian evaluation
CONST_SUPPORTED_FUNCTIONS_SYMEX_DEC: Final[List[str]] = [
    f"{fun} = {call.__module__}.{call.__name__}"
    if hasattr(call, "__module__")
    else f"{fun} = {call.__class__.__module__}.{call.__name__}"
    for (_, fun, call, _) in CONST_SUPPORTED_FUNCTIONS_TUP
]

# Options for loading mce equations
CONST_MCE_TYPES: Final[List[str]] = ["mcap", "wp", "mcap+wp", "all"]
