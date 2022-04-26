import numpy
from scipy.sparse import csr_matrix

# For mypy typing
from typing import List, Tuple, Callable, Union, Optional
from numpy import ndarray

# Imports from this package
import pyfrbus.constants as constants

# Import modules so functions can be loaded from them
import pyfrbus  # noqa: F401


# Globally define Sympy Min, Max, etc. and Heaviside, and Piecewise as numpy functions
# Used later when we eval elements of the Jacobian into lambdas
for declaration in constants.CONST_SUPPORTED_FUNCTIONS_SYMEX_DEC:
    exec(declaration)

Heaviside = lambda x: numpy.heaviside(x, 0)  # noqa: E731, F841


def Piecewise(*args):
    for (val, cond) in args:
        if cond:
            return val
    return numpy.nan


def jac_2_callable(jac: List[Tuple[int, int, str]]) -> List[Tuple[int, int, Callable]]:
    new_jac: List[Tuple[int, int, Callable]] = []
    for entry in jac:
        new_jac += [(entry[0], entry[1], eval("lambda x, data, z: " + entry[2]))]
    return new_jac


# Returns function to eval string version of jacobian
def eval_jac(
    jac: List[Tuple[int, int, str]], size: int, sparse: bool
) -> Callable[[ndarray, ndarray, Optional[ndarray]], Union[ndarray, csr_matrix]]:
    # Returned function takes two or three arguments
    # x, guess; data, lag/exo dataset; z, partial solution for previous blocks
    fun_jac = jac_2_callable(jac)

    if sparse:

        def e_j_sparse(
            x: ndarray, data: ndarray, z: Optional[ndarray] = None
        ) -> csr_matrix:
            row = []
            col = []
            mat = []
            for entry in fun_jac:
                row.append(entry[0])
                col.append(entry[1])
                mat.append(entry[2](x, data, z))
            return csr_matrix((mat, (row, col)), shape=(size, size))

        return e_j_sparse

    else:

        def e_j(x: ndarray, data: ndarray, z: Optional[ndarray] = None) -> ndarray:
            mat = numpy.zeros((size, size))
            for entry in fun_jac:
                mat[entry[0], entry[1]] = entry[2](x, data, z)
            return mat

        return e_j
