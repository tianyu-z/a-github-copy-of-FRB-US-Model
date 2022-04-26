import sympy
import symengine
import re

# For mypy typing
from typing import List, Dict, Tuple, Optional
from symengine.lib.symengine_wrapper import Expr, Symbol

# Imports from this package
from pyfrbus.lib import sub_dict_or_raise, invert_dict, join_escape_regex
import pyfrbus.constants as constants

# Imported so that symbolic ind_ltezero will be loaded properly from constants
# symengine is imported above so Max, etc. are already handled
import pyfrbus  # noqa: F401


# Converts a list of equations in terms of x[n], data[-i, j] into symengine expressions
def to_symengine_expr(xsub: List[str]) -> Tuple[List[Expr], Dict[str, str]]:
    # Set up symengine symbols
    # First, load supported functions
    # We assign the supported (numpy) functions used in the equations
    # as their symbolic counterparts, so they are recognized by symengine
    # for solving and differentiation
    for declaration in constants.CONST_SUPPORTED_FUNCTIONS_SYMB_DEC:
        exec(declaration)

    # Set up symbols corresponding to the x vector argument
    x = symengine.symbols([f"x[{i}]" for i in range(len(xsub))])  # noqa: F841

    # SymEngine doesn't yet support IndexedBase from SymPy
    # so we have to do something else
    data_hash: Dict[str, str] = data_to_vars("".join(xsub))
    inv_data_hash: Dict[str, str] = invert_dict(data_hash)
    symengine_xsub = sub_dict_or_raise(xsub, r"(data\[-\d+,\d+\])", inv_data_hash)
    data = symengine.symbols(list(data_hash.keys()))  # noqa: F841

    eqs: List = []
    for eq in symengine_xsub:
        eqs += [eval(eq)]
    return (eqs, data_hash)


# Take a list of equations in terms of x[j]'s and solve each for corresponding x[i]
def factor_out_xi(
    xsub: List[str], exprs: List, data_hash: Dict[str, str], skip: List[bool]
) -> List[Optional[str]]:

    # Output list of solved equations
    nox_xsub: List[Optional[str]] = []

    for i in range(len(xsub)):
        eq = xsub[i]
        # Skip equations that appear in simultaneous blocks
        # They will never be evaluated in a backwards-looking block
        if skip[i]:
            nox_xsub += [None]
            continue

        # Regex heuristics to handle most common cases in the XML,
        # since symbolic solver is slow
        just_x_regex = rf"\((x\[{i}\])-data\[-\d+,\d+\]\)"
        log_x_regex = rf"\((log\(x\[{i}\]\))-data\[-\d+,\d+\]\)"
        dlog_x_regex = (
            rf"\(\((log\(x\[{i}\]\))-\(?log\(data\[-\d+,\d+\]\)\)?\)-data\[-\d+,\d+\]\)"
        )

        # First, check for a LHS of (var - var_trac)
        match = re.findall(just_x_regex, eq)
        if match:
            nox_xsub += [eq.replace(match[0], "0")]
            continue

        # Then, check for (log(var) - var_trac)
        match = re.findall(log_x_regex, eq)
        if match:
            nox_xsub += ["".join(["exp(", eq.replace(match[0], "0"), ")"])]
            continue

        # Then, check for (dlog(var) - var_trac)
        match = re.findall(dlog_x_regex, eq)
        if match:
            nox_xsub += ["".join(["exp(", eq.replace(match[0], "0"), ")"])]
            continue

        # If all fail, use sympy solver
        # SymEngine doesn't support symbolic equation solving like this, yet
        nox_xsub += [
            # Flip data mapping and turn output symbolic functions back to numpy
            symengine2numpy(
                fix_symengine_data(
                    str(
                        sympy.solve(
                            exprs[i],
                            sympy.Symbol(f"x[{i}]"),
                            check=False,
                            numerical=False,
                            minimal=True,
                            simplify=False,
                            rational=False,
                        )[0]
                    ),
                    data_hash,
                )
            )
        ]

    return nox_xsub


# Use SymEngine to take the partial d eq / d w_resp_to
def take_symengine_partial(eq: Expr, w_resp_to: Symbol, data_hash: Dict[str, str]):
    # Attempt to take derivative
    deriv = str(eq.diff(w_resp_to))
    # SymEngine fails to recognize derivative of Max as Heaviside
    # So failover to Sympy
    if re.findall("Derivative", deriv):
        deriv = str(sympy.diff(eq, w_resp_to))
        # Once more: handling for abs
        # symengine.symbols doesn't let you pass real=True to fix the abs derivative
        # So we make a sympy symbol instead
        if re.findall("Derivative", deriv):
            deriv = str(sympy.diff(eq, sympy.symbols(str(w_resp_to), real=True)))

    # Reverse the data[-i,j] -> data[k] mapping used for SymEngine
    deriv = fix_symengine_data(deriv, data_hash)
    return deriv


# Reverse the data[-i,j] -> data[k] mapping used for SymEngine
def fix_symengine_data(eq: str, data_hash: Dict[str, str]) -> str:
    return sub_dict_or_raise([eq], r"(data\[\d+\])", data_hash)[0]


# Replace array brackets for min/max, and turn capitals to lowercase
# For post-processing symbolic output back to executable expressions
def symengine2numpy(eq: str) -> str:
    # Turn sympy functions into numpy functions, e.g. Abs -> abs
    repls = {
        sympy_name: numpy_name
        for (numpy_name, sympy_name, _, _) in constants.CONST_SUPPORTED_FUNCTIONS_TUP
    }
    eq = sub_dict_or_raise([eq], join_escape_regex(repls), repls)[0]
    return eq


# Get mapping of data[k] from data[-i,j], used in SymEngine expressions
def data_to_vars(cat_eqs: str) -> Dict[str, str]:
    # Get all references to the data frame as data[-i,j]
    data_refs = list(set(re.findall(r"data\[-\d+,\d+\]", cat_eqs)))
    # Map those references to elements of a 1D "data" array
    # Returned hash is data[k] => data[-i,j]
    data_hash = {f"data[{i}]": data_refs[i] for i in range(len(data_refs))}
    return data_hash
