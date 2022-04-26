import re


# For mypy typing
from typing import List, Dict, Set, Optional, Tuple, cast

# Imports from this package
from pyfrbus.lib import remove, flatten, unzip, sub_dict_or_raise
import pyfrbus.lexing as lexing
from pyfrbus.exceptions import MissingDataError, InvalidModelError


# Gives list of right-hand side variables for each equation
# NOT including that endogenous variable
def rhs_vars(xsub: List[str]) -> List[Set[str]]:
    return [
        set(remove(re.findall(r"x\[\d+\]", xsub[i]), f"x[{i}]"))
        for i in range(len(xsub))
    ]


def has_leads(lexed_eqs: List[List[Tuple[str, Optional[Tuple[str, int]]]]]) -> bool:
    for eq in lexed_eqs:
        # Take variable identifiers out of lexed equations
        # Last one is always None, so we drop it
        # and cast to satisfy type checker
        tokens = cast(List[Tuple[str, int]], unzip(eq)[1][0:-1])
        for token in tokens:
            # Check if token represents a lead
            if token[1] > 0:
                return True
    return False


# Returns duplicated variables, needed to solve stacked time system
# for n_periods time steps
# Ordered e.g. ([a,b,c], 3) -> [a_1,b_1,c_1,a_2,b_2,c_2]
def dupe_vars(var_list: List[str], n_periods: int) -> List[str]:
    return [f"{var}_{i}" for i in range(1, n_periods) for var in var_list]


# Returns duplicated equations for stacked time system
def dupe_eqs(
    lexed_eqs: List[List[Tuple[str, Optional[Tuple[str, int]]]]], n_periods: int
) -> List[List[Tuple[str, Optional[Tuple[str, int]]]]]:
    # Apply shift_eq for each equation, in each period
    return flatten(
        [[lexing.shift_eq(eq, i) for eq in lexed_eqs] for i in range(1, n_periods)]
    )


def clean_eq(eq: str) -> str:
    return re.sub(r"\s+", "", eq).strip()


# Given an equation with an equals sign, subtracts LHS from RHS
# giving an expression which = 0
def flip_equals(eq: str) -> str:
    halves: List[str] = eq.split("=")
    return halves[1] + "-(" + halves[0] + ")"


# Replace constants in equation with their values
def fill_constants(eqs: List[str], constants: Dict[str, float]) -> List[str]:
    repls: Dict[str, str] = {k: str(v) for k, v in constants.items()}
    try:
        return sub_dict_or_raise(eqs, r"\by_\w+_\d+\b", repls)
    except KeyError as err:
        raise InvalidModelError(f"Constant not found: {err.args[0]}") from None


# Replace lags and exogenous variables with data frame indexes
# Replace simultaneous endos with references to single vector "x"
def fill_lags_and_exos_xsub(
    lexed_eqs: List[List[Tuple[str, Optional[Tuple[str, int]]]]],
    data_idxs: Dict[str, int],
    exo_names: List[str],
    endo_names: List[str],
) -> List[str]:

    # If we fail to find data for a variable, raise a more descriptive error message
    try:
        lag_idxs = [data_idxs[name] for name in endo_names]
        exo_idxs = [data_idxs[name] for name in exo_names]
    except KeyError as err:
        raise MissingDataError(err.args[0]) from None

    endo_idx_dict = dict(zip(endo_names, range(0, len(endo_names))))
    lag_exo_idx_dict = dict(zip(endo_names + exo_names, lag_idxs + exo_idxs))
    is_exo = dict(
        zip(endo_names + exo_names, [False] * len(endo_names) + [True] * len(exo_names))
    )

    return [
        lexing.xsub(eq, is_exo, lag_exo_idx_dict, endo_idx_dict) for eq in lexed_eqs
    ]


# Replace all x[i] with x[i+offset]
# For use in MCE Jacobian computation, where we find dx_i/dy_j
# by substituting from dx_(i-1)/dy_(j-1)
def add_offset(ref_deriv: str, var_offset: int) -> str:
    regex = r"x\[(\d+)\]"
    return re.sub(regex, lambda mobj: f"x[{int(mobj.group(1))+var_offset}]", ref_deriv)


# Moves all data[-i,j] terms in an equation into the next period
def map_data_ij(
    ref_deriv: str, data_offset: int, period_offset: int, col_to_endo: Dict[int, int]
) -> str:
    regex = r"data\[-(\d+),(\d+)\]"
    # Use lead_data_ij to handle logic of turning data[-i,j] into either
    # the next period lag/exo data[-i,j+n] or the next period endo x[k]
    return re.sub(
        regex,
        lambda mobj: lead_data_ij(
            int(mobj.group(1)),
            int(mobj.group(2)),
            data_offset,
            period_offset,
            col_to_endo,
        ),
        ref_deriv,
    )


# Moves a single data[-i,j] term into the next period
def lead_data_ij(
    i: int, j: int, data_offset: int, period_offset: int, col_to_endo: Dict[int, int]
) -> str:
    # If i=1, it is a contemporaneous or forward exo -
    # Map it to the next period lead.
    if i == 1:
        # If it is time-zero, we find the first lead
        # data_offset is the number of columns corresponding to time zero
        # period_offset*j is all previous variables duplicated for all periods
        # Add them together to get to the first lead of the j'th variable
        if j < data_offset:
            return f"data[-1,{data_offset+period_offset*j}]"
        # Otherwise just add n_periods as columns are duplicated in order, per period
        else:
            return f"data[-1,{j+1}]"
    # If i>1, it is an endo lag or exo lag
    # exos mapped to next period
    # endos mapped to next period (i>2) OR an endo x[n] (i=2)
    else:  # i >= 2
        if i > 2 or j not in col_to_endo:
            return f"data[-{i-1},{j}]"
        else:  # endo, i=2, j corresponds to a time-zero variable
            # First, we get the name of the variable for the corresponding column
            # Then, we look up its index in the x vector, which is the same as
            # its index in frbus.endo_names
            return f"x[{col_to_endo[j]}]"


# Convert an endo variable name into its corresponding error term
def endo_to_trac(endo: str) -> str:
    match = re.search(r"(.*?)(_\d+)", endo)
    if not match:
        return endo + "_trac"
    else:
        # Turns ex. rff_1 to rff_trac_1
        return f"{match[1]}_trac{match[2]}"
