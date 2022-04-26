import symengine
import re
from collections import defaultdict

# For mypy typing
from typing import List, Tuple, Callable, Set, Dict
from symengine.lib.symengine_wrapper import Expr

# Imports from this package
from pyfrbus.lib import sub_dict_or_raise, join_escape_regex, idx_dict
from pyfrbus.symbolic import take_symengine_partial
from pyfrbus.equations import add_offset, map_data_ij


# Create Jacobian
# Takes filled, x-subbed eqs and returns the jacobian
# as a sparse matrix [i, j, di/dj]
def create_jacobian(
    n_eqs: int, rhs_vars: List[Set[str]], exprs: List[Expr], data_hash: Dict[str, str]
) -> List[Tuple[int, int, str]]:

    # Output is a List of triples, [i, j, partial]
    jac_list: List[Tuple[int, int, str]] = []

    for i in range(len(exprs)):
        eq = exprs[i]

        # Partial of x[i] with respect to itself
        deriv = take_symengine_partial(eq, symengine.symbols(f"x[{i}]"), data_hash)
        jac_list += [(i, i, deriv)]

        # Compute other partials for vars on RHS
        for var in rhs_vars[i]:
            j = int(re.findall(r"\d+", var)[0])
            deriv = take_symengine_partial(eq, symengine.symbols(var), data_hash)
            jac_list += [(i, j, deriv)]
    return jac_list


# Contains logic to speed up large stacked-time jacobians
# by exploiting the repetitive structure to avoid duplicating
# expensive symbolic differentiation
def mce_create_jacobian(
    n_eqs: int,
    rhs_vars: List[Set[str]],
    exprs: List[Expr],
    data_hash: Dict[str, str],
    endo_names: List[str],
    data_varnames: List[str],
) -> List[Tuple[int, int, str]]:

    # Output is a List of triples, [i, j, partial]
    jac_list: List[Tuple[int, int, str]] = []
    # To look up specific partials by (i,j)
    jac_dict: Dict[Tuple[int, int], str] = {}

    # The variable offset, i.e. number of endo vars/eqs from x_0 to x_1
    var_offset = endo_names.index(f"{endo_names[0]}_1")
    # The data offset, i.e. number of total data vars
    data_offset = data_varnames.index(f"{data_varnames[0]}_1")
    # The period offset, i.e. number of periods for which each column is duplicated
    period_offset = data_varnames.index(f"{data_varnames[1]}_1") - data_offset

    offsets: Tuple[int, int, int] = (var_offset, data_offset, period_offset)

    # Used for reverse lookups when converting a -1 lag to an endo
    endo_idx_dict: Dict[str, int] = idx_dict(endo_names)
    # The dict we need turns a data column index into an endo vector index
    col_to_endo: Dict[int, int] = {
        j: endo_idx_dict[data_varnames[j]]
        for j in filter(
            lambda j: data_varnames[j] in endo_idx_dict, range(0, len(data_varnames))
        )
    }

    for i in range(len(exprs)):
        # Equation for x[i]
        # Equations are in endo variable order
        eq = exprs[i]

        # Partial of x[i] with respect to itself
        deriv = take_partial(i, i, eq, data_hash, jac_dict, offsets, col_to_endo)
        jac_list += [(i, i, deriv)]
        jac_dict[(i, i)] = deriv

        # Compute other partials for vars on RHS
        for var in rhs_vars[i]:
            j = int(re.findall(r"\d+", var)[0])
            deriv = take_partial(i, j, eq, data_hash, jac_dict, offsets, col_to_endo)
            jac_list += [(i, j, deriv)]
            jac_dict[(i, j)] = deriv

    return jac_list


# Logic that determines whether we get the partial by substitution
# or through symbolic computation
def take_partial(
    i: int,
    j: int,
    eq: Expr,
    data_hash: Dict[str, str],
    jac_dict: Dict[Tuple[int, int], str],
    offsets: Tuple[int, int, int],
    col_to_endo: Dict[int, int],
) -> str:
    # If both are leads, we can potentially substitute
    # var_offset == the number of time-zero endogenous variables,
    # So if x[i] shows up after that, it must be a lead
    if i > offsets[0] and j > offsets[0]:
        deriv = compute_from_reference(i, j, jac_dict, offsets, col_to_endo)
    else:
        # Otherwise, take partial the standard way
        deriv = take_symengine_partial(eq, symengine.symbols(f"x[{j}]"), data_hash)

    return deriv


# Use substitution to get dy_i/dx_j from dy_(i-1)/d_x_(i-1)
def compute_from_reference(
    i: int,
    j: int,
    jac_dict: Dict[Tuple[int, int], str],
    offsets: Tuple[int, int, int],
    col_to_endo: Dict[int, int],
) -> str:
    ref_deriv = get_ref_deriv(i, j, jac_dict, offsets[0])
    # Both x[i] and x[j] are leads and we can substitute
    new_partial = add_offset(ref_deriv, offsets[0])

    # Reference derivative contains a data[n] reference, which is a lag or exo
    if "data" in ref_deriv:
        # Now, handle data[n] terms
        # Each vector element data[n] -> data[-i,j] frame element in data_hash
        # If i=1, it is a contemporaneous or forward exo -
        # Map it to the next period lead.
        # If i>1, it is an endo lag or exo lag
        # exos mapped to next period
        # endos mapped to next period (i>2) OR an endo x[n] (i=2)
        return map_data_ij(new_partial, offsets[1], offsets[2], col_to_endo)
    else:
        return new_partial


def get_ref_deriv(
    i: int, j: int, jac_dict: Dict[Tuple[int, int], str], var_offset: int
) -> str:
    # Leads x[i] x[j] correspond to variables x_m and x_n
    # The reference partial dx_(m-1)/dy_(n-1) is obtained
    # by skipping backwards one full generation of duplicated variables
    # using the offset
    return jac_dict[(i - var_offset, j - var_offset)]


# Pull entries in block, renumber indices,
# and call "replace" function on each Jacobian entry string
# "replace" used in jacobian_blocks to switch some x[i] references to z[i]'s
# AKA elements of the partial solution
def subset_jacobian(
    jac: List[Tuple[int, int, str]],
    block_dict: Dict[int, int],
    replace: Callable[[str], str],
) -> List[Tuple[int, int, str]]:
    return [
        (block_dict[entry[0]], block_dict[entry[1]], replace(entry[2]))
        for entry in jac
        if ((block_dict[entry[0]] > -1) and (block_dict[entry[1]] > -1))
    ]


# Like block_partials, we want Jacobian submatrices for each block
# with an argument for partial solution of previous blocks
def jacobian_blocks(
    jac: List[Tuple[int, int, str]],
    block: List[int],
    prev_blocks: List[List[int]],
    single_block: bool = False,
) -> List[Tuple[int, int, str]]:

    # Skip if single block
    if single_block:
        return jac

    flat_prev_blocks: List[int] = sum(prev_blocks, [])

    # Take principal submatrix of Jacobian corresponding to block
    # Replace references to x[i]'s from previous blocks with z[i]'s
    # and renumber current x[i]'s
    keys = [f"x[{real_idx}]" for real_idx in flat_prev_blocks] + [
        f"x[{block[j]}]" for j in range(len(block))
    ]
    values = [f"z[{real_idx}]" for real_idx in flat_prev_blocks] + [
        f"x[{j}]" for j in range(len(block))
    ]

    repls = dict(zip(keys, values))
    # Build a regex that matches they keys
    regex = join_escape_regex(repls)

    def replace_in_entry(jac_entry: str) -> str:
        return sub_dict_or_raise([jac_entry], regex, repls)[0]

    block_dict = defaultdict(lambda: -1, zip(block, range(0, len(block))))
    return subset_jacobian(jac, block_dict, replace_in_entry)
