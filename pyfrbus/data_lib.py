import re

# For mypy typing
from typing import List, Dict, Tuple
from pandas import DataFrame, Period, PeriodIndex
from numpy import ndarray

# Imports from this package
from pyfrbus.lib import idx_dict, np2df, get_periods_idxs, unzip, flatten


# Delete _n duplicated columns used in MCE solution, before data frame is given to user
def drop_mce_vars(data: DataFrame) -> DataFrame:
    aux_vars = [name for name in data.columns.values if re.search(r"_\d+", name)]
    return data.drop(aux_vars, axis=1)


# Copy single-period values from MCE solution into current-period variables
# Optimized for speed
# Note: varlist must be non-lead variables!
def copy_fwd_to_current(
    data: DataFrame, varlist: List[str], periods: PeriodIndex
) -> DataFrame:

    # Convert periods into indices in numpy arrays
    periods_idxs: List[int] = get_periods_idxs(periods, data)
    curr_pd: Period = periods_idxs[0]

    # Get mapping from variable names to column numbers
    var_idx_dict: Dict[str, int] = idx_dict(data.columns)

    # Get non-lead endos and their indexes
    var_idxs: List[int] = [var_idx_dict[name] for name in varlist]
    nonlead_var_idxs: List[Tuple[str, int]] = [
        (name, i) for (name, i) in zip(varlist, var_idxs)
    ]

    # Get numpy arrays out of dataframe
    vals: ndarray = data.values

    for (i, lead) in zip(periods_idxs[1:], range(1, len(periods_idxs))):
        # Indices of the lead `var`s at period `lead`
        # and indices for corresponding non-lead series
        lead_idxs, nonlead_idxs = unzip(
            [(var_idx_dict[f"{var}_{lead}"], j) for (var, j) in nonlead_var_idxs]
        )
        vals[i, nonlead_idxs] = vals[curr_pd, lead_idxs]

    return np2df(vals, data.index, data.columns)


# Optimized in-place update of DataFrame, since setting with .loc is really slow
# New values is a list of data to be input, by quarter
def fast_update_df(
    data: DataFrame,
    varlist: List[str],
    periods: PeriodIndex,
    new_values: List[List[float]],
) -> DataFrame:

    # Convert periods into indices in numpy arrays
    periods_idxs: List[int] = get_periods_idxs(periods, data)

    # Get mapping from variable names to column numbers
    var_idx_dict: Dict[str, int] = idx_dict(data.columns)

    # Get endos and their indexes
    var_idxs: List[int] = [var_idx_dict[name] for name in varlist]

    # Get numpy arrays out of dataframe
    vals: ndarray = data.values

    for i in range(len(periods_idxs)):
        vals[[periods_idxs[i]] * len(var_idxs), var_idxs] = new_values[i]

    return np2df(vals, data.index, data.columns)


# Returns fwd-looking vars in varlist
# Could do this than a smarter way than regex
def get_fwd_vars(varlist: List[str]) -> List[str]:
    return flatten([re.findall(r"(.*?_\d+)", var) for var in varlist])
