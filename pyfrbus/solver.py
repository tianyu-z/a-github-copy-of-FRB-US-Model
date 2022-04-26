import pandas as pd
from scipy.optimize import root
from numpy import array
from numpy.linalg import norm
import warnings
from scipy.sparse import csr_matrix


# For mypy typing
from typing import List, Callable, Optional, Union, Dict
from pandas.core.frame import DataFrame
from pandas import Period, PeriodIndex
from numpy import ndarray

# Imports from this package
from pyfrbus.lib import np2df, get_periods_idxs
from pyfrbus.data_lib import fast_update_df
from pyfrbus.equations import endo_to_trac
from pyfrbus.block_ordering import BlockOrdering
from pyfrbus.newton import newton, trust
from pyfrbus.exceptions import ComputationError, ConvergenceError


# Initialize addfactors (_trac) from simstart to simend, based on input_data
# Returns a data frame with the _trac values filled in
def init_trac(
    simstart: Union[str, Period],
    simend: Union[str, Period],
    data_frame: DataFrame,
    endo_names: List[str],
    endo_idxs: List[int],
    generic_feqs: Callable[[ndarray, ndarray], ndarray],
) -> DataFrame:

    # Get period range from simstart to simend
    periods: PeriodIndex = pd.period_range(simstart, simend, freq="Q")

    # Zero tracs before beginning
    err_names = [endo_to_trac(endo) for endo in endo_names]
    data_frame = fast_update_df(
        data_frame, err_names, periods, [[0] * len(err_names)] * len(periods)
    )

    # List of new trac values, by quarter
    err_list: List[List[float]] = []

    # Convert periods into indices in numpy arrays
    periods_idxs: List[int] = get_periods_idxs(periods, data_frame)

    # Get numpy arrays out of dataframe
    vals: ndarray = data_frame.values

    for i in periods_idxs:
        # Set up "data" and "x" variables for evals
        x = vals[i][endo_idxs]
        data = vals[: (i + 1)]
        # Eval equations to get residuals from input data
        # These are the values of the _tracs
        # Handle numerical warnings like overflow, division by 0, etc.
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                errs = generic_feqs(x, data)
            except RuntimeWarning as war:
                raise ComputationError(war.args[0], "init_trac") from None

        # Store values
        err_list.append([-x for x in errs])

    # Overwrite _tracs in output for all periods
    data_frame = fast_update_df(data_frame, err_names, periods, err_list)
    return data_frame


# Block-based solution method
# Alternates solving for endogenous variables already determined by previous steps
# and solving the smallest remaining block of simultaneous equations
def fsolve_blocks(
    guess: ndarray,
    vals: ndarray,
    blocks: BlockOrdering,
    generic_feqs: Callable[[ndarray, ndarray], ndarray],
    options: Dict,
) -> ndarray:

    # Retrieve solver options
    debug: bool = options["debug"]
    rtol: float = options["rtol"]
    use_newton: Optional[str] = options["newton"]

    # Initialize solution vector
    solution = array([None] * len(guess))

    for k in range(len(blocks.blocks)):
        block = blocks.blocks[k]

        # Use def so we can profile function calls
        def feqs(*args):
            return blocks.block_eqs[k](*args)

        def back_fun(*args):
            return blocks.block_eqs_nox[k](*args)

        # Handle both csr_matrix and standard ndarray
        def call_jac(*args):
            output = blocks.block_jacs[k](*args)
            if type(output) == csr_matrix and not use_newton:
                return output.toarray()
            return output

        # Solve
        if blocks.is_block_simul[k]:
            # Pass vals, solution to feqs and call_jac
            # With handling for warnings from overflow, zero division, etc.
            if not use_newton:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    try:
                        z = root(
                            feqs,
                            [guess[j] for j in block],
                            jac=call_jac,
                            args=(vals, solution),
                        )
                        print(z) if debug else None  # type: ignore
                        # Check that solver reports success (last step < xtol)
                        # AND check that residual is sufficiently small
                        if z.success:
                            if norm(z.fun) < rtol:
                                y = z.x
                            else:
                                raise ConvergenceError(
                                    f"Solver has converged, but with large residual; resid = {norm(z.fun)}"  # noqa:E501
                                )
                        else:
                            raise ConvergenceError(
                                "Solver has diverged, no solution found."
                            )
                    except RuntimeWarning as war:
                        # There are some cases where a warning is raised
                        # but the solution is fine. So we check for that
                        warnings.filterwarnings("ignore")
                        z = root(
                            feqs,
                            [guess[j] for j in block],
                            jac=call_jac,
                            args=(vals, solution),
                        )
                        print(z) if debug else None  # type: ignore
                        if z.success:
                            if norm(z.fun) < rtol:
                                y = z.x
                            else:
                                raise ConvergenceError(
                                    f"Solver has converged, but with large residual; resid = {norm(z.fun)}"  # noqa:E501
                                ) from None
                        else:
                            raise ComputationError(
                                war.args[0], "solver - scipy.optimize.root"
                            ) from None

            # Calculate using variants of Newton's method with sparse Jacobian
            # Potentially less robust than root
            elif use_newton == "trust":
                # Use trust-region Newton method
                y = trust(
                    feqs,
                    call_jac,
                    array([guess[j] for j in block]),
                    vals,
                    solution,
                    options,
                )

            else:  # use_newton == "newton"
                # Use standard Newton's method
                y = newton(
                    feqs,
                    call_jac,
                    array([guess[j] for j in block]),
                    vals,
                    solution,
                    options,
                )

        else:
            # No need to solve, just evaluate at vals, solution
            # Note: first argument does nothing as no x[i]s should appear
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    y = back_fun([None] * len(block), vals, solution)
                except RuntimeWarning as war:
                    # Handling for warnings from overflow, zero division, log(-x)
                    raise ComputationError(
                        war.args[0], "solver - nonsimultaneous block evaluation"
                    ) from None

        # Fill in solution vector
        solution[block] = y

    return solution


def solve(
    simstart: Union[str, Period],
    simend: Union[str, Period],
    data: DataFrame,
    endo_idxs: List[int],
    blocks: BlockOrdering,
    generic_feqs: Callable[[ndarray, ndarray], ndarray],
    options: Dict,
) -> DataFrame:

    # Get period range from simstart to simend
    periods: PeriodIndex = pd.period_range(simstart, simend, freq="Q")
    # Convert periods into indices in numpy arrays
    periods_idxs: List[int] = get_periods_idxs(periods, data)

    # Get numpy arrays out of dataframe
    vals: ndarray = data.values

    for i in periods_idxs:
        # Set up internally-stored data ending at the period to be solved
        # We index into this from the end for exos, lags
        current_data = vals[: (i + 1)]

        # Guess is the value for this period
        guess = current_data[-1][endo_idxs]

        # Solve!
        vals[i, endo_idxs] = fsolve_blocks(
            guess, current_data, blocks, generic_feqs, options
        )

    # Convert numpy arrays back to dataframe, to return
    return np2df(vals, data.index, data.columns)
