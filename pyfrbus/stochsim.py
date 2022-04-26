import random
import pandas
import numpy

from functools import partial
from multiprocess import Pool
from copy import deepcopy

# For mypy typing
from typing import List, Union, Dict, Optional
from pandas import DataFrame, PeriodIndex, Period

# Imports from this package
from pyfrbus.solver_opts import solver_defaults


def stochsim(
    frbus,
    nrepl: int,
    with_adds: DataFrame,
    simstart: Union[str, Period],
    simend: Union[str, Period],
    residstart: Union[str, Period],
    residend: Union[str, Period],
    multiproc: bool,
    nextra: int,
    seed: int,
    options: Optional[Dict],
) -> List[Union[DataFrame, str]]:

    # Total number of replications to run
    nrepl = nrepl + nextra

    # Set of equations to be shocked
    shocks = [var + "_trac" for var in frbus.stoch_shocks]

    # Copy historical residuals
    shock_mat = with_adds.loc[residstart:residend, shocks]  # type: ignore
    # Demean
    shock_mat = shock_mat - shock_mat.mean()

    # Sim period
    sim_qtrs: PeriodIndex = pandas.period_range(simstart, simend, freq="Q")
    # Residual period
    shock_qtrs: PeriodIndex = pandas.period_range(residstart, residend, freq="Q")

    # Number of quarters of simulation
    nqtrs = len(sim_qtrs)
    # Number of quarters of residuals
    nqtrs_resid = len(shock_qtrs)

    # Generate matrix of quarters to be used as shocks in each replication
    index_mat = pandas.DataFrame(numpy.full((nqtrs, nrepl), -1))
    # Set random seed before taking draws so they will be the same every time
    random.seed(seed)
    index_mat = index_mat.applymap(
        lambda _: shock_qtrs[random.randrange(0, nqtrs_resid)]
    )

    # Break index_mat into a list of columns
    index_mat_list = [col for (_, col) in index_mat.iteritems()]

    # Run stochastic simulation
    solutions: List[Union[DataFrame, str]] = []
    if multiproc:
        with Pool() as p:
            # Partial eval to get a function that runs on an index mat
            z = partial(
                run_repl,
                baseline=with_adds,
                shock_mat=shock_mat,
                sim_qtrs=sim_qtrs,
                shocks=shocks,
                frbus=frbus,
                options=options,
            )
            # Map over index mats
            solutions = p.map(z, index_mat_list)
    else:
        # Singlethreaded, in a loop
        for i in range(0, len(index_mat_list)):
            solutions += [
                run_repl(
                    shock_qtrs=index_mat_list[i],
                    baseline=with_adds,
                    shock_mat=shock_mat,
                    sim_qtrs=sim_qtrs,
                    shocks=shocks,
                    frbus=frbus,
                    options=options,
                )
            ]

    # Throw out up to nextra failed sims
    # any additional failures have error messages returned to user
    nfailures = 0
    i = 0
    while nfailures < nextra and i < len(solutions):
        if not isinstance(solutions[i], DataFrame):
            solutions.pop(i)
            nfailures += 1
        else:
            i += 1
    # Drop any other extra sims
    solutions = solutions[0:nrepl]

    return solutions


# Function to run a single replication
# Returns exception as string if a replication fails
def run_repl(
    shock_qtrs: PeriodIndex,
    baseline: DataFrame,
    shock_mat: DataFrame,
    sim_qtrs: PeriodIndex,
    shocks: List[str],
    frbus,
    options: Optional[Dict],
) -> Union[DataFrame, str]:
    sim = baseline.copy()
    # Set shocks over sim period to historical _tracs from quarters drawn above
    sim.loc[sim_qtrs, shocks] += shock_mat.loc[shock_qtrs, shocks].values
    # Solve
    try:
        return frbus.solve(sim_qtrs[0], sim_qtrs[-1], sim, options=options)
    # Fail over to Newton
    except Exception as e:
        # Skip if already using Newton or trust-region
        if options and "newton" in options and options["newton"]:
            return str(e)
        else:
            options = solver_defaults(deepcopy(options))
            options["newton"] = "newton"
            try:
                return frbus.solve(sim_qtrs[0], sim_qtrs[-1], sim, options=options)
            except Exception as ee:
                return str(ee)
