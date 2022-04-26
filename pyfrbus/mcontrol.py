from copy import deepcopy
from numpy import nan, isnan

# For mypy typing
from typing import Union, List, Dict, Optional
from pandas.core.frame import DataFrame
from pandas import Period

# Imports from this package
from pyfrbus.exceptions import InvalidArgumentError, ConvergenceError


# Solves the model while forcing the target variable to the specified trajectory
# by moving the instrument
# Instrument must be exo, target must be endo
def mcontrol(
    orig_frbus,
    start: Union[str, Period],
    end: Union[str, Period],
    input_data: DataFrame,
    targ: List[str],
    traj: List[str],
    inst: List[str],
    options: Optional[Dict],
) -> DataFrame:

    # Ensure that the same number of targs, trajs, and insts are passed in
    if not (len(targ) == len(traj) and len(traj) == len(inst)):
        raise InvalidArgumentError(
            "mcontrol", "targets, trajectories, and instruments must be the same length"
        )

    # Copy original data and model so they are not modified
    data = deepcopy(input_data)
    frbus = deepcopy(orig_frbus)

    # Init data for forcing period switch
    # And store inst values for non-force periods
    new_eqs = {}
    for (curr_targ, curr_traj, curr_inst) in zip(targ, traj, inst):
        data.loc[start:end, f"old_{curr_inst}"] = data.loc[  # type: ignore
            start:end, curr_inst  # type: ignore
        ]
        data.loc[start:end, f"targetswitch_{curr_targ}"] = 0  # type: ignore
        # Get forcing dates from where trajectory is not nan
        data.loc[start:end, f"targetswitch_{curr_targ}"] = 1 * ~isnan(  # type: ignore
            data.loc[start:end, curr_traj]  # type: ignore
        )

        # Change nans in trajectory to a placeholder, to avoid solver errors
        data.loc[data[f"targetswitch_{curr_targ}"] == 0, curr_traj] = 0

        # Append equation for instrument to impose trajectory on target
        new_eqs[
            curr_inst
        ] = f"({curr_inst}-old_{curr_inst})*(1-targetswitch_{curr_targ}) = ({curr_traj} - {curr_targ})*targetswitch_{curr_targ}"  # noqa: E501

    frbus.append_replace(new_eqs)

    # Note: must use single block and Newton solver to deal with appended equations
    # that are not in endo = f(endos, exos) format
    if options:
        options["newton"] = (
            options["newton"] or "newton" if "newton" in options else "newton"
        )
        options["single_block"] = True
    else:
        options = {"newton": "newton", "single_block": True}

    # Solve for mcontrol solution
    try:
        sim = frbus.solve(start, end, data, options=options)
    # Return a custom message if solver fails to converge
    except ConvergenceError:
        # Not raising from None here, so that user can also see how the solver stopped
        raise ConvergenceError(
            "mcontrol has failed to converge - check that instruments can move targets"  # noqa: E501
        )

    # Re-insert trajectory nans
    for (curr_targ, curr_traj) in zip(targ, traj):
        sim.loc[data[f"targetswitch_{curr_targ}"] == 0, curr_traj] = nan

    # Delete columns used for configuring mcontrol
    series = (
        [f"old_{s}" for s in inst]
        + [f"targetswitch_{s}" for s in targ]
        + [f"{s}_aerr" for s in inst]
        + [f"{s}_trac" for s in inst]
    )
    sim.drop(series, axis=1, inplace=True)
    return sim
