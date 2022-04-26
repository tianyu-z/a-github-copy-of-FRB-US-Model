import lxml.etree as ElementTree
from copy import deepcopy
import re
import pandas as pd
import numpy

# For mypy typing
from typing import List, Set, Dict, Callable, Optional, Tuple, Union
from lxml.etree import Element
from pandas.core.frame import DataFrame
from numpy import ndarray
from symengine.lib.symengine_wrapper import Expr
from pandas import Period, PeriodIndex

# Imports from this package
import pyfrbus.xml_model as xml_model
import pyfrbus.equations as equations
import pyfrbus.run as run
import pyfrbus.symbolic as symbolic
from pyfrbus.block_ordering import BlockOrdering
import pyfrbus.jacobian as jacobian
import pyfrbus.solver as solver
from pyfrbus.solver_opts import solver_defaults
import pyfrbus.mcontrol as mcontrol
import pyfrbus.stochsim as stochsim
from pyfrbus.lib import flatten, np2df, idx_dict
from pyfrbus.data_lib import drop_mce_vars, copy_fwd_to_current, get_fwd_vars
import pyfrbus.lexing as lexing
import pyfrbus.constants as constants
from pyfrbus.exceptions import InvalidArgumentError, MissingDataError


class Frbus:
    def __init__(self, filepath: str, mce: Optional[str] = None):
        """
        Initialize FRB/US model object.

        Parameters
        ----------
        filepath: str
            Path to FRB/US model file in .xml format
        mce: Optional[str]
            Option to load MCE equations.
            Valid MCE types are ``all``, ``mcap``, ``wp``, and ``mcap+wp``

        Returns
        -------
        Frbus
            FRB/US model object

        """

        # Parse model XML file
        xml: Element = ElementTree.parse(filepath).getroot()

        # Names of endogenous variables
        # orig_ fields are the ones read from XML
        # others may be edited during model setup
        self.orig_endo_names: List[str] = xml_model.endo_names_from_xml(xml)
        self.endo_names = deepcopy(self.orig_endo_names)
        # Corresponding equations
        eqs: List[str] = xml_model.equations_from_xml(xml)
        # Constants from those equations
        self.constants: Dict[str, float] = xml_model.constants_from_xml(xml)
        # Variables to be shocked in stochastic sims
        self.stoch_shocks: List[str] = xml_model.stoch_shocks(xml)

        # Read in MCE equations/constants if needed
        self.has_leads = bool(mce)
        if mce:
            # Throw error if invalid mce type is given
            if mce not in constants.CONST_MCE_TYPES:
                raise InvalidArgumentError("Frbus constructor", "mce", mce)
            (mce_eqs, mce_vars) = xml_model.mce_from_xml(xml, mce)
            mce_idxs = [self.endo_names.index(var) for var in mce_vars]
            # Replace equations with MCE version
            for (i, eq) in zip(mce_idxs, mce_eqs):
                eqs[i] = eq
            # Add/replace constants
            self.constants.update(xml_model.mce_constants_from_xml(xml, mce))

        # Add tracking residual variables
        eqs = [eqs[i] + f"+{self.endo_names[i]}_trac" for i in range(len(eqs))]

        # Names of exogenous variables
        # First we must drop unused series from the model.xml
        tmp_exos = xml_model.exo_names_from_xml(xml)
        tmp_exos = [exo for exo in tmp_exos if any([exo in eq for eq in eqs])]
        # Add in _aerrs and _tracs for every endogenous variable
        self.orig_exo_names: List[str] = tmp_exos + [
            endo + "_aerr" for endo in self.endo_names
        ] + [endo + "_trac" for endo in self.endo_names]
        self.exo_names = deepcopy(self.orig_exo_names)

        # Fill in constants
        filled_eqs: List[str] = equations.fill_constants(
            [equations.flip_equals(eq) for eq in eqs], self.constants
        )

        # Lex to separate variable identifier from everything else
        self.orig_lexed_eqs = lexing.lex_eqs(filled_eqs)
        self.lexed_eqs = deepcopy(self.orig_lexed_eqs)

        # Field to store dataframe column names
        self.data_varnames: List[str] = []
        # List of additional exogenous variables, specified at runtime
        self.exoglist: Set[str] = set()
        # Model updated flags
        self.exoglist_changed: bool = False
        self.eqs_changed: bool = False
        # Substituted equation placeholder
        self.xsub: List[str] = []
        # Jacobian placeholder
        self.jac: Optional[List[Tuple[int, int, str]]] = None

    # Takes a list of endogenous variables to exogenize
    def exogenize(self, exoglist: List[str]) -> None:
        """
        Exogenize a list of endogenous variables in the model.

        This function takes a list of endogenous variables present in the model and
        transforms them into exogenous variables.
        Note: all variables that you want to be exogenous must
        be passed in as `exoglist` at the same time. Calling ``exogenize`` twice in a
        row will give a model with only the second `exoglist` variables exogenized.

        Parameters
        ----------
        exoglist: List[str]
            List of endogenous model variables

        """

        self.exoglist = set(exoglist)
        self.exoglist_changed = True

    # Add a set of equations to the model
    # Replaces equations if the endo already exists
    # Input is a map of endo name -> equation string
    # Dict MUST INCLUDE all new endos used in other new equations
    def append_replace(self, eqs_map: Dict[str, str]) -> None:
        """
        Append new equations to the model, or replace existing equations.

        Parameters
        ---------
        eqs_map: Dict[str, str]
            Dictionary mapping equation names to equations

        """

        # First, standardize format by cleaning equations and removing =
        # And append tracking residuals
        eqs_map = {
            endo: equations.flip_equals(equations.clean_eq(eq + f"+{endo}_trac"))
            for (endo, eq) in eqs_map.items()
        }

        # Split endos; old_endos is replaced, new_endos is appended
        old_endos_map = {
            endo: eq for (endo, eq) in eqs_map.items() if endo in self.orig_endo_names
        }
        new_endos_map = {
            endo: eq
            for (endo, eq) in eqs_map.items()
            if endo not in self.orig_endo_names
        }
        # Append new endos
        self.orig_endo_names = self.orig_endo_names + list(new_endos_map.keys())
        self.endo_names = deepcopy(self.orig_endo_names)

        # Replace equations for old endos
        for (endo, repl_eq) in old_endos_map.items():
            index = self.orig_endo_names.index(endo)
            self.orig_lexed_eqs[index] = lexing.lex_eq(repl_eq)

        # Append new endo equations
        self.orig_lexed_eqs = self.orig_lexed_eqs + [
            lexing.lex_eq(eq) for eq in new_endos_map.values()
        ]
        self.lexed_eqs = deepcopy(self.orig_lexed_eqs)

        # Assemble list of tokens which are not new exos
        # i.e. functions, endos, and old exos
        ban_list = (
            constants.CONST_SUPPORTED_FUNCTIONS_EX + self.endo_names + self.exo_names
        )

        # Any token in an equation that is NOT an endo is assumed to be an exo
        tokens = flatten([re.sub(r"[^\w]", " ", eq).split() for eq in eqs_map.values()])

        # We want unique new tokens
        # We use dict->list instead of set because it preserves order
        new_exos = {
            token: None
            for token in tokens
            if (token not in ban_list and not re.match(r"^\d", token))
        }

        # Update has_leads if appended equations are forward-looking
        self.has_leads = equations.has_leads(self.lexed_eqs)

        # Add new exos
        self.orig_exo_names = self.orig_exo_names + list(new_exos.keys())
        self.exo_names = deepcopy(self.orig_exo_names)

        # Ensure that xsub, etc. is regenerated when model is next used
        self.eqs_changed = True
        # Also because endo_names is updated, may need to re-exogenize
        self.exoglist_changed = True

    # Adds missing aerrs to dataframe, along with tracking residuals
    # Exogenizes additional variables from .exogenize
    # Substitutes references to "x", "data" in equations
    # Turns equations into lambdas
    def _solve_setup(
        self,
        data: DataFrame,
        start: str = "",
        end: str = "",
        single_block: bool = False,
    ) -> DataFrame:

        # First, make a copy of input data to be returned
        data = data.copy()
        # Add missing variables
        data = _fix_errs_in_data(data, self.endo_names)

        # Exogenize specified variables, if necessary
        if self.exoglist_changed:
            self._reset_model()

        # If MCE, fill in columns for fwd-looking equations
        # We do this every time, as they get wiped before data is returned to the user
        if self.has_leads:
            data = _populate_mce_data(
                data, start, len(pd.period_range(start, end, freq="Q"))
            )

        # Check if equation setup needs to be run
        # Get names of columns in DataFrame
        # If they are the same as last time, we do not need to run setup
        # as data indices will be correct.
        # UNLESS equations have been added/modified
        # or the requested block decomposition is different from the stored one
        if (
            self.data_varnames != list(data.columns)
            or self.eqs_changed
            or not hasattr(self, "blocks")
            or ((single_block or self.has_leads) and len(self.blocks.blocks) > 1)
            or (not (single_block or self.has_leads) and len(self.blocks.blocks) == 1)
        ):

            # If there are leads, do MCE setup
            if self.has_leads:
                data = self._mce_setup(data, start, end)

            # Store names of data frame columns
            # Important, related to how lags/exos are substituted in equations
            # If dataset changes, we need to redo the setup
            self.data_varnames = list(data.columns)
            # Convert endo names into indices in numpy arrays
            # Gives a mapping colname -> index for a dataframe
            # Used to speed up multi-gets and multi-sets
            data_varnames_idx_dict: Dict[str, int] = idx_dict(data.columns)
            try:
                self.endo_idxs: List[int] = [
                    data_varnames_idx_dict[name] for name in self.endo_names
                ]
            except KeyError as err:
                raise MissingDataError(err.args[0]) from None

            # Reset model changed flag
            self.eqs_changed = False

            # Turn equations into expressions that = 0
            # Fill in lags and exos, so only contemporaneous terms remain
            # Pass in data_varnames so fill_lags and fill_exos have correct indexes
            # Replace variable names with x[i]s so it can be eval'd
            self.xsub = equations.fill_lags_and_exos_xsub(
                self.lexed_eqs, data_varnames_idx_dict, self.exo_names, self.endo_names
            )

            # Corresponding functions, as lambdas with arguments x, data
            self.generic_feqs: Callable[[ndarray, ndarray], ndarray] = run.fun_form(
                self.xsub, ["x", "data"]
            )

            # Convert equation to SymPy/SymEngine exprs, for equation solving, jacobian
            # Tokens like data[-i,j] are converted to data[k]
            # in some array of "data" symbols, so SymEngine can understand them
            # data_hash is the mapping  data[k] => data[-i,j]
            self.exprs: List[Expr] = []
            self.data_hash: Dict[str, str] = {}
            self.exprs, self.data_hash = symbolic.to_symengine_expr(self.xsub)

            # Set up Jacobian, if needed
            if not self.jac:
                # Compute Jacobian
                if self.has_leads:
                    self.jac = jacobian.mce_create_jacobian(
                        len(self.xsub),
                        equations.rhs_vars(self.xsub),
                        self.exprs,
                        self.data_hash,
                        self.endo_names,
                        self.data_varnames,
                    )
                else:
                    self.jac = jacobian.create_jacobian(
                        len(self.xsub),
                        equations.rhs_vars(self.xsub),
                        self.exprs,
                        self.data_hash,
                    )

            # Compute block ordering
            # Always use single block if MCE
            self.blocks: BlockOrdering = BlockOrdering(
                self.xsub,
                self.exprs,
                self.data_hash,
                self.endo_names,
                single_block or self.has_leads,
                self.generic_feqs,
            )
            # Add Jacobian to the block ordering
            self.blocks.add_jac(self.jac)

        # Return fixed data
        return data

    # Setup required for MCE simulations
    def _mce_setup(self, data: DataFrame, start: str, end: str):
        # First, reset MCE state in case setup has already been run
        self._reset_model()

        # Number of periods to terminal period
        periods: PeriodIndex = pd.period_range(start, end, freq="Q")
        n_periods: int = len(periods)

        # Duplicate endogenous variables as leads
        dupe_endos = equations.dupe_vars(self.endo_names, n_periods)
        # Duplicate exogenous variables as leads
        dupe_exos = equations.dupe_vars(self.exo_names, n_periods)

        self.endo_names += dupe_endos
        self.exo_names += dupe_exos

        # Build stacked time system by duplicating equations
        self.lexed_eqs = self.lexed_eqs + equations.dupe_eqs(self.lexed_eqs, n_periods)

        # Handle terminal condition: if period > end
        # Terminals are substituted, data from start_date+period
        start_date: Period = pd.Period(start)

        # Turn leads into new contemporaneous variables
        # and substitute out terminal condition
        self.lexed_eqs = lexing.remove_leads(
            self.lexed_eqs, data, start_date, n_periods
        )

        return data

    # Method to reset model state from original state
    def _reset_model(self) -> None:
        # Reset endos, exos, and equations
        self.endo_names = deepcopy(self.orig_endo_names)
        self.exo_names = deepcopy(self.orig_exo_names)
        self.lexed_eqs = deepcopy(self.orig_lexed_eqs)

        # Remove corresponding endos and equations,
        # add vars to list of exos
        idxs = sorted([self.endo_names.index(x) for x in self.exoglist], reverse=True)
        [self.endo_names.pop(i) for i in idxs]
        [self.lexed_eqs.pop(i) for i in idxs]
        self.exo_names.extend(self.exoglist)

        # And ensure that none of the original exos have been given equations
        self.exo_names = [exo for exo in self.exo_names if exo not in self.endo_names]

        # Delete previous Jacobian
        self.jac = None
        # Indicate that equations need to be re-generated
        # to change newly exog variables and point them to "data" instead of "x"
        self.eqs_changed = True
        # Exoglist is now consistent with model
        self.exoglist_changed = False

    # Initialize addfactors (_trac) from start to end, based on input_data
    # Returns a data frame with the _trac values filled in
    def init_trac(
        self, start: Union[str, Period], end: Union[str, Period], input_data: DataFrame
    ) -> DataFrame:
        """
        Initialize tracking residuals (add-factors).

        Given a baseline dataset `input_data` and a date range `start` to `end`,
        ``init_trac`` will compute tracking residuals such that each equation will solve
        to the trajectory given for that variable in `input_data`.
        As in, ``output = model.init_trac(start, end, input_data)``, then
        ``model.solve(start, end, output)`` will be the same as ``output``.

        Tracking residuals are stored in additional columns in ``output``, named
        according to their respective endogenous variables, e.g. the tracking residual
        for ``xgdp`` is stored as ``xgdp_trac``.

        Parameters
        ----------
        start: Union[str, Period]
            Date to begin computing residuals

        end: Union[str, Period]
            Date to end residuals (inclusive)

        input_data: DataFrame
            Dataset with trajectories that residuals are computed with respect to

        Returns
        -------
        output: DataFrame
            `input_data` plus columns for tracking residuals

        """

        # Set up substituted equations, data
        data: DataFrame = self._solve_setup(input_data, start, end)

        # Create frame with initialized _tracs
        # We only need them for one period in the MCE solver
        if self.has_leads:
            with_adds = solver.init_trac(
                start, start, data, self.endo_names, self.endo_idxs, self.generic_feqs
            )
            # Copy fwd-looking _tracs to their contemporaneous versions before we drop
            fwd_endos = set(get_fwd_vars(self.endo_names))
            errs = [f"{var}_trac" for var in self.endo_names if var not in fwd_endos]
            with_adds = copy_fwd_to_current(
                with_adds, errs, pd.period_range(start, end, freq="Q")
            )
            # Drop MCE columns and return
            return drop_mce_vars(with_adds)
        else:
            return solver.init_trac(
                start, end, data, self.endo_names, self.endo_idxs, self.generic_feqs
            )

    # Solves the model from start to end on input data
    # Returns a data frame with endo solutions filled in
    def solve(
        self,
        start: Union[str, Period],
        end: Union[str, Period],
        input_data: DataFrame,
        options: Optional[Dict] = None,
    ) -> DataFrame:
        """
        Solve the model over the given dataset.

        Given the DataFrame `input_data`, this procedure returns the model solution
        over the period from `start` to `end`. Detailed information on the solution
        algorithm can be found in the User Guide.

        ``Frbus.solve`` will solve both backwards-looking VAR and forwards-looking MCE
        (model-consistent expectations/rational expectations) models. The `options`
        dictionary can be passed to configure the solution algorithm, as specified
        below.

        Parameters
        ----------
        start: Union[str, Period]
            Date to begin computing solution

        end: Union[str, Period]
            Date to end solution (inclusive)

        input_data: DataFrame
            Dataset to solve over

        options: Optional[Dict]
            Options to pass to solver:
                ``newton: Optional[str]``
                    Whether to use sparse Newton's method solver (``"newton"``),
                    sparse trust-region solver (``"trust"``), or dense solver from SciPy
                    (``None``).
                    Defaults to ``None``.
                ``single_block: bool``
                    When set to ``True``, disables the VAR  block decomposition step.
                    Defaults to ``False``.
                ``debug: bool``
                    When set to ``True``, enables verbose output of solver status.
                    Defaults to ``False``.
                ``xtol: float``
                    Set stepsize termination condition for Newton and trust-region
                    algorithms. Use a smaller value for more precise solution, at the
                    expense of runtime. Defaults to ``1e-6``.
                ``rtol: float``
                    Set residual termination condition for Newton and trust-region
                    algorithms. Solver will fail if step reaches ``xtol`` but residual
                    is greater than ``rtol``. Defaults to ``5e-4``.
                ``maxiter: int``
                    Maximum number of iterations for Newton and trust-region algorithms.
                    Solver will fail if algorithm iterates ``maxiter`` times without
                    reaching ``xtol``. Increase to allow more time for convergence.
                    Defaults to ``50``.
                ``trust_radius: float``
                    Maximum size of the trust radius in trust-region algorithm. Altering
                    radius can change convergence path. Defaults to ``1000000``.
                ``precond: bool``
                    When set to ``True``, use matrix preconditioner to decrease
                    condition number of Jacobian. Disable if it causes problems.
                    Defaults to ``True``.

        Returns
        -------
        output: DataFrame
            Dataset shaped like `input_data`, with trajectories for endogenous variables
            produced by model solution between `start` and `end`, inclusive. Data in
            `output` from outside this period is identical to `input_data`.

        """

        # Get defaults for omitted options
        options = solver_defaults(options)

        # Set up substituted equations, data, jacobian
        data: DataFrame = self._solve_setup(
            input_data, start, end, options["single_block"]
        )

        # Solve for period start:end (inclusive)
        # Call the MCE solver if there are leads
        if self.has_leads:
            # MCE assumes model has been setup in stacked time format
            # Solves for a single period and substitutes endo data from leads
            # Defaults to Newton if not specified
            options["newton"] = options["newton"] or "newton"
            soln: DataFrame = solver.solve(
                start,
                start,
                data,
                self.endo_idxs,
                self.blocks,
                self.generic_feqs,
                options,
            )

            # Copy single-period solution back to original columns
            fwd_endos = set(get_fwd_vars(self.endo_names))
            var_endo_names = [var for var in self.endo_names if var not in fwd_endos]
            soln = copy_fwd_to_current(
                soln, var_endo_names, pd.period_range(start, end, freq="Q")
            )
            # Drop MCE columns and return
            return drop_mce_vars(soln)
        else:
            return solver.solve(
                start,
                end,
                data,
                self.endo_idxs,
                self.blocks,
                self.generic_feqs,
                options,
            )

    # Solves the model while forcing the target variable to the specified trajectory
    # by moving the instrument
    def mcontrol(
        self,
        start: Union[str, Period],
        end: Union[str, Period],
        input_data: DataFrame,
        targ: List[str],
        traj: List[str],
        inst: List[str],
        options: Optional[Dict] = None,
    ) -> DataFrame:
        """
        Solve model, forcing target variables to specified trajectories.

        ``mcontrol`` is a trajectory-matching control procedure which adjusts the value
        of instrument variables such that target variables are forced to specified
        trajectories, as mediated by the model's dynamics.

        `targ` is a list of model variables ("targets"), and `traj` is the list of
        series in `input_data` that those variables should be forced to
        ("trajectories"), in the same order; e.g. the first variable in `targ` is
        matched to the first trajectory in `traj`, etc.

        `inst` is a list of exogenous model variables ("instruments") which will take
        on values such that the trajectories are achieved. The selected instruments may
        be unable to achieve the specified trajectories - e.g. because there is no
        instrument which is able to affect one or more of the specified targets, or
        because one or more of the specified trajectories contains an invalid value for
        that target variable. In that case, the model will fail to solve and an error
        will instruct you to verify that your setup for ``mcontrol`` is valid.

        Targets are only forced to their trajectories when a trajectory is present. A
        particular target can be disabled by setting the corresponding trajectory to
        ``numpy.nan`` over the date range where it should be inactive. When the
        trajectory is set to ``numpy.nan``, the corresponding target is allowed to take
        on the value produced by the model.

        Parameters
        ----------
        start: Union[str, Period]
            Date to begin computing solution

        end: Union[str, Period]
            Date to end solution (inclusive)

        input_data: DataFrame
            Dataset to solve over, including series for trajectories specified in `traj`

        targ: List[str]
            List of endogenous model variables to force (in order)

        traj: List[str]
            List of trajectories in `input_data` to force `targ` variables to (in order)

        inst: List[str]
            Instruments used to control forcing procedure

        options: Optional[Dict]
            Options passed to solver - see additional documentation under
            ``Frbus.solve``. Some options will be overridden if they are unable to be
            used with ``mcontrol`` procedure; ``mcontrol`` requires the use of either
            Newton or trust-region solvers (defaults to ``"newton"``) and requires
            ``single_block`` set to ``True``.

        Returns
        -------
        output: DataFrame
            Dataset shaped like `input_data`, with a solution consistent with the
            specified forced trajectories between `start` and `end`, inclusive. Data
            from `output` outside this period is identical to `input_data`.

        """

        return mcontrol.mcontrol(
            self, start, end, input_data, targ, traj, inst, options
        )

    # Runs nrepl stochastic simulations from start to end on input data
    # Using specified residual dates for shock quarters, drawn IID
    # Returns a list of data frames for each replication
    def stochsim(
        self,
        nrepl: int,
        input_data: DataFrame,
        simstart: Union[str, Period],
        simend: Union[str, Period],
        residstart: Union[str, Period],
        residend: Union[str, Period],
        multiproc: bool = True,
        nextra: int = 0,
        seed: int = 1000,
        options: Optional[Dict] = None,
    ) -> List[Union[DataFrame, str]]:
        """
        Runs a series of simulations with shocks drawn from historical residuals.

        The ``stochsim`` procedure performs a stochastic simulation by applying
        sequences of shocks to the model, as drawn randomly from historical residuals.
        Before using this procedure, the DataFrame `input_data` must have residuals
        computed over both history and the simulation period with ``Frbus.init_trac'``.

        The procedure begins by drawing `nrepl` sequences of quarters from the dataset
        `input_data` over the periods `residstart` to `residend`, where the length of
        that sequence goes from `simstart` to `simend`. That is, for a particular
        replication, each quarter in the simulation period is randomly assigned a
        quarter from residual period. In that quarter of the simulation, all stochastic
        variables (specified with a ``stochastic_type`` tag in the model) have a shock
        applied from a particular quarter in the residual period, where the shock values
        are pulled from the ``_trac`` variables in ``input_data`` - e.g., the variable
        ``xgdp`` is shocked with a value pulled from ``xgdp_trac`` over history.

        These scenarios are passed to the solver, in parallel if `multiproc` is set to
        ``True``, and the solutions are returned as a list of DataFrames. Any failed
        simulations will be returned as an error string. The argument `nextra` can be
        passed to allow the procedure to run extra simulations to replace those
        failures.


        Parameters
        ----------
        nrepl: int
            Number of simulations to run

        input_data: DataFrame
            Dataset to solve over

        simstart: Union[str, Period]
            Date to begin computing solution

        simend: Union[str, Period]
            Date to end solution (inclusive)

        residstart: Union[str, Period]
            Date to begin drawing shocks

        residend: Union[str, Period]
            Date to end drawing shocks (inclusive)

        multiproc: bool
            Option to run simulations in parallel

        nextra: int
            Option to specify how many additional simulations to run, in case of errors

        seed: int
            Random seed used when drawing shocks

        options: Optional[Dict]
            Options passed to solver - see additional documentation under
            ``Frbus.solve``.

        Returns
        -------
        solutions: List[Union[DataFrame, str]]
            List of datasets shaped like `input_data`, with solutions to stochastic
            simulations as returned from ``Frbus.solve``. List contains error strings
            instead of data for simulations that fail.

        """

        return stochsim.stochsim(
            self,
            nrepl,
            input_data,
            simstart,
            simend,
            residstart,
            residend,
            multiproc,
            nextra,
            seed,
            options,
        )

    # Define custom getstate/setstate for pickle and deepcopy
    # Delete symengine exprs before copying because they are not pickleable
    def __getstate__(self):
        has_exprs = hasattr(self, "exprs")
        if has_exprs:
            tmp = self.exprs
            self.exprs = []
        state = deepcopy(self.__dict__)
        if has_exprs:
            # Restore exprs to original object
            self.exprs = tmp
        return state

    def __setstate(self, newstate):
        # If it should have exprs, reconstruct them from xsub
        if "exprs" in newstate:
            newstate["exprs"], _ = symbolic.to_symengine_expr(newstate["xsub"])
        self.__dict__.update(newstate)


# Make sure dataset includes a _aerr, _trac for each endogenous variable
# ex. USEGBX leaves out qlf_aerr, qlfpr_aerr
def _fix_errs_in_data(data: DataFrame, endo_names: List[str]) -> DataFrame:
    # Skip fwd-looking endos of the form 'zrff5_1'
    all_aerrs = [
        aerr
        for aerr in [
            f"{endo}_aerr"
            for endo in endo_names
            if not re.search(r"_\d+", endo) and f"{endo}_aerr" not in data.columns
        ]
    ]
    all_tracs = [
        aerr
        for aerr in [
            f"{endo}_trac"
            for endo in endo_names
            if not re.search(r"_\d+", endo) and f"{endo}_trac" not in data.columns
        ]
    ]
    # Add to the data
    # Fastest method I've found
    data = pd.concat(
        [data, pd.DataFrame(0, index=data.index, columns=all_aerrs + all_tracs)], axis=1
    )
    return data


# Populate data frame with data for fwd-looking variables
# Only needed for the period start
# Exos need data, endos need a starting "guess" for the solver
# These are my hacks to try to speed up data frame population
def _populate_mce_data(data: DataFrame, start: str, n_periods: int) -> DataFrame:
    # Create template data frame, for each period of leads added
    tmp = numpy.empty((data.shape[0], data.shape[1] * (n_periods - 1)))

    # Set up values for new variables in-order
    values: List[float] = flatten(
        [
            data.loc[(pd.Period(start) + 1) : (pd.Period(start) + n_periods - 1), nm]
            for nm in data.columns
        ]
    )

    # Set values for new lead variables at time 0
    i = list(data.index).index(pd.Period(start))
    tmp[i] = values

    # Append templated frame to end of data; better to concat as numpy
    # Otherwise pandas creates two FloatBlocks to be consolidated later
    tmp2 = numpy.concatenate((data.values, tmp), axis=1)
    data = np2df(
        tmp2,
        data.index,
        list(data.columns)
        + [f"{nm}_{n}" for nm in data.columns for n in range(1, n_periods)],
    )
    return data
