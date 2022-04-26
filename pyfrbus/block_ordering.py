import re
import networkx as nx

# For mypy typing
from typing import List, Callable, Tuple, Set, cast, Dict, Optional
from networkx.classes.digraph import DiGraph
from numpy import ndarray

# Imports from this package
import pyfrbus.equations as equations
import pyfrbus.run as run
import pyfrbus.symbolic as symbolic
import pyfrbus.jacobian as jacobian
import pyfrbus.run_jac as run_jac
from pyfrbus.digraph_lib import indegree_zero, simul_component
from pyfrbus.lib import sub_dict_or_raise, join_escape_regex


class BlockOrdering:

    # Initialize block ordering
    def __init__(
        self,
        xsub: List[str],
        exprs: List,
        data_hash: Dict[str, str],
        endo_names: List[str],
        single_block: bool,
        generic_feqs: Callable[[ndarray, ndarray], ndarray],
    ):

        # Block structure of the problem
        self.blocks: List[List[int]] = []
        # Flag whether each block is simultaneous or backwards-looking
        self.is_block_simul: List[bool] = []
        # Flag whether we are using our Newton's method and only require a single block
        self.single_block = single_block
        # Placeholder for jacobian blocks
        self.block_jacs: List[Callable[[ndarray, ndarray, ndarray], ndarray]] = []

        # Compute block ordering
        rhs_vars = equations.rhs_vars(xsub)
        if self.single_block:
            self.blocks = [[i for i in range(len(endo_names))]]
            self.is_block_simul = [True]
        else:
            self.blocks, self.is_block_simul = compute_blocks(rhs_vars, endo_names)

        # Solve equations for respective xi's
        # so they can be evaluated at vals when no unknowns appear
        # Only eqs in non-simultaneous blocks will be evaluated that way
        # and factoring is slow, so we skip the others
        is_eq_simul: List[bool] = [False] * len(xsub)
        for (block, simul) in zip(self.blocks, self.is_block_simul):
            if simul:
                for i in block:
                    is_eq_simul[i] = True
        solved = symbolic.factor_out_xi(xsub, exprs, data_hash, is_eq_simul)

        # Callable related to each block
        # Set up substituted equations for each block
        # This is the version passed to root, each eq set = 0
        # Only needed for simultaneous blocks
        # And it's faster to skip nonsimul blocks
        if not single_block:
            self.block_eqs: List[
                Optional[Callable[[ndarray, ndarray, ndarray], ndarray]]
            ] = [
                # Callable functions, with arguments x, data, z
                (
                    run.fun_form(
                        block_partials(
                            xsub, self.blocks[i], self.blocks[:i], self.single_block
                        ),
                        ["x", "data", "z"],
                    )
                    if self.is_block_simul[i]
                    else None
                )
                for i in range(len(self.blocks))
            ]
        # For single_block mode, we can use the already-eval'd full model
        else:
            self.block_eqs = [lambda x, data, z: generic_feqs(x, data)]

        # These are the backward-looking blocks that can just be called
        # Each eq gives the value of that endo
        # Only needed for non-simultaneous blocks
        self.block_eqs_nox: List[
            Optional[Callable[[ndarray, ndarray, ndarray], ndarray]]
        ] = [
            # Callable functions, with arguments x, data, z
            (
                run.fun_form(
                    block_partials(solved, self.blocks[i], self.blocks[:i]),
                    ["x", "data", "z"],
                )
                if not self.is_block_simul[i]
                else None
            )
            for i in range(len(self.blocks))
        ]

    # Add in block-decomposed Jacobian
    # Defaults to Jacobian functions to return in sparse format
    def add_jac(self, jac: List[Tuple[int, int, str]], sparse: bool = True) -> None:
        # Compute submatrices for each block, convert to function
        self.block_jacs = [
            run_jac.eval_jac(
                jacobian.jacobian_blocks(
                    jac, self.blocks[i], self.blocks[:i], self.single_block
                ),
                len(self.blocks[i]),
                sparse,
            )
            for i in range(len(self.blocks))
        ]


# Compute block-ordering for fsolve_blocks
def compute_blocks(
    rhs_vars: List[Set[str]], endo_names: List[str]
) -> Tuple[List[List[int]], List[bool]]:
    g: DiGraph = nx.DiGraph()
    g.add_nodes_from([f"x[{i}]" for i in range(len(endo_names))])
    g.add_edges_from(rhs_vars_2_edges(rhs_vars))

    blocks: List[List[int]] = []
    is_simul: List[bool] = []
    while len(g) > 0:
        # Add all vars dependent only on variables already solved
        recurs: List[int] = indegree_zero(g)
        while len(recurs) > 0:
            # Casts to satisfy type checker
            blocks.append([int(re.findall(r"\d+", cast(str, x))[0]) for x in recurs])
            is_simul += [False]
            g.remove_nodes_from(recurs)
            recurs = indegree_zero(g)

        # Break if this exhausts the list
        if len(g) == 0:
            break
        # Now, find contemporaneous dependence
        simuls: List[int] = simul_component(g)
        # Ensure blocks come back sorted
        blocks.append(
            sorted([int(re.findall(r"\d+", cast(str, x))[0]) for x in simuls])
        )
        is_simul += [True]
        g.remove_nodes_from(simuls)
    return (blocks, is_simul)


# Returns a function to evaluate given block
# Assuming prev_blocks have all been solved for already
# and will have solutions passed in as vector y
def block_partials(
    xsub, block: List[int], prev_blocks: List[List[int]], single_block: bool = False
) -> List[str]:
    # Skip if only one block
    if single_block:
        return xsub

    # Flatten prev_blocks
    # Rename x's in prev_blocks to z's
    z_eqs = prev_xs_to_zs([xsub[j] for j in block], sum(prev_blocks, []))
    # Renumber x's in equations in this block, in order
    return renumber_xs(z_eqs, block)


# Renumbers "x[i]"s in eqs with respect to the order they appear in block
def renumber_xs(eqs: List[str], block: List[int]) -> List[str]:
    repls = dict([(f"x[{block[i]}]", f"x[{i}]") for i in range(len(block))])
    regex = join_escape_regex(repls)
    # Could be redone without using strings/regex
    return sub_dict_or_raise(eqs, regex, repls)


# Changes "x[i]"s from previous blocks to "z[i]"s to refer to solution vector
def prev_xs_to_zs(eqs: List[str], block: List[int]) -> List[str]:
    if not block:
        return eqs
    repls = dict([(f"x[{real_idx}]", f"z[{real_idx}]") for real_idx in block])
    regex = join_escape_regex(repls)
    return sub_dict_or_raise(eqs, regex, repls)


# Returns pairs (rhs_var, endo_name) reflecting the structure of the model equations
def rhs_vars_2_edges(rhs_vars: List[Set[str]]) -> List[Tuple[str, str]]:
    return sum(
        [
            list(zip(rhs_vars[i], [f"x[{i}]"] * len(rhs_vars[i])))
            for i in range(len(rhs_vars))
        ],
        [],
    )
