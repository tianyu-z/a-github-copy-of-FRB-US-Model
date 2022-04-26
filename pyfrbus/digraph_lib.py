import networkx as nx

# For mypy typing
from networkx.classes.digraph import DiGraph
from typing import List


# Returns list of nodes with indegree 0
# AKA nodes with only backward-looking terms
def indegree_zero(g: DiGraph) -> List[int]:
    sorted_degs = sorted(dict(g.in_degree).items(), key=lambda pair: pair[1])
    return [vertex for (vertex, degree) in sorted_degs if degree == 0]


# Returns smallest block of purely simultaneous equations in g
def simul_component(g: DiGraph) -> List[int]:
    # Get all simultaneous blocks larger than a single element
    s_blocks = [x for x in nx.strongly_connected_components(g) if len(x) > 1]
    # Eliminate blocks that depend on outside variables
    # Check by looking at the block as a subgraph and comparing degrees
    s_blocks = [
        block
        for block in s_blocks
        if [g.in_degree(v) for v in block]
        == [g.subgraph(block).in_degree(v) for v in block]
    ]
    # Return smallest block
    return min(s_blocks, key=len)
