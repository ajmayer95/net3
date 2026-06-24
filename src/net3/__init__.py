"""
net3 — binary mask → NetworkX graph.

Main entry point:
    from net3 import vectorize
    G = vectorize("mask.tif")

For a branch-level graph (one edge per branch with full path
geometry preserved):
    from net3 import collapse_to_branch_graph
    branch_G = collapse_to_branch_graph(G)
"""

from .pipeline import vectorize, save_graph, load_graph
from .graph import collapse_to_branch_graph

# Back-compat alias — older code that imports the prior name still works.
collapse_to_vessel_graph = collapse_to_branch_graph

__version__ = "3.0.0"

__all__ = [
    "vectorize",
    "save_graph",
    "load_graph",
    "collapse_to_branch_graph",
    "collapse_to_vessel_graph",  # alias, kept for back-compat
    "__version__",
]
