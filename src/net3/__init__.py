"""
net3 — binary mask → NetworkX graph.

Main entry point:
    from net3 import vectorize
    G = vectorize("mask.tif")

For a vessel-level graph (one edge per vessel with path geometry):
    from net3 import collapse_to_vessel_graph
    vessel_G = collapse_to_vessel_graph(G)
"""

from .pipeline import vectorize, save_graph, load_graph
from .graph import collapse_to_vessel_graph

__version__ = "3.0.0"

__all__ = [
    "vectorize",
    "save_graph",
    "load_graph",
    "collapse_to_vessel_graph",
    "__version__",
]
