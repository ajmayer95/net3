"""
GraphEditor — the GUI-free backend that holds the working graph,
selection, undo stack, and exposes the editing operations the
frontend wires keys/buttons to.

Modernised from Jana Lasser's `gegui` (Python 2, NetworkX 1.x) to
Python 3 + NetworkX 3.x.  Operations preserved from the original:
delete-selected, add-node (with radius sampled from distance map),
add-edge, find-cycles, streamline (degree-2 collapse), make-digraph,
undo, save.

Everything is testable without a GUI — see tests/test_edit_core.py.
"""

from __future__ import annotations

import math
import pickle
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional, Tuple, Set, List

import networkx as nx
import numpy as np


def _load_gpickle(path: Path) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_gpickle(G: nx.Graph, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(G, f)


def _load_image(path: Path) -> np.ndarray:
    """Load a TIFF/PNG image as a 2-D uint8 array (gray)."""
    from PIL import Image
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.uint8)


class GraphEditor:
    """Stateful wrapper around a NetworkX graph for interactive editing.

    Attributes
    ----------
    graph : nx.Graph
        The current working graph.  Modified in-place by every edit
        operation; restored from snapshots by `undo()`.
    selected : set[Hashable]
        Currently-selected node ids.
    mask : np.ndarray | None
        Optional 2-D image overlay (mask or microscopy frame) for
        display.  Not used by the backend operations themselves.
    distance_map : np.ndarray | None
        Optional 2-D float array used to auto-assign radius to newly
        created nodes (radius = distance_map[y, x]).  Falls back to a
        default radius if not provided.
    """

    DEFAULT_RADIUS = 1.0
    MAX_UNDO_DEPTH = 20

    # ── Construction ───────────────────────────────────────────────

    def __init__(
        self,
        graph: nx.Graph,
        mask: Optional[np.ndarray] = None,
        distance_map: Optional[np.ndarray] = None,
    ):
        self.graph = graph
        self.mask = mask
        self.distance_map = distance_map
        self.selected: Set = set()
        self._undo_stack: deque = deque(maxlen=self.MAX_UNDO_DEPTH)
        # Stamp an initial snapshot so the very first edit is undoable.
        self._snapshot()

    @classmethod
    def from_gpickle(
        cls,
        graph_path: str | Path,
        mask_path: str | Path | None = None,
        distance_map_path: str | Path | None = None,
    ) -> "GraphEditor":
        """Load a graph (and optional mask / distance map) from disk."""
        graph = _load_gpickle(Path(graph_path))
        mask = _load_image(Path(mask_path)) if mask_path else None
        dmap = (_load_image(Path(distance_map_path)).astype(np.float32)
                 if distance_map_path else None)
        return cls(graph, mask=mask, distance_map=dmap)

    # ── Snapshot / undo ────────────────────────────────────────────

    def _snapshot(self) -> None:
        """Save a deep copy of (graph, selection) onto the undo stack."""
        self._undo_stack.append((deepcopy(self.graph), set(self.selected)))

    def undo(self) -> bool:
        """Restore the snapshot saved before the most recent mutation.
        No-op if the stack has only the initial state.  Returns True
        if anything was reverted.

        Snapshots are taken BEFORE each mutation, so popping the top
        of the stack yields the "pre-mutation" state of the most
        recent edit — exactly what we want to restore.  Peeking at
        stack[-1] after popping would restore two edits back.
        """
        if len(self._undo_stack) <= 1:
            return False
        g, sel = self._undo_stack.pop()
        self.graph = deepcopy(g)
        self.selected = set(sel)
        return True

    # ── Selection ──────────────────────────────────────────────────

    def clear_selection(self) -> None:
        self.selected.clear()

    def select(self, node) -> None:
        if node in self.graph:
            self.selected.add(node)

    def deselect(self, node) -> None:
        self.selected.discard(node)

    def toggle_selection(self, node) -> None:
        if node in self.selected:
            self.selected.remove(node)
        elif node in self.graph:
            self.selected.add(node)

    def select_in_rect(
        self, x0: float, y0: float, x1: float, y1: float,
        additive: bool = False,
    ) -> None:
        """Select every node whose (x, y) lies in the inclusive box.
        `additive=False` clears the existing selection first.
        Coordinates are in graph space (the same frame as `node['x']`
        and `node['y']`)."""
        xl, xh = sorted((x0, x1))
        yl, yh = sorted((y0, y1))
        if not additive:
            self.selected.clear()
        for n, d in self.graph.nodes(data=True):
            x = d.get("x"); y = d.get("y")
            if x is None or y is None:
                continue
            if xl <= float(x) <= xh and yl <= float(y) <= yh:
                self.selected.add(n)

    # ── Hit testing ────────────────────────────────────────────────

    def node_at(
        self, x: float, y: float, tolerance: float = 8.0,
    ) -> Optional[Tuple]:
        """Find the nearest node within `tolerance` of (x, y).
        Returns None if no node qualifies.  Tolerance is in the same
        units as the node coordinates (typically pixels)."""
        best = None
        best_d2 = tolerance * tolerance
        for n, d in self.graph.nodes(data=True):
            nx_ = d.get("x"); ny = d.get("y")
            if nx_ is None or ny is None:
                continue
            dx = float(nx_) - x
            dy = float(ny) - y
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_d2 = d2
                best = n
        return best

    # ── Mutations (each snapshots state before mutating) ───────────

    def _next_node_id(self):
        """Smallest integer not already used as a node id.  Falls back
        to a unique large id if the graph uses non-integer keys."""
        ids = [n for n in self.graph.nodes if isinstance(n, (int, np.integer))]
        if not ids:
            return 0
        return int(max(ids)) + 1

    def add_node(
        self,
        x: float,
        y: float,
        radius: Optional[float] = None,
    ):
        """Add a new node at (x, y) and return its id.  Radius is
        sampled from the distance map if not given and the map is
        loaded; otherwise falls back to DEFAULT_RADIUS."""
        self._snapshot()
        n = self._next_node_id()
        if radius is None and self.distance_map is not None:
            iy = int(round(y))
            ix = int(round(x))
            H, W = self.distance_map.shape
            if 0 <= iy < H and 0 <= ix < W:
                radius = float(self.distance_map[iy, ix])
            else:
                radius = self.DEFAULT_RADIUS
        if radius is None:
            radius = self.DEFAULT_RADIUS
        self.graph.add_node(n, x=float(x), y=float(y), radius=float(radius))
        return n

    def delete_selected(self) -> int:
        """Remove every selected node and its incident edges.  Returns
        the count of nodes deleted.  Clears the selection."""
        if not self.selected:
            return 0
        self._snapshot()
        n_removed = 0
        for node in list(self.selected):
            if node in self.graph:
                self.graph.remove_node(node)
                n_removed += 1
        self.selected.clear()
        return n_removed

    def add_edge(self, u, v) -> bool:
        """Connect two nodes.  No-op if the nodes are the same, the
        edge already exists, or one isn't in the graph.  Edge weight
        is the Euclidean length between the nodes; edge radius is the
        mean of endpoint radii.  Returns True if an edge was added."""
        if u == v or u not in self.graph or v not in self.graph:
            return False
        if self.graph.has_edge(u, v):
            return False
        self._snapshot()
        xu = float(self.graph.nodes[u].get("x", 0.0))
        yu = float(self.graph.nodes[u].get("y", 0.0))
        xv = float(self.graph.nodes[v].get("x", 0.0))
        yv = float(self.graph.nodes[v].get("y", 0.0))
        length = math.hypot(xu - xv, yu - yv)
        ru = float(self.graph.nodes[u].get("radius", self.DEFAULT_RADIUS))
        rv = float(self.graph.nodes[v].get("radius", self.DEFAULT_RADIUS))
        self.graph.add_edge(u, v, weight=length, radius=0.5 * (ru + rv))
        return True

    def connect_selected_pair(self) -> bool:
        """Add an edge between the exactly-two currently-selected
        nodes.  Returns True iff exactly two were selected AND the
        edge was added."""
        if len(self.selected) != 2:
            return False
        a, b = list(self.selected)
        return self.add_edge(a, b)

    # ── Topology helpers ───────────────────────────────────────────

    def find_cycles(self) -> List[List]:
        """Return the cycle basis as a list of node lists.  Empty list
        if the graph is acyclic.  Operates on the undirected version
        even if `self.graph` is a DiGraph."""
        G = self.graph.to_undirected() if self.graph.is_directed() else self.graph
        try:
            return [list(c) for c in nx.cycle_basis(G)]
        except nx.NetworkXNoCycle:
            return []

    def select_cycles(self) -> int:
        """Add every node that participates in any cycle to the
        current selection.  Returns the count of newly-selected
        nodes."""
        before = set(self.selected)
        for cycle in self.find_cycles():
            self.selected.update(cycle)
        return len(self.selected - before)

    def streamline(self) -> int:
        """Collapse every degree-2 node by merging its two edges.
        Edge weight is the sum; edge radius is the length-weighted
        mean.  Returns the number of nodes removed.  Iterates to
        convergence (so a long chain collapses fully, not just one
        layer)."""
        self._snapshot()
        removed = 0
        while True:
            to_collapse = [n for n in self.graph.nodes
                            if self.graph.degree(n) == 2]
            if not to_collapse:
                break
            stepped = False
            for n in to_collapse:
                if n not in self.graph or self.graph.degree(n) != 2:
                    continue  # may have been collapsed in this pass
                u, v = list(self.graph.neighbors(n))
                if u == v:
                    continue  # self-loop refuses to collapse cleanly
                lu = float(self.graph.edges[n, u].get("weight", 1.0))
                lv = float(self.graph.edges[n, v].get("weight", 1.0))
                ru = float(self.graph.edges[n, u].get(
                    "radius", self.DEFAULT_RADIUS))
                rv = float(self.graph.edges[n, v].get(
                    "radius", self.DEFAULT_RADIUS))
                new_len = lu + lv
                new_rad = ((ru * lu + rv * lv) / new_len
                           if new_len > 0 else 0.5 * (ru + rv))
                if not self.graph.has_edge(u, v):
                    self.graph.add_edge(u, v,
                                         weight=new_len, radius=new_rad)
                self.graph.remove_node(n)
                removed += 1
                stepped = True
            if not stepped:
                break
        # If the collapse removed any node in `selected`, drop the
        # stale ids.
        self.selected = {n for n in self.selected if n in self.graph}
        return removed

    def make_digraph(self, root) -> bool:
        """Convert `self.graph` to a directed BFS/DFS tree rooted at
        `root`.  Mirrors gegui's behavior.  Returns True on success.
        No-op if `root` isn't in the graph."""
        if root not in self.graph:
            return False
        self._snapshot()
        T = nx.dfs_tree(self.graph, source=root)
        # Carry over node + edge attributes.
        for n, attrs in T.nodes(data=True):
            attrs.update(self.graph.nodes[n])
        for u, v, attrs in T.edges(data=True):
            if self.graph.has_edge(u, v):
                attrs.update(self.graph.edges[u, v])
            elif self.graph.has_edge(v, u):
                attrs.update(self.graph.edges[v, u])
        self.graph = T
        self.selected = {n for n in self.selected if n in self.graph}
        return True

    # ── Save ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Write the current graph to `path` as a gpickle.  Parent
        dir is created if missing."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_gpickle(self.graph, p)

    # ── Convenience read-only views ────────────────────────────────

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    def __repr__(self) -> str:
        return (f"GraphEditor({self.n_nodes} nodes, {self.n_edges} edges, "
                f"{len(self.selected)} selected, "
                f"undo depth {len(self._undo_stack)})")
