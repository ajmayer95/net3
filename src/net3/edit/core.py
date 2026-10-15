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
        self.selected: Set = set()                  # selected node ids
        self.selected_edges: Set[Tuple] = set()     # canonical (u, v) tuples
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
        """Save a deep copy of (graph, node selection, edge selection)
        onto the undo stack."""
        self._geom_cache = None
        self._undo_stack.append((
            deepcopy(self.graph),
            set(self.selected),
            set(self.selected_edges),
        ))

    def _push_removal(self, nodes: Iterable, edges: Iterable[Tuple]) -> None:
        """Lightweight undo record for deletions: store only the nodes and
        edges about to be removed (with their attributes), instead of a deep
        copy of the whole graph — deletions stay instant on large graphs."""
        G = self.graph
        nodes = [n for n in nodes if n in G]
        edge_set = {self._canonical_edge(u, v) for u, v in edges if G.has_edge(u, v)}
        for n in nodes:
            for nb in G.neighbors(n):
                edge_set.add(self._canonical_edge(n, nb))
        self._geom_cache = None
        self._undo_stack.append(dict(
            kind="removal",
            nodes=[(n, dict(G.nodes[n])) for n in nodes],
            edges=[(u, v, dict(G.edges[u, v])) for u, v in edge_set],
            selected=set(self.selected),
            selected_edges=set(self.selected_edges),
        ))

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
        snap = self._undo_stack.pop()
        self._geom_cache = None
        if isinstance(snap, dict) and snap.get("kind") == "removal":
            for n, attrs in snap["nodes"]:
                self.graph.add_node(n, **attrs)
            for u, v, attrs in snap["edges"]:
                self.graph.add_edge(u, v, **attrs)
            self.selected = set(snap["selected"])
            self.selected_edges = set(snap["selected_edges"])
            return True
        # Older snapshots may have only 2 fields (graph, node-sel) before
        # edge selection was added — tolerate either shape.
        if len(snap) == 3:
            g, sel, sel_e = snap
        else:
            g, sel = snap
            sel_e = set()
        self.graph = deepcopy(g)
        self.selected = set(sel)
        self.selected_edges = set(sel_e)
        return True

    # ── Selection ──────────────────────────────────────────────────

    def clear_selection(self) -> None:
        """Clear BOTH node and edge selections."""
        self.selected.clear()
        self.selected_edges.clear()

    # ── Edge selection ─────────────────────────────────────────────

    def select_edge(self, u, v) -> bool:
        """Select an edge by endpoints (order-insensitive).  Returns
        True if the edge exists and was added."""
        if not self.graph.has_edge(u, v):
            return False
        self.selected_edges.add(self._canonical_edge(u, v))
        return True

    def deselect_edge(self, u, v) -> None:
        self.selected_edges.discard(self._canonical_edge(u, v))

    def toggle_edge_selection(self, u, v) -> None:
        key = self._canonical_edge(u, v)
        if key in self.selected_edges:
            self.selected_edges.remove(key)
        elif self.graph.has_edge(u, v):
            self.selected_edges.add(key)

    def delete_selected_edges(self) -> int:
        """Remove every selected edge (keeping endpoint nodes).
        Returns the count of edges deleted.  Clears edge selection."""
        if not self.selected_edges:
            return 0
        self._push_removal([], list(self.selected_edges))
        n = 0
        for u, v in list(self.selected_edges):
            if self.graph.has_edge(u, v):
                self.graph.remove_edge(u, v)
                n += 1
        self.selected_edges.clear()
        return n

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

    def select_in_polygon(self, poly_xy, additive: bool = False) -> Tuple[int, int]:
        """Select every node inside the polygon `poly_xy` ((N, 2) graph x, y
        vertices — rectangle, ellipse outline or lasso), plus every edge
        whose midpoint lies inside (catches edges that cross the region
        with both endpoints outside).  Returns (n nodes, n edges) selected."""
        from matplotlib.path import Path as _MPath
        poly = _MPath(np.asarray(poly_xy, float))
        node_ids, node_xy, edges_list, ec = self._geometry()
        if not additive:
            self.selected.clear(); self.selected_edges.clear()
        n_in = []
        if len(node_ids):
            n_in = [node_ids[i] for i in np.flatnonzero(poly.contains_points(node_xy))]
        e_in = []
        if len(edges_list):
            mid = 0.5 * (ec[:, :2] + ec[:, 2:])
            e_in = [edges_list[i] for i in np.flatnonzero(poly.contains_points(mid))]
        self.selected |= set(n_in)
        self.selected_edges |= set(e_in)
        return len(n_in), len(e_in)

    def component_sizes(self):
        """[(nodes, total centreline length px)] per connected component,
        largest (by length) first."""
        G = self.graph
        def elen(u, v, d):
            p = d.get("path")
            if p is not None and len(p) >= 2:
                a = np.asarray(p, float)[:, :2]
                return float(np.hypot(*np.diff(a, axis=0).T).sum())
            nu, nv = G.nodes[u], G.nodes[v]
            return float(np.hypot(float(nu.get("x", 0)) - float(nv.get("x", 0)),
                                  float(nu.get("y", 0)) - float(nv.get("y", 0))))
        L = {}
        for u, v, d in G.edges(data=True):
            le = elen(u, v, d); L[u] = L.get(u, 0.0) + le / 2; L[v] = L.get(v, 0.0) + le / 2
        comps = [(c, sum(L.get(n, 0.0) for n in c)) for c in nx.connected_components(G)]
        return sorted(comps, key=lambda t: -t[1])

    def select_small_components(self, max_length_px: float = 0.0,
                                additive: bool = False) -> Tuple[int, int]:
        """Select every connected component other than the largest whose
        total centreline length is <= `max_length_px` (0 → every component
        except the largest).  Delete with `delete_selected`.  Returns
        (n components, n nodes) selected."""
        comps = self.component_sizes()[1:]
        if max_length_px and max_length_px > 0:
            comps = [c for c in comps if c[1] <= max_length_px]
        if not additive:
            self.selected.clear(); self.selected_edges.clear()
        nodes = set().union(*[c for c, _ in comps]) if comps else set()
        self.selected |= nodes
        return len(comps), len(nodes)

    # ── Hit testing ────────────────────────────────────────────────

    def node_at(
        self, x: float, y: float, tolerance: float = 8.0,
    ) -> Optional[Tuple]:
        """Find the nearest node within `tolerance` of (x, y).
        Returns None if no node qualifies.  Tolerance is in the same
        units as the node coordinates (typically pixels)."""
        node_ids, node_xy, _, _ = self._geometry()
        if not len(node_ids):
            return None
        d2 = (node_xy[:, 0] - x) ** 2 + (node_xy[:, 1] - y) ** 2
        i = int(np.argmin(d2))
        return node_ids[i] if d2[i] <= tolerance * tolerance else None

    def _geometry(self):
        """Node and edge coordinate arrays for hit-testing, cached until the
        next mutation (every mutation goes through _snapshot/_push_removal,
        which drop the cache) — clicks stay fast on 100k+ element graphs."""
        if getattr(self, "_geom_cache", None) is None:
            G = self.graph
            node_ids, node_xy = [], []
            for n, d in G.nodes(data=True):
                if d.get("x") is not None and d.get("y") is not None:
                    node_ids.append(n); node_xy.append((float(d["x"]), float(d["y"])))
            edges_list, coords = [], []
            for u, v in G.edges:
                nu = G.nodes[u]; nv = G.nodes[v]
                if None in (nu.get("x"), nu.get("y"), nv.get("x"), nv.get("y")):
                    continue
                edges_list.append(self._canonical_edge(u, v))
                coords.append((float(nu["x"]), float(nu["y"]), float(nv["x"]), float(nv["y"])))
            self._geom_cache = (node_ids, np.asarray(node_xy, float).reshape(-1, 2),
                                edges_list, np.asarray(coords, float).reshape(-1, 4))
        return self._geom_cache

    def vessel_of(self, seed) -> Tuple[List, List[Tuple]]:
        """The vessel (maximal chain through degree-2 nodes) containing `seed`
        — an edge (u, v) or a degree-2 node.  Returns (interior nodes, edges);
        the two end nodes (junctions / tips) are not included in `interior`."""
        G = self.graph
        if isinstance(seed, tuple):
            u, v = seed
            if not G.has_edge(u, v):
                return [], []
            starts = [(u, v), (v, u)]
        else:
            if seed not in G or G.degree(seed) != 2:
                return [], []
            a, b = list(G.neighbors(seed))
            starts = [(seed, a), (seed, b)]
        interior = set() if isinstance(seed, tuple) else {seed}
        edges = set()
        for prev, cur in starts:
            seen = {prev}
            edges.add(self._canonical_edge(prev, cur))
            while G.degree(cur) == 2 and cur not in seen:
                interior.add(cur); seen.add(cur)
                nxt = [w for w in G.neighbors(cur) if w != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                edges.add(self._canonical_edge(prev, cur))
        return list(interior), list(edges)

    def delete_selected_vessels(self) -> Tuple[int, int]:
        """Delete every vessel containing a selected edge or selected
        degree-2 node: all its edges and interior nodes, keeping its two end
        nodes.  Returns (vessels deleted, nodes removed).  Selected junction
        or tip nodes are ignored (ambiguous: they belong to several vessels)."""
        seeds = list(self.selected_edges) + [n for n in self.selected
                                             if n in self.graph and self.graph.degree(n) == 2]
        if not seeds:
            return 0, 0
        all_nodes, all_edges, n_vessels, done = set(), set(), 0, set()
        for s in seeds:
            if s in done:
                continue
            nodes, edges = self.vessel_of(s)
            if not edges:
                continue
            n_vessels += 1
            all_nodes |= set(nodes); all_edges |= set(edges)
            done |= set(edges) | set(nodes)
        if not all_edges:
            return 0, 0
        self._push_removal(all_nodes, all_edges)
        self.graph.remove_edges_from(all_edges)
        self.graph.remove_nodes_from(all_nodes)
        self.selected = {n for n in self.selected if n in self.graph}
        self.selected_edges = {(u, v) for (u, v) in self.selected_edges if self.graph.has_edge(u, v)}
        return n_vessels, len(all_nodes)

    def edge_at(
        self, x: float, y: float, tolerance: float = 8.0,
    ) -> Optional[Tuple]:
        """Find the nearest edge within `tolerance` of (x, y).
        Returns the (u, v) canonical edge tuple, or None.  Distance
        is point-to-segment, vectorised over cached coordinates."""
        _, _, edges_list, c = self._geometry()
        if not len(edges_list):
            return None
        ax, ay, bx, by = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        dx, dy = bx - ax, by - ay
        seg_len2 = np.maximum(dx * dx + dy * dy, 1e-12)
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / seg_len2, 0.0, 1.0)
        px, py = ax + t * dx, ay + t * dy
        d2 = (px - x) ** 2 + (py - y) ** 2
        i = int(np.argmin(d2))
        if d2[i] > tolerance * tolerance:
            return None
        return edges_list[i]

    @staticmethod
    def _canonical_edge(u, v) -> Tuple:
        """Canonical key for an undirected edge — always (min, max) so
        (u, v) and (v, u) share the same tuple."""
        try:
            return (u, v) if u <= v else (v, u)
        except TypeError:
            # Mixed-type node ids — fall back to repr ordering.
            return (u, v) if repr(u) <= repr(v) else (v, u)

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
        snap_window: int = 0,
    ):
        """Add a new node at graph coords (x, y) and return its id.

        Coordinate convention: ``y`` is in net3's math-y convention
        (``y = image_height - row``).  When sampling the distance map
        we convert to image row internally.

        Parameters
        ----------
        x, y : float
            Click position in graph coordinates.
        radius : float, optional
            Explicit radius.  If None and a distance map is loaded,
            radius is sampled from the map at (x, y) after the snap.
            Falls back to DEFAULT_RADIUS otherwise.
        snap_window : int, default 0
            If positive AND a distance map is loaded, the node is
            placed at the local distance-transform peak within a
            ``(2 * snap_window + 1)`` square centred on the click.
            This snaps clicks to the vessel centerline so newly
            created connections look clean.
        """
        self._snapshot()
        n = self._next_node_id()
        if (snap_window > 0 and self.distance_map is not None):
            x, y = self._snap_to_centerline(x, y, snap_window)
        if radius is None and self.distance_map is not None:
            H, W = self.distance_map.shape
            iy = int(round(H - y))      # math y → image row
            ix = int(round(x))
            if 0 <= iy < H and 0 <= ix < W:
                radius = float(self.distance_map[iy, ix])
            else:
                radius = self.DEFAULT_RADIUS
        if radius is None:
            radius = self.DEFAULT_RADIUS
        self.graph.add_node(n, x=float(x), y=float(y), radius=float(radius))
        return n

    def move_node(
        self,
        node,
        x: float,
        y: float,
        snap_window: int = 0,
    ) -> bool:
        """Reposition an existing node to graph coords (x, y) and
        replace every incident edge's ``path`` with a straight line
        connecting the moved node to its neighbour at the new position.

        Coordinate convention: same as ``add_node`` — ``y`` is math-y
        (``y = image_height - row``).

        Behaviour:
            - Updates ``self.graph.nodes[node]['x']`` and ``['y']`` to the
              new coords.
            - If a distance map is loaded and ``snap_window > 0``, the
              new position is snapped to the local centerline peak like
              ``add_node`` does.
            - For every incident edge (``node``, ``nbr``), the ``path``
              attribute is REPLACED with a two-point straight segment
              between ``nbr`` and ``node``, in the same direction as the
              old path (``nbr → node`` if there was none).  Radius attributes
              stay as-is (the geometry-derivable ones will be recomputed
              downstream if you run ``patch_R_aggregations`` or similar).
            - The edge ``length`` attribute (if present) is recomputed as
              the Euclidean distance between the two endpoints.

        Returns True if the move succeeded, False if the node does not
        exist.
        """
        if node not in self.graph:
            return False
        self._snapshot()
        if snap_window > 0 and self.distance_map is not None:
            x, y = self._snap_to_centerline(x, y, snap_window)
        # Update the moved node's coords.
        self.graph.nodes[node]['x'] = float(x)
        self.graph.nodes[node]['y'] = float(y)
        # Replace every incident edge's path with a straight line, keeping
        # the old path's direction: downstream OF signs flow along path
        # order, so reversing a path flips the measured flux sign.
        for nbr in list(self.graph.neighbors(node)):
            nbx = float(self.graph.nodes[nbr]['x'])
            nby = float(self.graph.nodes[nbr]['y'])
            edata = self.graph.edges[node, nbr]
            ends = [[nbx, nby], [float(x), float(y)]]
            old = edata.get('path')
            if old is not None and len(old) >= 2:
                o = np.asarray(old, dtype=float)
                if (math.hypot(o[-1, 0] - nbx, o[-1, 1] - nby)
                        < math.hypot(o[0, 0] - nbx, o[0, 1] - nby)):
                    ends = ends[::-1]      # old path ran node → nbr
            path = np.array(ends, dtype=float)
            edata['path'] = path
            edata['length'] = float(math.hypot(x - nbx, y - nby))
            # Drop any per-vertex arrays that no longer match the new
            # 2-vertex path — measurements etc. become stale; the sync
            # step (or a `radii` re-sampling pass) will refresh them.
            for stale_key in ('radii', 'orientations', 'radii_raw'):
                edata.pop(stale_key, None)
        return True

    def _snap_to_centerline(
        self, x: float, y: float, window: int,
    ) -> Tuple[float, float]:
        """Move (x, y) to the local distance-transform maximum within
        ``window`` pixels.  Returns the snapped graph coordinates.
        No-op if no distance map."""
        if self.distance_map is None:
            return x, y
        H, W = self.distance_map.shape
        iy = int(round(H - y))
        ix = int(round(x))
        y0, y1 = max(0, iy - window), min(H, iy + window + 1)
        x0, x1 = max(0, ix - window), min(W, ix + window + 1)
        if y1 <= y0 or x1 <= x0:
            return x, y
        patch = self.distance_map[y0:y1, x0:x1]
        if patch.size == 0:
            return x, y
        flat = int(np.argmax(patch))
        dy, dx = np.unravel_index(flat, patch.shape)
        new_iy = y0 + int(dy)
        new_ix = x0 + int(dx)
        # image row → math y
        return float(new_ix), float(H - new_iy)

    def delete_selected(self) -> int:
        """Remove every selected node and its incident edges.  Returns
        the count of nodes deleted.  Clears node selection AND drops
        any selected edges that no longer exist."""
        if not self.selected:
            return 0
        self._push_removal(list(self.selected), [])
        n_removed = 0
        for node in list(self.selected):
            if node in self.graph:
                self.graph.remove_node(node)
                n_removed += 1
        self.selected.clear()
        # Drop edges whose endpoints were just removed.
        self.selected_edges = {
            (u, v) for (u, v) in self.selected_edges
            if self.graph.has_edge(u, v)
        }
        return n_removed

    def add_edge(self, u, v) -> bool:
        """Connect two nodes.  No-op if the nodes are the same, the
        edge already exists, or one isn't in the graph.

        Sets `weight` = `length` = Euclidean distance, `radius` = mean
        of endpoint radii, and `path` = a 2-point straight-line polyline
        ([[xu,yu], [xv,yv]]).  The `path` attribute matches the schema
        produced by `net3.collapse_to_branch_graph`, so downstream
        consumers that expect every edge to carry a centerline polyline
        (PerTileFlow's ROI generation, for instance) don't need a
        special case for editor-added edges.

        Returns True if an edge was added.
        """
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
        # Densify the straight-line path so downstream consumers that
        # require N>=5 sample points (e.g. PerTileFlow's `get_chain_coords`,
        # which rejects edges with len(coords) < 5) accept editor-added
        # edges.  ~1 sample every 2 px, min 5, max 200.
        n_samples = int(np.clip(round(length / 2.0), 5, 200))
        ts = np.linspace(0.0, 1.0, n_samples)
        path = np.column_stack([xu + ts * (xv - xu),
                                  yu + ts * (yv - yu)])
        self.graph.add_edge(
            u, v,
            weight=length,
            length=length,
            radius=0.5 * (ru + rv),
            path=path,
        )
        return True

    def connect_selected_pair(self) -> bool:
        """Add an edge between the exactly-two currently-selected
        nodes.  Returns True iff exactly two were selected AND the
        edge was added."""
        if len(self.selected) != 2:
            return False
        a, b = list(self.selected)
        return self.add_edge(a, b)

    # ── Gap bridging ──────────────────────────────────────────────

    def find_gap_candidates(
        self,
        d_max: float = 15.0,
        require_different_components: bool = True,
        topo_min: Optional[int] = None,
        k_neighbors: int = 6,
        allow_leaf_to_any: bool = False,
        d_max_leaf_to_any: Optional[float] = None,
    ) -> List[Tuple]:
        """Find pairs of nodes that are geometrically close but
        topologically far — reasonable bridge candidates.

        By default only pairs of **degree-1 leaves** are considered
        (leaf-to-leaf, the classic tip-to-tip gap).  Set
        ``allow_leaf_to_any=True`` to also emit **leaf → interior-node**
        candidates (T-junction case where a small tributary was
        traced but not connected to the trunk it should join).  Those
        candidates use a separate, typically stricter, `d_max`.

        Parameters
        ----------
        d_max : float
            Maximum Euclidean distance (px) between paired leaves.
        require_different_components : bool
            If True, only pair leaves that live in different connected
            components of the graph.
        topo_min : int, optional
            If set (and same-component pairs are allowed), require the
            shortest-path hop count between the two nodes to exceed
            this value.
        k_neighbors : int
            Number of nearest neighbours to consider per leaf.
        allow_leaf_to_any : bool
            If True, also pair every leaf with its nearest non-leaf
            node under the stricter `d_max_leaf_to_any` threshold.
        d_max_leaf_to_any : float, optional
            Distance threshold for leaf → interior pairs.  Defaults to
            `0.6 * d_max` (stricter than tip-to-tip so we do not
            propose stubby side connections everywhere).

        Returns
        -------
        candidates : list of (u, v, dist) tuples
            Deduplicated, sorted by ascending distance.
        """
        from scipy.spatial import cKDTree
        deg = dict(self.graph.degree())
        leaves = [n for n, d in deg.items() if d == 1]
        if not leaves:
            return []

        # Component labels — computed once, used by both leaf-leaf and
        # leaf-any paths below.
        comp_of = {}
        for ci, comp in enumerate(nx.connected_components(self.graph)):
            for n in comp:
                comp_of[n] = ci

        def _passes_topo(u, v) -> bool:
            same_cc = comp_of.get(u) == comp_of.get(v)
            if require_different_components and same_cc:
                if topo_min is None:
                    return False
            if topo_min is not None:
                if same_cc:
                    try:
                        hops = nx.shortest_path_length(self.graph, u, v)
                    except nx.NetworkXNoPath:
                        hops = float("inf")
                    if hops <= topo_min:
                        return False
            return True

        seen = set()
        out: List[Tuple] = []

        # ── leaf → leaf ──────────────────────────────────────
        if len(leaves) >= 2:
            leaf_pts = np.array([[float(self.graph.nodes[n].get("x", 0.0)),
                                    float(self.graph.nodes[n].get("y", 0.0))]
                                   for n in leaves])
            tree_l = cKDTree(leaf_pts)
            k = min(k_neighbors + 1, len(leaves))
            dists, idxs = tree_l.query(leaf_pts, k=k)
            for i, u in enumerate(leaves):
                for j_idx in range(1, k):
                    j = int(idxs[i, j_idx])
                    d = float(dists[i, j_idx])
                    if not np.isfinite(d) or d > d_max:
                        continue
                    v = leaves[j]
                    key = (u, v) if u < v else (v, u)
                    if key in seen or self.graph.has_edge(u, v):
                        continue
                    if not _passes_topo(u, v):
                        continue
                    seen.add(key)
                    out.append((u, v, d))

        # ── leaf → any interior node (T-junction) ────────────
        if allow_leaf_to_any:
            d_lta = (d_max_leaf_to_any if d_max_leaf_to_any is not None
                      else 0.6 * d_max)
            all_nodes = list(self.graph.nodes)
            all_pts = np.array([[float(self.graph.nodes[n].get("x", 0.0)),
                                   float(self.graph.nodes[n].get("y", 0.0))]
                                  for n in all_nodes])
            tree_a = cKDTree(all_pts)
            node_to_idx = {n: i for i, n in enumerate(all_nodes)}
            for u in leaves:
                iu = node_to_idx[u]
                # k neighbours in the full-node tree; +1 for self
                k = min(k_neighbors + 1, len(all_nodes))
                dists, idxs = tree_a.query(all_pts[iu], k=k)
                for pos in range(1, k):
                    j = int(idxs[pos]); d = float(dists[pos])
                    if not np.isfinite(d) or d > d_lta:
                        continue
                    v = all_nodes[j]
                    if v == u:
                        continue
                    # Skip if v is also a leaf — leaf-leaf handled above
                    if deg.get(v, 0) <= 1:
                        continue
                    key = (u, v) if u < v else (v, u)
                    if key in seen or self.graph.has_edge(u, v):
                        continue
                    if not _passes_topo(u, v):
                        continue
                    seen.add(key)
                    out.append((u, v, d))

        out.sort(key=lambda t: t[2])
        return out

    def bridge_gaps(self, candidates: Iterable[Tuple]) -> int:
        """Apply `add_edge` to every (u, v[, dist]) in `candidates`.

        Takes a single undo snapshot for the whole batch (not per pair),
        so a Bridge-All op is one undo step.  Returns the number of
        edges actually added.
        """
        pairs = [(t[0], t[1]) for t in candidates
                  if len(t) >= 2 and t[0] in self.graph and t[1] in self.graph]
        if not pairs:
            return 0
        self._snapshot()
        n = 0
        for u, v in pairs:
            if u == v or self.graph.has_edge(u, v):
                continue
            xu = float(self.graph.nodes[u].get("x", 0.0))
            yu = float(self.graph.nodes[u].get("y", 0.0))
            xv = float(self.graph.nodes[v].get("x", 0.0))
            yv = float(self.graph.nodes[v].get("y", 0.0))
            length = math.hypot(xu - xv, yu - yv)
            ru = float(self.graph.nodes[u].get("radius", self.DEFAULT_RADIUS))
            rv = float(self.graph.nodes[v].get("radius", self.DEFAULT_RADIUS))
            n_samples = int(np.clip(round(length / 2.0), 5, 200))
            ts = np.linspace(0.0, 1.0, n_samples)
            path = np.column_stack([xu + ts * (xv - xu),
                                      yu + ts * (yv - yu)])
            # Bridges represent the *tributary* / stub joining another
            # vessel, not a widening.  Use the smaller endpoint radius
            # so a leaf→trunk bridge doesn't inflate to the trunk width.
            self.graph.add_edge(u, v, weight=length, length=length,
                                 radius=float(min(ru, rv)), path=path)
            n += 1
        return n

    # ── True-tip marking (for prune-unmarked workflow) ─────────────

    def mark_selected_as_true_tips(self) -> int:
        """Set ``true_tip=True`` on every currently-selected node.
        Returns the count of nodes flagged.  Snapshots undo state.
        Nodes that aren't degree-1 are still flagged — the flag is a
        user annotation, not a topology check — so a junction the user
        deems important is also preservable."""
        if not self.selected:
            return 0
        self._snapshot()
        n = 0
        for node in self.selected:
            if node in self.graph:
                self.graph.nodes[node]["true_tip"] = True
                n += 1
        return n

    def unmark_selected_as_true_tips(self) -> int:
        """Clear ``true_tip`` on every currently-selected node."""
        if not self.selected:
            return 0
        self._snapshot()
        n = 0
        for node in self.selected:
            if node in self.graph and self.graph.nodes[node].get("true_tip"):
                del self.graph.nodes[node]["true_tip"]
                n += 1
        return n

    def true_tip_nodes(self) -> Set:
        """All nodes flagged with ``true_tip=True``."""
        return {n for n, d in self.graph.nodes(data=True)
                 if d.get("true_tip")}

    def shorten_tip(self, node, trim_px: float) -> bool:
        """Move a degree-1 tip inward along its incident edge's path
        by ``trim_px`` (arc length) and truncate the path.  Resamples
        the tip's `radius` from the distance map at the new position
        if one is loaded.  Returns True if the tip was shortened.

        Purpose: triangulation-based vectorizers place tips exactly at
        the outermost mask pixel, where the vessel has tapered to
        near-zero.  M5's Q integration then uses a nonsense R (=1-5 px).
        Shortening pulls the tip back to where R is stable — the
        anatomical tip presence is preserved; only the tapered fringe
        is dropped.  Path is truncated + rebound to the new tip
        position; edge length updates.
        """
        if node not in self.graph or self.graph.degree(node) != 1:
            return False
        nb = next(iter(self.graph.neighbors(node)))
        ed = self.graph.edges[node, nb]
        path = ed.get("path")
        if path is None or len(path) < 3:
            return False
        path = np.asarray(path, dtype=float)
        # Determine which end of path is at the tip vs the neighbour.
        # Path indexing runs from tip → neighbour (index 0 = tip).
        tip_xy = np.array([self.graph.nodes[node]["x"],
                            self.graph.nodes[node]["y"]])
        d_start = np.hypot(path[0, 0] - tip_xy[0], path[0, 1] - tip_xy[1])
        d_end   = np.hypot(path[-1, 0] - tip_xy[0], path[-1, 1] - tip_xy[1])
        reversed_ = d_end < d_start
        walk = path[::-1] if reversed_ else path
        # Walk arc-length from tip end until trim_px reached.
        seg = np.hypot(np.diff(walk[:, 0]), np.diff(walk[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] <= trim_px:
            # Can't shorten — the trim would consume the whole edge.
            return False
        # Find first index where cum > trim_px; interpolate for the
        # exact new-tip position (fractional between two path points).
        i = int(np.searchsorted(cum, trim_px, side="right"))
        # i is now the first index at arc >= trim_px; new tip goes at
        # the linear interpolation between i-1 and i.
        excess = trim_px - cum[i - 1]
        seg_len = cum[i] - cum[i - 1]
        t = 0.0 if seg_len == 0 else excess / seg_len
        new_xy = walk[i - 1] + t * (walk[i] - walk[i - 1])
        # Build the truncated path from new_xy → neighbour.
        new_walk = np.vstack([new_xy[None, :], walk[i:]])
        new_path = new_walk[::-1] if reversed_ else new_walk
        # New length = sum of remaining segment lengths.
        new_length = float(np.sum(np.hypot(
            np.diff(new_walk[:, 0]), np.diff(new_walk[:, 1]))))
        self._snapshot()
        self.graph.nodes[node]["x"] = float(new_xy[0])
        self.graph.nodes[node]["y"] = float(new_xy[1])
        # Resample radius from distance map at the new position.
        if self.distance_map is not None:
            H, W = self.distance_map.shape
            iy = int(round(H - new_xy[1]))
            ix = int(round(new_xy[0]))
            if 0 <= iy < H and 0 <= ix < W:
                self.graph.nodes[node]["radius"] = float(
                    self.distance_map[iy, ix])
        ed["path"] = new_path
        ed["length"] = new_length
        ed["weight"] = new_length
        # Also refresh edge['radius'] — it was set at vectorize time from
        # the FULL path's EDT mean and includes the tapered fringe we
        # just trimmed.  Resample from the distance map along the new
        # path and take the median (robust to any straggler low-EDT
        # samples near path endpoints).  M5 reads edge['radius'] as its
        # per-tile mean_radius, so a stale value here inflates Q by
        # (R_stale/R_true)².
        if self.distance_map is not None:
            H, W = self.distance_map.shape
            xs = np.clip(np.round(new_path[:, 0]).astype(int), 0, W - 1)
            ys = np.clip(np.round(H - new_path[:, 1]).astype(int),
                          0, H - 1)
            r_samples = self.distance_map[ys, xs]
            interior = r_samples[r_samples > 0]
            if len(interior) >= 3:
                ed["radius"] = float(np.median(interior))
        return True

    def shorten_selected_tips(self, trim_px: float) -> int:
        """Apply :meth:`shorten_tip` to every selected degree-1 node.
        Returns count of tips actually shortened."""
        if not self.selected:
            return 0
        n = 0
        for node in list(self.selected):
            if self.shorten_tip(node, trim_px):
                n += 1
        return n

    def prune_unmarked_tips(self, cascade: bool = True) -> int:
        """Delete every degree-1 node whose ``true_tip`` attribute is
        not True.  Returns total count of pruned nodes across all
        iterations.  Snapshots undo once for the whole operation.

        Parameters
        ----------
        cascade : bool
            If True (default), iterate to convergence — after deleting
            the current unmarked tips, any node that just became
            degree-1 (and isn't marked) is also deleted, until no more
            unmarked tips remain.  This peels off whole spur chains
            without requiring the user to hit the button repeatedly.
            If False, runs one pass only (legacy behaviour).
        """
        first_pass = [n for n in list(self.graph.nodes)
                       if self.graph.degree(n) == 1
                       and not self.graph.nodes[n].get("true_tip")]
        if not first_pass:
            return 0
        self._snapshot()
        total = 0
        while True:
            to_delete = [n for n in list(self.graph.nodes)
                          if self.graph.degree(n) == 1
                          and not self.graph.nodes[n].get("true_tip")]
            if not to_delete:
                break
            self.graph.remove_nodes_from(to_delete)
            self.selected -= set(to_delete)
            total += len(to_delete)
            if not cascade:
                break
        return total

    def merge_close_junctions(self, d_max: float = 3.0) -> int:
        """Contract every group of degree-≥3 junction nodes that lie
        within `d_max` px of each other into a single node.

        Uses union-find over junction pairs whose Euclidean distance
        is ≤ `d_max`.  Each cluster becomes one node whose position is
        the centroid of the group and whose radius is the **max** of
        the members' radii (bigger vessel wins).  All incident edges
        from any cluster member are re-attached to the merged node.
        Self-loops and duplicate edges that result are removed.

        Returns the number of nodes removed (i.e. members - clusters
        that actually contained > 1 junction).

        Notes
        -----
        This is the fix for the vectorize artifact where a fat vessel
        blob produces N triangles sharing a triple-point, each with
        its own degree-3 node at (nearly) the same pixel.
        """
        from scipy.spatial import cKDTree
        junctions = [n for n, d in self.graph.degree() if d >= 3]
        if len(junctions) < 2 or d_max <= 0:
            return 0
        pts = np.array([[float(self.graph.nodes[n].get("x", 0.0)),
                          float(self.graph.nodes[n].get("y", 0.0))]
                         for n in junctions])
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=float(d_max), output_type="ndarray")
        if len(pairs) == 0:
            return 0

        # Union-find over junction indices
        parent = list(range(len(junctions)))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for i, j in pairs:
            union(int(i), int(j))

        # Group junction node-ids by cluster root
        clusters: dict = {}
        for idx, node in enumerate(junctions):
            clusters.setdefault(find(idx), []).append(node)
        clusters = {k: v for k, v in clusters.items() if len(v) > 1}
        if not clusters:
            return 0

        self._snapshot()
        removed = 0

        def _snap_path_endpoint(path, old_xy, new_xy, tol=1e-6):
            """Rewrite whichever endpoint of `path` is at `old_xy` to
            sit at `new_xy` instead. Preserves the polyline's shape by
            translating only that endpoint (not the whole path)."""
            if path is None: return path
            p = np.asarray(path, dtype=float).copy()
            if p.ndim != 2 or p.shape[1] < 2 or len(p) < 2: return p
            d_start = float(np.hypot(p[0, 0] - old_xy[0], p[0, 1] - old_xy[1]))
            d_end   = float(np.hypot(p[-1, 0] - old_xy[0], p[-1, 1] - old_xy[1]))
            if d_start <= d_end:
                p[0, 0] = new_xy[0]; p[0, 1] = new_xy[1]
            else:
                p[-1, 0] = new_xy[0]; p[-1, 1] = new_xy[1]
            return p

        for root, members in clusters.items():
            # Keep the first member as the surviving node.
            keeper = members[0]
            others = members[1:]
            xs = [float(self.graph.nodes[n].get("x", 0.0)) for n in members]
            ys = [float(self.graph.nodes[n].get("y", 0.0)) for n in members]
            rs = [float(self.graph.nodes[n].get("radius",
                                                  self.DEFAULT_RADIUS))
                   for n in members]
            # Old keeper position (before centroid shift) — used to snap
            # paths that already terminated at keeper.
            keeper_old = (float(self.graph.nodes[keeper]["x"]),
                          float(self.graph.nodes[keeper]["y"]))
            new_xy = (float(np.mean(xs)), float(np.mean(ys)))
            self.graph.nodes[keeper]["x"] = new_xy[0]
            self.graph.nodes[keeper]["y"] = new_xy[1]
            self.graph.nodes[keeper]["radius"] = float(max(rs))
            # Snap paths on edges that already touch `keeper` from the
            # old keeper position onto the new centroid.
            for nbr in list(self.graph.neighbors(keeper)):
                ed = self.graph.edges[keeper, nbr]
                p = ed.get("path")
                if p is not None:
                    ed["path"] = _snap_path_endpoint(p, keeper_old, new_xy)
            for n in others:
                # Snap paths on edges that touch `n` from the old n
                # position onto keeper's new centroid.  Doing this
                # BEFORE reattaching so we preserve the polyline shape.
                n_old_xy = (float(self.graph.nodes[n].get("x", 0.0)),
                            float(self.graph.nodes[n].get("y", 0.0)))
                for nbr in list(self.graph.neighbors(n)):
                    if nbr == keeper or nbr == n:
                        continue
                    if not self.graph.has_edge(keeper, nbr):
                        attrs = dict(self.graph.edges[n, nbr])
                        # Recompute length as Euclidean distance from
                        # keeper's new centroid position; radius as
                        # max of any parallel edges (see below).
                        xk = float(self.graph.nodes[keeper]["x"])
                        yk = float(self.graph.nodes[keeper]["y"])
                        xn = float(self.graph.nodes[nbr].get("x", 0.0))
                        yn = float(self.graph.nodes[nbr].get("y", 0.0))
                        L = math.hypot(xk - xn, yk - yn)
                        attrs["weight"] = L
                        attrs["length"] = L
                        # Snap the reattached polyline's n-side endpoint
                        # onto the keeper's new centroid; nbr end is
                        # untouched.
                        p = attrs.get("path")
                        if p is not None:
                            attrs["path"] = _snap_path_endpoint(
                                p, n_old_xy, new_xy)
                        self.graph.add_edge(keeper, nbr, **attrs)
                    else:
                        # Merge: max radius, min length wins
                        e_keep = self.graph.edges[keeper, nbr]
                        e_new = self.graph.edges[n, nbr]
                        e_keep["radius"] = max(
                            float(e_keep.get("radius", self.DEFAULT_RADIUS)),
                            float(e_new.get("radius", self.DEFAULT_RADIUS)))
                        e_keep["weight"] = min(
                            float(e_keep.get("weight", 1.0)),
                            float(e_new.get("weight", 1.0)))
                        if "length" in e_keep or "length" in e_new:
                            e_keep["length"] = e_keep["weight"]
                self.graph.remove_node(n)
                removed += 1
        self.selected = {n for n in self.selected if n in self.graph}
        return removed

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
