"""
Rediscretize a dense vessel graph into ~uniform segments.

Takes a dense graph (many nodes) and resamples it so edges are
approximately `char_length` pixels apart. Useful for consistent
sampling for flow analysis.
"""

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from typing import Tuple, Optional
from tqdm import tqdm


def build_radius_kdtree(graph: nx.Graph) -> Tuple[KDTree, np.ndarray]:
    """Build KDTree for fast radius lookup from nearest node."""
    pts, radii = [], []
    for _, data in graph.nodes(data=True):
        pts.append((float(data["x"]), float(data["y"])))
        radii.append(float(data.get("radius", 1.0)))
    return KDTree(np.asarray(pts)), np.asarray(radii, dtype=float)


def nearest_radius(x: float, y: float, tree: KDTree, radius_array: np.ndarray) -> float:
    """Get radius from nearest node."""
    _, idx = tree.query((x, y))
    return float(radius_array[int(idx)])


def _trace_chain(G: nx.Graph, start: int, nbr: int):
    """Trace a degree-2 chain from start→…→end, returning path list."""
    path = [start, nbr]
    visited = {start, nbr}
    curr = nbr
    while G.degree(curr) == 2:
        nxt = [n for n in G.neighbors(curr) if n not in visited]
        if len(nxt) != 1:
            break
        visited.add(curr)
        curr = nxt[0]
        path.append(curr)
    return path


def _segment_points_along_polyline(coords: np.ndarray, target_len: float) -> np.ndarray:
    """Return interior sample points along a polyline at ~target_len spacing."""
    if coords.shape[0] < 2:
        return np.empty((0, 2), dtype=float)

    seg = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    arclen = np.insert(np.cumsum(seg), 0, 0.0)
    total_L = float(arclen[-1])

    if total_L <= target_len:
        return np.empty((0, 2), dtype=float)

    n_seg = max(1, int(round(total_L / target_len)))
    samples = np.linspace(0.0, total_L, n_seg + 1)[1:-1]  # interior only

    out = []
    j = 0
    for s in samples:
        while not (arclen[j] <= s <= arclen[j + 1]):
            j += 1
        denom = (arclen[j + 1] - arclen[j]) or 1e-12
        frac = (s - arclen[j]) / denom
        pt = coords[j] * (1 - frac) + coords[j + 1] * frac
        out.append(pt)
    return np.asarray(out, dtype=float)


def rediscretize_graph(
    graph: nx.Graph,
    char_length_px: float = 15.0,
    radius_field: str = "radius",
    show_progress: bool = True,
) -> nx.Graph:
    """
    Rediscretize graph to ~uniform segment lengths.

    Walks each chain between junctions (degree != 2) and inserts nodes
    at approximately `char_length_px` intervals.

    Parameters
    ----------
    graph : nx.Graph
        Input dense graph with 'x', 'y', 'radius' attributes
    char_length_px : float
        Target segment length in pixels
    radius_field : str
        Name of radius attribute
    show_progress : bool
        Show progress bar

    Returns
    -------
    nx.Graph
        Rediscretized graph with uniform segments
    """
    newG = nx.Graph()

    # Ensure x,y are floats
    for n, d in graph.nodes(data=True):
        if "x" not in d or "y" not in d:
            raise ValueError(f"Node {n} missing 'x'/'y' attributes.")
        d["x"] = float(d["x"])
        d["y"] = float(d["y"])

    tree, radius_arr = build_radius_kdtree(graph)
    junctions = [n for n, deg in graph.degree() if deg != 2]
    seen = set()

    it = tqdm(junctions, desc="Rediscretizing", disable=not show_progress)
    for start in it:
        for nbr in graph.neighbors(start):
            if (start, nbr) in seen or (nbr, start) in seen:
                continue
            seen.add((start, nbr))

            path = _trace_chain(graph, start, nbr)
            end = path[-1]

            x0, y0 = graph.nodes[start]["x"], graph.nodes[start]["y"]
            x1, y1 = graph.nodes[end]["x"], graph.nodes[end]["y"]
            r0 = nearest_radius(x0, y0, tree, radius_arr)
            r1 = nearest_radius(x1, y1, tree, radius_arr)

            # Add endpoints (junctions)
            newG.add_node(start, x=x0, y=y0, **{radius_field: r0})
            newG.add_node(end, x=x1, y=y1, **{radius_field: r1})

            # Original polyline coords along path
            coords = np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in path], dtype=float)

            # Interior samples ~char_length apart
            inter = _segment_points_along_polyline(coords, char_length_px)

            prev = start
            prev_r = r0

            # Add interior nodes & edges
            for i, (xx, yy) in enumerate(inter, start=1):
                nid = f"int_{start}_{end}_{i}"
                rr = nearest_radius(xx, yy, tree, radius_arr)
                newG.add_node(nid, x=float(xx), y=float(yy), **{radius_field: rr})
                L = float(np.hypot(xx - newG.nodes[prev]["x"], yy - newG.nodes[prev]["y"]))
                edge_r = 0.5 * (prev_r + rr)
                newG.add_edge(prev, nid, length=L, **{radius_field: edge_r})
                prev, prev_r = nid, rr

            # Final edge to end
            Lf = float(np.hypot(x1 - newG.nodes[prev]["x"], y1 - newG.nodes[prev]["y"]))
            edge_rf = 0.5 * (prev_r + r1)
            newG.add_edge(prev, end, length=Lf, **{radius_field: edge_rf})

    return newG


def merge_near_duplicate_nodes(G: nx.Graph, tol_px: float = 1e-3) -> Tuple[nx.Graph, int]:
    """
    Merge nodes whose (x,y) are within tol_px.

    Returns (graph, n_merged).
    """
    nodes = list(G.nodes())
    coords = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes], dtype=float)
    keys = np.round(coords / tol_px).astype(np.int64)

    grid_to_rep = {}
    merged = 0

    for node, key in zip(nodes, map(tuple, keys)):
        if key not in grid_to_rep:
            grid_to_rep[key] = node
        else:
            rep = grid_to_rep[key]
            # Rewire edges
            for nbr, ed in list(G[node].items()):
                if nbr == rep:
                    continue
                if not G.has_edge(rep, nbr):
                    G.add_edge(rep, nbr, **ed)
            G.remove_node(node)
            merged += 1

    return G, merged


def reindex_nodes_integer(G: nx.Graph, first_label: int = 0) -> nx.Graph:
    """Relabel nodes to sequential integers."""
    return nx.convert_node_labels_to_integers(G, first_label=first_label)


def flip_y_coordinates(G: nx.Graph, height: Optional[int] = None) -> nx.Graph:
    """
    Flip Y coordinates (for image coordinate alignment).

    If height is provided, flips about (height - 1).
    Otherwise flips about max(y) in the graph.
    """
    if height is not None:
        y_about = height - 1
    else:
        y_about = max(float(d["y"]) for _, d in G.nodes(data=True))

    for _, data in G.nodes(data=True):
        data["y"] = float(y_about) - float(data["y"])

    G.graph["y_flip_about"] = float(y_about)
    return G
