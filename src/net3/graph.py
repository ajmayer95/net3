"""
Graph creation and manipulation for vectorized networks.

Creates NetworkX graphs from triangle adjacency, with node coordinates
and edge properties (length, radius).
"""

import math
import numpy as np
import networkx as nx
import scipy.sparse
from typing import List, Optional


def create_graph(adjacency_matrix, triangles: List, image_height: int) -> nx.Graph:
    """
    Create NetworkX graph from triangle adjacency matrix.

    Nodes get (x, y) coordinates and radius from triangle centers.
    Edges get length (Euclidean distance) and radius (mean of endpoint radii).

    Parameters
    ----------
    adjacency_matrix : scipy.sparse matrix
        Triangle neighborhood matrix
    triangles : List
        List of triangle objects with centers and radii
    image_height : int
        Image height for y-coordinate adjustment

    Returns
    -------
    nx.Graph
        Network graph with node/edge attributes
    """
    G = nx.Graph(adjacency_matrix)

    # Set node coordinates from triangle centers
    x_coords = {i: t.get_center().get_x() for i, t in enumerate(triangles) if i in G.nodes()}
    y_coords = {i: image_height - t.get_center().get_y() for i, t in enumerate(triangles) if i in G.nodes()}
    radii = {i: t.get_radius() for i, t in enumerate(triangles) if i in G.nodes()}

    nx.set_node_attributes(G, x_coords, 'x')
    nx.set_node_attributes(G, y_coords, 'y')
    nx.set_node_attributes(G, radii, 'radius')

    # Set edge attributes
    edge_radius = {}
    edge_length = {}

    for u, v in G.edges():
        # Radius = mean of endpoint radii
        r1 = G.nodes[u].get('radius', 1)
        r2 = G.nodes[v].get('radius', 1)
        edge_radius[(u, v)] = (r1 + r2) / 2.0

        # Length = Euclidean distance
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        edge_length[(u, v)] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    nx.set_edge_attributes(G, edge_radius, 'radius')
    nx.set_edge_attributes(G, edge_length, 'weight')

    return G


def remove_redundant_nodes(G: nx.Graph, mode: str = "all", verbose: bool = False) -> nx.Graph:
    """
    Remove degree-2 nodes by merging their edges.

    Degree-2 nodes are "redundant" - they just connect two edges.
    Removing them simplifies the graph while preserving topology.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    mode : str
        "all" - remove all degree-2 nodes
        "half" - remove half (2 iterations)
        "none" - do nothing
    verbose : bool
        Print progress

    Returns
    -------
    nx.Graph
        Simplified graph
    """
    if isinstance(G, scipy.sparse.lil_matrix):
        G = nx.Graph(G)

    if mode == "none":
        return G

    iteration = 0
    prev_order = G.order()

    while True:
        if mode == "half" and iteration > 2:
            break

        nodes_to_remove = []

        for node in list(G.nodes()):
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 2:
                n1, n2 = neighbors

                # Get edge properties
                w1 = G[node][n1].get('weight', 1)
                w2 = G[node][n2].get('weight', 1)
                r1 = G[node][n1].get('radius', 1)
                r2 = G[node][n2].get('radius', 1)

                # Combined edge properties
                new_length = w1 + w2
                if new_length == 0:
                    new_length = 1
                new_radius = (r1 * w1 + r2 * w2) / new_length

                # Add direct edge between neighbors
                G.add_edge(n1, n2, weight=new_length, radius=new_radius)
                nodes_to_remove.append(node)

        # Remove nodes (careful not to remove all if only 2 left)
        if len(nodes_to_remove) == len(G.nodes()) - 1:
            G.remove_node(nodes_to_remove[0])
        else:
            for node in nodes_to_remove:
                if node in G:
                    G.remove_node(node)

        new_order = G.order()
        if verbose:
            print(f"  Iteration {iteration}: {prev_order} -> {new_order} nodes")

        # Convergence: a full pass that removed nothing → done.
        if new_order == prev_order:
            break

        prev_order = new_order
        iteration += 1
        # NOTE: previous versions had a second
        #   `if mode == "all" and new_order == prev_order: break`
        # here, but because `prev_order` was just reassigned to
        # `new_order` on the line above, that comparison was trivially
        # true on every iteration — the loop aborted after one pass
        # and left long degree-2 chains uncollapsed.  Removed.

    return G


def prune_dangling_branches(G: nx.Graph, min_length: float = 0.0, verbose: bool = False) -> nx.Graph:
    """
    Remove dangling branches (degree-1 endpoints back to nearest junction).

    Iteratively removes degree-1 nodes and their edges. If min_length > 0,
    only removes branches whose total edge weight (length) is below the
    threshold. If min_length == 0, removes ALL dangling branches.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    min_length : float
        Minimum branch length to keep (pixels). 0 = remove all dangling.
    verbose : bool
        Print progress

    Returns
    -------
    nx.Graph
        Graph with dangling branches removed
    """
    G = G.copy()
    total_removed = 0

    while True:
        to_remove = []
        for node in list(G.nodes()):
            if G.degree(node) == 1:
                if min_length > 0:
                    # Walk the chain from the tip until we reach a
                    # junction (degree ≥ 3) or another tip (degree 1),
                    # summing edge weights as we go.
                    #
                    # NOTE: previous versions used `while G.degree(current) == 1`
                    # as the outer loop condition, but `current` becomes a
                    # degree-2 chain node after the first step, so the loop
                    # exited and only the first edge was counted.  That made
                    # the prune trivially eat every chain edge-by-edge from
                    # the tips inward.  Fixed by walking until we hit a
                    # junction explicitly.
                    branch_length = 0.0
                    current = node
                    visited = {current}
                    while True:
                        neighbors = [n for n in G.neighbors(current)
                                      if n not in visited]
                        if not neighbors:
                            break
                        nxt = neighbors[0]
                        edge_data = G.edges[current, nxt]
                        branch_length += edge_data.get(
                            'weight', edge_data.get('length', 1.0))
                        if G.degree(nxt) != 2:
                            break  # junction or another endpoint
                        visited.add(nxt)
                        current = nxt
                    if branch_length < min_length:
                        to_remove.append(node)
                else:
                    to_remove.append(node)

        if not to_remove:
            break

        # Remove one layer of degree-1 nodes at a time
        G.remove_nodes_from(to_remove)
        total_removed += len(to_remove)

    if verbose:
        print(f"  Pruned {total_removed} dangling nodes")

    return G


def keep_largest_component(G: nx.Graph) -> nx.Graph:
    """
    Keep only the largest connected component.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    nx.Graph
        Subgraph of largest component
    """
    if G.number_of_nodes() == 0:
        return G

    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()


def relabel_nodes_sequential(G: nx.Graph) -> nx.Graph:
    """
    Relabel nodes to sequential integers starting from 0.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    nx.Graph
        Graph with sequential node labels
    """
    mapping = {old: new for new, old in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping)


def ensure_edge_radius(G: nx.Graph) -> nx.Graph:
    """
    Ensure 'radius' attribute exists on all edges (default 1.0).

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    nx.Graph
        Graph with radius attribute on edges
    """
    for u, v, data in G.edges(data=True):
        if 'radius' not in data:
            data['radius'] = 1.0
    return G


def add_edge_length_from_weight(G: nx.Graph) -> nx.Graph:
    """
    Add 'length' attribute to edges (copy of weight).

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    nx.Graph
        Graph with length attribute on edges
    """
    for u, v, data in G.edges(data=True):
        data['length'] = data.get('weight', 1.0)
    return G


def _trace_vessel_chain(G: nx.Graph, start, nbr):
    """
    Trace a degree-2 chain from start through nbr until hitting another junction.

    Returns list of node IDs forming the path (including start and end junctions).
    """
    path = [start, nbr]
    visited = {start, nbr}
    curr = nbr

    while G.degree(curr) == 2:
        neighbors = [n for n in G.neighbors(curr) if n not in visited]
        if len(neighbors) != 1:
            break
        visited.add(curr)
        curr = neighbors[0]
        path.append(curr)

    return path


def collapse_to_vessel_graph(G: nx.Graph, verbose: bool = False) -> nx.Graph:
    """
    Collapse dense graph to vessel graph (one edge per vessel).

    Each edge in the output represents a complete vessel segment between
    junctions (nodes with degree != 2). The path geometry is preserved
    as an edge attribute for visualization and kymograph extraction.

    Parameters
    ----------
    G : nx.Graph
        Dense input graph with 'x', 'y', 'radius' node attributes
    verbose : bool
        Print progress info

    Returns
    -------
    nx.Graph
        Vessel graph where each edge has:
        - 'path': np.ndarray of shape (N, 2) with (x, y) coordinates
        - 'radii': np.ndarray of shape (N,) with radius at each path point
        - 'length': total arc length of the vessel
        - 'radius': length-weighted mean radius
    """
    vessel_G = nx.Graph()

    # Find junctions (degree != 2) and endpoints (degree == 1)
    junctions = [n for n, deg in G.degree() if deg != 2]

    if verbose:
        print(f"Found {len(junctions)} junctions")

    # Add junction nodes to new graph
    for j in junctions:
        data = G.nodes[j]
        vessel_G.add_node(j, x=float(data['x']), y=float(data['y']),
                         radius=float(data.get('radius', 1.0)))

    # Track which edges we've processed
    seen_edges = set()

    for start in junctions:
        for nbr in G.neighbors(start):
            # Skip if we've already traced this vessel from the other end
            if (start, nbr) in seen_edges or (nbr, start) in seen_edges:
                continue

            # Trace the vessel
            path_nodes = _trace_vessel_chain(G, start, nbr)
            end = path_nodes[-1]

            # Mark all edges in this path as seen
            for i in range(len(path_nodes) - 1):
                seen_edges.add((path_nodes[i], path_nodes[i+1]))
                seen_edges.add((path_nodes[i+1], path_nodes[i]))

            # Extract path coordinates and radii
            path_coords = np.array([
                [G.nodes[n]['x'], G.nodes[n]['y']]
                for n in path_nodes
            ], dtype=float)

            path_radii = np.array([
                G.nodes[n].get('radius', 1.0)
                for n in path_nodes
            ], dtype=float)

            # Smooth path to remove triangulation jaggedness
            # Pin endpoints so junctions stay connected
            if len(path_coords) > 5:
                from scipy.ndimage import gaussian_filter1d
                start_pt = path_coords[0].copy()
                end_pt = path_coords[-1].copy()
                path_coords = np.column_stack([
                    gaussian_filter1d(path_coords[:, 0], sigma=2.0, mode='nearest'),
                    gaussian_filter1d(path_coords[:, 1], sigma=2.0, mode='nearest'),
                ])
                path_coords[0] = start_pt
                path_coords[-1] = end_pt
                # Smooth radii too
                path_radii = gaussian_filter1d(path_radii, sigma=2.0, mode='nearest')

            # Compute arc length
            segments = np.linalg.norm(path_coords[1:] - path_coords[:-1], axis=1)
            total_length = float(np.sum(segments))

            # Length-weighted mean radius
            if total_length > 0 and len(segments) > 0:
                # Weight each radius by the length of adjacent segments
                weights = np.zeros(len(path_radii))
                weights[0] = segments[0] / 2
                weights[-1] = segments[-1] / 2
                for i in range(1, len(path_radii) - 1):
                    weights[i] = (segments[i-1] + segments[i]) / 2
                mean_radius = float(np.sum(path_radii * weights) / np.sum(weights))
            else:
                mean_radius = float(np.mean(path_radii))

            # Add edge with all vessel properties
            vessel_G.add_edge(
                start, end,
                path=path_coords,
                radii=path_radii,
                length=total_length,
                radius=mean_radius,
            )

    if verbose:
        print(f"Collapsed to {vessel_G.number_of_nodes()} nodes, "
              f"{vessel_G.number_of_edges()} vessels")

    return vessel_G
