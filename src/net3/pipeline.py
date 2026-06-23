"""
Main vectorization pipeline.

Orchestrates the full process: image → contours → triangulation → graph
"""

import numpy as np
import networkx as nx
from typing import Optional, Tuple
from pathlib import Path
import pickle

from .image import preprocess_for_vectorization
from .contours import (
    extract_and_process_contours,
    add_noise_to_contours
)
from .triangulate import (
    build_mesh,
    triangulate_mesh,
    build_triangles,
    classify_triangles,
    prune_triangles,
    set_triangle_centers,
    create_adjacency_matrix
)
from .graph import (
    create_graph,
    remove_redundant_nodes,
    prune_dangling_branches,
    keep_largest_component,
    relabel_nodes_sequential,
    ensure_edge_radius,
    add_edge_length_from_weight
)


def vectorize(
    image_path: str,
    min_feature_size: int = 3000,
    smoothing: Optional[int] = None,
    bridge_gaps: Optional[int] = None,
    invert: bool = False,
    prune_order: int = 5,
    remove_redundant: str = "all",
    prune_dangling: bool = False,
    prune_dangling_min_length: float = 0.0,
    verbose: bool = True,
    return_triangles: bool = False
):
    """
    Full vectorization pipeline: binary image → NetworkX graph.

    Parameters
    ----------
    image_path : str
        Path to binary mask image
    min_feature_size : int
        Remove features smaller than this (pixels)
    smoothing : int, optional
        Morphological smoothing kernel size
    bridge_gaps : int, optional
        Closing radius to bridge gaps between disconnected regions (e.g., 10).
        Useful when vessels are cut at tile boundaries.
    invert : bool
        If True, invert the mask (swap foreground/background).
        Use this when vessels are black on white background.
    prune_order : int
        Number of triangles to prune from branch ends
    remove_redundant : str
        "all", "half", or "none" for redundant node removal
    verbose : bool
        Print progress
    return_triangles : bool
        If True, return (graph, triangles) tuple instead of just graph

    Returns
    -------
    nx.Graph or Tuple[nx.Graph, List]
        Vectorized network graph, optionally with triangles list
    """
    if verbose:
        print(f"Vectorizing: {image_path}")

    # Step 1: Preprocess image
    if verbose:
        print("  [1/7] Preprocessing image...")
    image, distance_map = preprocess_for_vectorization(
        image_path,
        min_feature_size=min_feature_size,
        smoothing=smoothing,
        bridge_gaps=bridge_gaps,
        invert=invert
    )
    height, width = image.shape
    if verbose:
        print(f"        Image size: {width} x {height}")

    # Step 2: Extract contours
    if verbose:
        print("  [2/7] Extracting contours...")
    contours, longest_idx = extract_and_process_contours(image)
    if verbose:
        print(f"        Found {len(contours)} contours")

    # Add noise for triangulation stability
    contours = add_noise_to_contours(contours)

    # Step 3: Build mesh
    if verbose:
        print("  [3/7] Building mesh...")
    mesh_points, mesh_facets, hole_points = build_mesh(contours, longest_idx)
    if verbose:
        print(f"        {len(mesh_points)} points, {len(hole_points)} holes")

    # Step 4: Triangulate
    if verbose:
        print("  [4/7] Triangulating...")
    triangulation = triangulate_mesh(mesh_points, mesh_facets, hole_points)

    # Step 5: Build and classify triangles
    if verbose:
        print("  [5/7] Building triangles...")
    triangles = build_triangles(triangulation)
    triangles, counts = classify_triangles(triangles)
    if verbose:
        print(f"        {len(triangles)} triangles: {counts}")

    # Prune short branches
    if prune_order > 0:
        if verbose:
            print(f"        Pruning (order={prune_order})...")
        triangles = prune_triangles(triangles, prune_order, verbose=False)

    # Set centers from distance map
    triangles = set_triangle_centers(triangles, distance_map)

    # Step 6: Create adjacency matrix
    if verbose:
        print("  [6/7] Creating adjacency matrix...")
    adj_matrix = create_adjacency_matrix(triangles)

    # Step 7: Build graph
    if verbose:
        print("  [7/7] Building graph...")
    G = create_graph(adj_matrix, triangles, height)

    # Remove redundant nodes
    if remove_redundant != "none":
        if verbose:
            print(f"        Removing redundant nodes (mode={remove_redundant})...")
        G = remove_redundant_nodes(G, mode=remove_redundant, verbose=False)

    # Prune dangling branches (degree-1 endpoints)
    if prune_dangling:
        if verbose:
            print(f"        Pruning dangling branches (min_length={prune_dangling_min_length:.0f}px)...")
        G = prune_dangling_branches(G, min_length=prune_dangling_min_length, verbose=verbose)

    # Keep largest component
    G = keep_largest_component(G)

    # Relabel nodes sequentially
    G = relabel_nodes_sequential(G)

    # Add convenience attributes (ensure radius/length exist)
    G = ensure_edge_radius(G)
    G = add_edge_length_from_weight(G)

    if verbose:
        print(f"  Done! Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if return_triangles:
        return G, triangles
    return G


def save_graph(G: nx.Graph, path: str, format: str = "gpickle") -> None:
    """
    Save graph to file.

    Parameters
    ----------
    G : nx.Graph
        Graph to save
    path : str
        Output path
    format : str
        "gpickle" (default), "graphml", "gml", "edgelist"
    """
    path = Path(path)

    if format == "gpickle":
        with open(path, 'wb') as f:
            pickle.dump(G, f)
    elif format == "graphml":
        nx.write_graphml(G, path)
    elif format == "gml":
        nx.write_gml(G, path)
    elif format == "edgelist":
        nx.write_edgelist(G, path)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_graph(path: str) -> nx.Graph:
    """
    Load graph from gpickle file.

    Parameters
    ----------
    path : str
        Path to .gpickle file

    Returns
    -------
    nx.Graph
        Loaded graph
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
