"""
Triangulation for network vectorization.

Creates a constrained Delaunay triangulation of the vessel mask,
then classifies and prunes triangles to extract the network skeleton.
"""

import numpy as np
import meshpy.triangle as triangle
from typing import List, Tuple, Any

# Cython extension is required, not optional — `pip install -e .` builds
# it automatically; manual fallback is `python setup.py build_ext --inplace`.
# The try/except is kept so `import net3` succeeds even on a broken build
# (we can then produce a useful error at call time instead of crashing on
# import); public functions that need it raise RuntimeError explicitly.
try:
    from .C_net_functions import CbuildTriangles, CbruteforcePruning
    from .C_net_functions import CcreateTriangleAdjacencyMatrix
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    import warnings as _warnings
    _warnings.warn(
        "net3.C_net_functions Cython extension is not built. Run "
        "`pip install -e .` from the net3 repo root, or "
        "`python setup.py build_ext --inplace` for a manual build.",
        RuntimeWarning, stacklevel=2,
    )


def build_mesh(
    contours: List,
    longest_index: int
) -> Tuple[List, List, List]:
    """
    Build mesh points and facets from contours for triangulation.

    Parameters
    ----------
    contours : List
        List of contours (each is list of [x,y] points)
    longest_index : int
        Index of the longest (outer) contour

    Returns
    -------
    Tuple[List, List, List]
        (mesh_points, mesh_facets, hole_points)
    """
    from .contours import round_trip_connect, get_interior_point

    # Start with longest contour
    mesh_points = list(contours[longest_index])
    mesh_facets = round_trip_connect(0, len(mesh_points) - 1)

    # Add other contours as holes
    hole_points = []
    for i, contour in enumerate(contours):
        if i == longest_index:
            continue

        curr_length = len(mesh_points)
        interior = get_interior_point(contour)
        hole_points.append(interior)
        mesh_points.extend(contour)
        mesh_facets.extend(round_trip_connect(curr_length, len(mesh_points) - 1))

    return mesh_points, mesh_facets, hole_points


def triangulate_mesh(
    mesh_points: List,
    mesh_facets: List,
    hole_points: List
) -> Any:
    """
    Perform constrained Delaunay triangulation.

    Parameters
    ----------
    mesh_points : List
        List of (x, y) points
    mesh_facets : List
        List of (i, j) index pairs defining edges
    hole_points : List
        List of (x, y) points inside holes to exclude

    Returns
    -------
    meshpy.triangle.MeshInfo
        Triangulation result
    """
    info = triangle.MeshInfo()
    info.set_points(mesh_points)
    if hole_points:
        info.set_holes(hole_points)
    info.set_facets(mesh_facets)

    triangulation = triangle.build(
        info,
        verbose=False,
        allow_boundary_steiner=False,
        allow_volume_steiner=False,
        quality_meshing=False
    )

    return triangulation


def build_triangles(triangulation: Any) -> List:
    """
    Build triangle objects from triangulation.

    Parameters
    ----------
    triangulation : meshpy.triangle result
        Triangulation from meshpy

    Returns
    -------
    List
        List of triangle objects
    """
    if not CYTHON_AVAILABLE:
        raise RuntimeError("C_net_functions required. Build with setup.py")

    points = list(triangulation.points)
    for p in points:
        p[0] = np.round(p[0])
        p[1] = np.round(p[1])

    triangle_indices = list(triangulation.elements)
    return CbuildTriangles(points, triangle_indices)


def classify_triangles(triangles: List) -> Tuple[List, dict]:
    """
    Initialize and classify triangles by type.

    Types:
    - junction: 3 neighbors (branch point)
    - normal: 2 neighbors (along vessel)
    - end: 1 neighbor (vessel endpoint)
    - isolated: 0 neighbors (remove these)

    Parameters
    ----------
    triangles : List
        List of triangle objects

    Returns
    -------
    Tuple[List, dict]
        (triangles without isolated, type counts)
    """
    counts = {'junction': 0, 'normal': 0, 'end': 0, 'isolated': 0}
    isolated_indices = []

    for i, t in enumerate(triangles):
        t.init_triangle_mesh()
        ttype = t.get_type()
        counts[ttype] = counts.get(ttype, 0) + 1
        if ttype == 'isolated':
            isolated_indices.append(i)

    # Remove isolated triangles
    triangles = list(np.delete(np.asarray(triangles), isolated_indices))

    return triangles, counts


def prune_triangles(triangles: List, prune_order: int = 5, verbose: bool = False) -> List:
    """
    Prune short branches from triangle network.

    Removes 'prune_order' triangles from branch ends to reduce
    noise from jagged contours.

    Parameters
    ----------
    triangles : List
        List of triangle objects
    prune_order : int
        Number of triangles to prune from each end
    verbose : bool
        Print progress

    Returns
    -------
    List
        Pruned triangles
    """
    if not CYTHON_AVAILABLE:
        raise RuntimeError("C_net_functions required. Build with setup.py")

    return CbruteforcePruning(np.asarray(triangles), prune_order, verbose)


def set_triangle_centers(triangles: List, distance_map: np.ndarray) -> List:
    """
    Set center point and radius for each triangle from distance map.

    Parameters
    ----------
    triangles : List
        List of triangle objects
    distance_map : np.ndarray
        Distance transform of binary image

    Returns
    -------
    List
        Triangles with centers set
    """
    for t in triangles:
        t.init_triangle_mesh()
        t.set_center(distance_map)
    return triangles


def create_adjacency_matrix(triangles: List):
    """
    Create sparse adjacency matrix of triangle neighborhoods.

    Parameters
    ----------
    triangles : List
        List of triangle objects

    Returns
    -------
    scipy.sparse matrix
        Triangle adjacency matrix with distances as weights
    """
    if not CYTHON_AVAILABLE:
        raise RuntimeError("C_net_functions required. Build with setup.py")

    return CcreateTriangleAdjacencyMatrix(list(triangles))
