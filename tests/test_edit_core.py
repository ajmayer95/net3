"""Backend tests for GraphEditor — pure graph operations, no GUI."""

import networkx as nx
import numpy as np
import pytest

from net3.edit import GraphEditor


def _toy_graph():
    """5-node graph: 0─1─2 plus a triangle 2─3─4─2 hanging off node 2.

        0 — 1 — 2 — 3
                |   |
                +───4

    Has exactly one cycle (2─3─4─2).  Node 0 is a tip.  Node 2 is the
    only junction (degree 3).  Node 1 is degree-2 redundant.
    """
    G = nx.Graph()
    coords = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (2, 1)}
    for n, (x, y) in coords.items():
        G.add_node(n, x=x, y=y, radius=1.0)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 2)]:
        G.add_edge(u, v, weight=1.0, radius=1.0)
    return G


# ── Selection ─────────────────────────────────────────────────────


def test_select_and_toggle():
    ed = GraphEditor(_toy_graph())
    ed.select(3)
    assert ed.selected == {3}
    ed.toggle_selection(3)
    assert ed.selected == set()
    ed.toggle_selection(3)
    assert ed.selected == {3}
    ed.clear_selection()
    assert ed.selected == set()


def test_select_ignores_unknown_node():
    ed = GraphEditor(_toy_graph())
    ed.select(999)
    assert 999 not in ed.selected


def test_select_in_rect():
    ed = GraphEditor(_toy_graph())
    # Box covering nodes 2, 3, 4 (x in [1.5, 3.5], y in [-0.5, 1.5]).
    ed.select_in_rect(1.5, -0.5, 3.5, 1.5)
    assert ed.selected == {2, 3, 4}
    # Additive grows the selection.
    ed.select_in_rect(-0.5, -0.5, 1.5, 0.5, additive=True)
    assert ed.selected == {0, 1, 2, 3, 4}


# ── Hit testing ───────────────────────────────────────────────────


def test_node_at_finds_closest():
    ed = GraphEditor(_toy_graph())
    assert ed.node_at(0.1, 0.1, tolerance=0.5) == 0
    assert ed.node_at(2.9, 0.1, tolerance=0.5) == 3
    assert ed.node_at(10, 10, tolerance=0.5) is None


# ── Add / delete ──────────────────────────────────────────────────


def test_add_node_with_default_radius():
    ed = GraphEditor(_toy_graph())
    n = ed.add_node(5.0, 5.0)
    assert n == 5  # next int after max(0..4)
    assert ed.graph.nodes[n]["x"] == 5.0
    assert ed.graph.nodes[n]["radius"] == ed.DEFAULT_RADIUS


def test_add_node_samples_distance_map():
    # Distance map is in image-row convention; graph y is math-y
    # (y = H - row).  A dmap value at row 3 col 5 should be sampled
    # when the user passes graph coords (x=5, y=H-3=7).
    dmap = np.zeros((10, 10), dtype=np.float32)
    dmap[3, 5] = 7.5
    ed = GraphEditor(_toy_graph(), distance_map=dmap)
    n = ed.add_node(5.0, 7.0)
    assert ed.graph.nodes[n]["radius"] == pytest.approx(7.5)


def test_delete_selected_removes_node_and_edges():
    ed = GraphEditor(_toy_graph())
    ed.select(2)
    removed = ed.delete_selected()
    assert removed == 1
    assert 2 not in ed.graph
    # 2's incident edges should be gone — node 1, 3, 4 should lose them.
    assert not ed.graph.has_edge(1, 2)
    assert not ed.graph.has_edge(2, 3)
    assert not ed.graph.has_edge(2, 4)
    # Selection should clear after delete.
    assert ed.selected == set()


def test_delete_selected_noop_when_empty():
    ed = GraphEditor(_toy_graph())
    assert ed.delete_selected() == 0


# ── Add edge ──────────────────────────────────────────────────────


def test_add_edge_basic():
    ed = GraphEditor(_toy_graph())
    assert ed.add_edge(0, 3) is True
    assert ed.graph.has_edge(0, 3)
    assert ed.graph.edges[0, 3]["weight"] == pytest.approx(3.0)


def test_add_edge_refuses_self_loop_and_dupes():
    ed = GraphEditor(_toy_graph())
    assert ed.add_edge(0, 0) is False
    assert ed.add_edge(0, 1) is False  # already exists


def test_connect_selected_pair():
    ed = GraphEditor(_toy_graph())
    ed.select(0); ed.select(3)
    assert ed.connect_selected_pair() is True
    assert ed.graph.has_edge(0, 3)


def test_connect_selected_pair_needs_exactly_two():
    ed = GraphEditor(_toy_graph())
    assert ed.connect_selected_pair() is False
    ed.select(0)
    assert ed.connect_selected_pair() is False
    ed.select(1); ed.select(2)
    assert ed.connect_selected_pair() is False


# ── Cycles ────────────────────────────────────────────────────────


def test_find_cycles_returns_triangle():
    ed = GraphEditor(_toy_graph())
    cycles = ed.find_cycles()
    assert len(cycles) == 1
    assert set(cycles[0]) == {2, 3, 4}


def test_select_cycles_grows_selection():
    ed = GraphEditor(_toy_graph())
    n_added = ed.select_cycles()
    assert n_added == 3
    assert ed.selected == {2, 3, 4}


def test_find_cycles_acyclic_graph():
    G = nx.path_graph(5)
    for n in G.nodes:
        G.nodes[n]["x"] = n
        G.nodes[n]["y"] = 0
        G.nodes[n]["radius"] = 1.0
    for u, v in G.edges:
        G.edges[u, v]["weight"] = 1.0
        G.edges[u, v]["radius"] = 1.0
    ed = GraphEditor(G)
    assert ed.find_cycles() == []


# ── Streamline ────────────────────────────────────────────────────


def test_streamline_collapses_degree_2_chain():
    # 0—1—2—3—4 (linear chain), expect 0—4 after streamline.
    G = nx.path_graph(5)
    for n in G.nodes:
        G.nodes[n]["x"] = n
        G.nodes[n]["y"] = 0
        G.nodes[n]["radius"] = 1.0
    for u, v in G.edges:
        G.edges[u, v]["weight"] = 1.0
        G.edges[u, v]["radius"] = 1.0
    ed = GraphEditor(G)
    removed = ed.streamline()
    assert removed == 3
    assert set(ed.graph.nodes) == {0, 4}
    assert ed.graph.edges[0, 4]["weight"] == pytest.approx(4.0)


# ── Undo ──────────────────────────────────────────────────────────


def test_undo_reverts_delete():
    ed = GraphEditor(_toy_graph())
    n_before = ed.n_nodes
    ed.select(0)
    ed.delete_selected()
    assert ed.n_nodes == n_before - 1
    assert ed.undo() is True
    assert ed.n_nodes == n_before
    assert 0 in ed.graph


def test_undo_returns_false_at_root():
    ed = GraphEditor(_toy_graph())
    # No edits → only the initial snapshot is in the stack.
    assert ed.undo() is False


def test_undo_chain():
    ed = GraphEditor(_toy_graph())
    ed.add_node(7, 7)
    ed.add_node(8, 8)
    ed.add_node(9, 9)
    assert ed.n_nodes == 8
    ed.undo()  # remove (9,9)
    assert ed.n_nodes == 7
    ed.undo()  # remove (8,8)
    assert ed.n_nodes == 6
    ed.undo()  # remove (7,7)
    assert ed.n_nodes == 5


# ── Save round-trip ───────────────────────────────────────────────


def test_save_round_trip(tmp_path):
    ed = GraphEditor(_toy_graph())
    ed.add_node(7, 7)
    out = tmp_path / "edited.gpickle"
    ed.save(out)
    G2 = GraphEditor.from_gpickle(out).graph
    assert G2.number_of_nodes() == 6
    assert G2.number_of_edges() == 5


# ── Edge hit-testing ──────────────────────────────────────────────


def test_edge_at_finds_nearest_segment():
    ed = GraphEditor(_toy_graph())
    # Edge 0-1 runs from (0,0) to (1,0). Point near the segment middle
    # at (0.5, 0.1) is well inside tol=0.5.
    hit = ed.edge_at(0.5, 0.1, tolerance=0.5)
    assert hit == (0, 1)
    # Click far from any edge → None.
    assert ed.edge_at(10, 10, tolerance=0.5) is None


def test_edge_at_returns_canonical_order():
    ed = GraphEditor(_toy_graph())
    # Whatever edge is closest to (2.5, 0.1) — segment 2-3 — should be
    # returned in (min, max) order regardless of internal nx order.
    hit = ed.edge_at(2.5, 0.1, tolerance=0.5)
    assert hit == (2, 3)


# ── Edge selection ────────────────────────────────────────────────


def test_select_and_toggle_edge():
    ed = GraphEditor(_toy_graph())
    assert ed.select_edge(0, 1) is True
    assert (0, 1) in ed.selected_edges
    ed.toggle_edge_selection(1, 0)  # order-insensitive
    assert (0, 1) not in ed.selected_edges
    ed.toggle_edge_selection(0, 1)
    assert (0, 1) in ed.selected_edges


def test_select_edge_refuses_missing():
    ed = GraphEditor(_toy_graph())
    assert ed.select_edge(0, 99) is False
    assert ed.select_edge(0, 4) is False  # nodes exist but no edge


def test_clear_selection_clears_both_kinds():
    ed = GraphEditor(_toy_graph())
    ed.select(0); ed.select_edge(0, 1)
    ed.clear_selection()
    assert ed.selected == set()
    assert ed.selected_edges == set()


# ── Edge delete ───────────────────────────────────────────────────


def test_delete_selected_edges_removes_only_edges():
    ed = GraphEditor(_toy_graph())
    ed.select_edge(2, 3); ed.select_edge(3, 4)
    n = ed.delete_selected_edges()
    assert n == 2
    assert not ed.graph.has_edge(2, 3)
    assert not ed.graph.has_edge(3, 4)
    # Endpoints still in the graph.
    assert 2 in ed.graph and 3 in ed.graph and 4 in ed.graph
    # Selection cleared.
    assert ed.selected_edges == set()


def test_delete_selected_node_drops_stale_edge_selection():
    ed = GraphEditor(_toy_graph())
    ed.select_edge(0, 1); ed.select_edge(1, 2)
    ed.select(1)
    ed.delete_selected()
    # Edge selections referencing node 1 should be dropped.
    assert ed.selected_edges == set()


def test_delete_selected_edges_then_undo():
    ed = GraphEditor(_toy_graph())
    ed.select_edge(2, 3)
    ed.delete_selected_edges()
    assert not ed.graph.has_edge(2, 3)
    ed.undo()
    assert ed.graph.has_edge(2, 3)


# ── Snap-to-centerline ────────────────────────────────────────────


def test_add_node_snaps_to_distance_map_peak():
    # 11x11 distance map with the peak at image-row 3, col 5.
    # In graph coords: x=5, y = H - 3 = 8.
    dmap = np.zeros((11, 11), dtype=np.float32)
    dmap[3, 5] = 10.0
    ed = GraphEditor(_toy_graph(), distance_map=dmap)
    # Click 2 pixels off the peak — snap should pull it back.
    n = ed.add_node(7.0, 8.0, snap_window=3)
    assert ed.graph.nodes[n]["x"] == pytest.approx(5.0)
    assert ed.graph.nodes[n]["y"] == pytest.approx(8.0)
    assert ed.graph.nodes[n]["radius"] == pytest.approx(10.0)


def test_add_node_snap_window_zero_disabled():
    dmap = np.zeros((11, 11), dtype=np.float32)
    dmap[3, 5] = 10.0
    ed = GraphEditor(_toy_graph(), distance_map=dmap)
    n = ed.add_node(7.0, 8.0, snap_window=0)
    # Stays at click position.
    assert ed.graph.nodes[n]["x"] == pytest.approx(7.0)
    assert ed.graph.nodes[n]["y"] == pytest.approx(8.0)


def test_add_node_snap_no_distance_map_is_noop():
    ed = GraphEditor(_toy_graph())
    n = ed.add_node(7.0, 8.0, snap_window=10)
    assert ed.graph.nodes[n]["x"] == pytest.approx(7.0)
    assert ed.graph.nodes[n]["y"] == pytest.approx(8.0)
