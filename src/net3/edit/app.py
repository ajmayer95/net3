"""
napari-based frontend for net3's GraphEditor.

Requires the ``[gui]`` install extra (``pip install net3[gui]``).
The backend in :mod:`net3.edit.core` is pure-Python and has no
dependency on this module; the frontend just wires up napari layers,
mouse / key callbacks, and a small dock widget for buttons.

Coordinate convention
---------------------
Net3 graphs store ``x`` and ``y`` per node, with ``y`` flipped to
mathematical convention (``y = image_height - row``).  napari displays
images with ``(row, col)`` and origin at top.  So a graph node at
``(x, y)`` renders at napari position ``(image_height - y, x)``.
We do the flip on display only; the backend speaks graph coords
throughout.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .core import GraphEditor


# Visual constants — easy to tweak.
NODE_SIZE_DEFAULT = 6
NODE_SIZE_SELECTED = 10
NODE_COLOR_UNSELECTED = "red"
NODE_COLOR_SELECTED = "yellow"
NODE_COLOR_JUNCTION = "cyan"
NODE_COLOR_TIP = "lime"
NODE_COLOR_TRUE_TIP = "magenta"
NODE_SIZE_TRUE_TIP = 12
EDGE_COLOR = "orange"
EDGE_COLOR_SELECTED = "yellow"
EDGE_WIDTH = 1.2
EDGE_WIDTH_SELECTED = 2.4
RUBBER_BAND_COLOR = "yellow"
RUBBER_BAND_WIDTH = 0.8
# Default hit-test tolerance (pixels in graph coords) when zoom info
# isn't available.  Scales inversely with zoom in `_click_tolerance`.
NEW_NODE_TOLERANCE_FALLBACK = 8.0
# Snap-to-centerline search radius (graph pixels).  0 disables.
SNAP_WINDOW_DEFAULT = 6
# Pixels-of-canvas-movement above which a press/release is treated
# as a drag instead of a click.
DRAG_THRESHOLD_PX = 4.0


# ── Public entry point ────────────────────────────────────────────


def run_editor(
    graph_path: str | Path,
    mask_path: str | Path | None = None,
    distance_map_path: str | Path | None = None,
    save_path: str | Path | None = None,
    tile_positions_path: str | Path | None = None,
    video_dirs: list | None = None,
    video_pattern: str | None = None,
) -> GraphEditor:
    """Open the editor on the given graph + optional mask / distance
    map.  Blocks until the napari window is closed.  Returns the
    (possibly-edited) :class:`GraphEditor` instance.

    Save target: ``save_path`` if provided, otherwise overwrite
    ``graph_path`` on Ctrl-S / button click (with a confirmation
    dialog).

    Distance map fallback: if a mask is passed but no distance map is
    provided, the editor computes one in-memory from the mask via
    `cv2.distanceTransform` so newly-created nodes get a sensible
    radius from the local distance-transform peak instead of the
    default fallback.

    Optional video overlays: pass ``tile_positions_path`` +
    ``video_dirs`` + ``video_pattern`` to enable a "load tile video"
    dropdown in the dock.  Videos are loaded one at a time on demand
    (memory-conservative).  See :mod:`net3.edit.video_overlay`.
    """
    editor = GraphEditor.from_gpickle(
        graph_path,
        mask_path=mask_path,
        distance_map_path=distance_map_path,
    )
    if editor.distance_map is None and editor.mask is not None:
        editor.distance_map = _distance_map_from_mask(editor.mask)
        print("  computed distance map from mask "
              f"(max radius {float(editor.distance_map.max()):.1f} px)")
    _maybe_warn_sparse(editor)
    app = _EditorApp(editor,
                      default_save_path=Path(save_path or graph_path),
                      tile_positions_path=tile_positions_path,
                      video_dirs=video_dirs,
                      video_pattern=video_pattern)
    app.run()


def _distance_map_from_mask(mask):
    """Compute a distance transform of `mask` so newly-created nodes
    can sample radius from it.  Uses OpenCV's L2 transform (matches
    what `pertile/vectorize/image.py` produces, so radii read out are
    in the same units as the graph's existing radius attributes)."""
    import cv2
    import numpy as np
    bw = (mask > 127).astype(np.uint8) if mask.dtype != bool else mask.astype(np.uint8)
    return cv2.distanceTransform(bw, cv2.DIST_L2, 5).astype(np.float32)
    return editor


def _maybe_warn_sparse(editor: GraphEditor) -> None:
    """Print a hint when the loaded graph has so few degree-2 nodes
    that straight-line edges will cut across the mask instead of
    tracing its centerline.  This happens when a graph was vectorised
    with `remove_redundant='all'` — fine for topology, but visually
    misleading in an interactive editor."""
    if editor.n_nodes == 0:
        return
    deg2 = sum(1 for _, d in editor.graph.degree() if d == 2)
    deg2_frac = deg2 / editor.n_nodes
    # < 5% degree-2 nodes is the "topologically collapsed" regime.
    if deg2_frac < 0.05:
        import sys
        print(
            "\nnet3 edit: heads-up — this graph has only "
            f"{deg2}/{editor.n_nodes} degree-2 nodes "
            f"({deg2_frac:.1%}).  Straight-line edges between "
            "junctions and tips will cut across the mask instead of "
            "tracing it.  To get a graph that visibly follows the "
            "centerline, re-vectorise with:\n"
            "    net3 vectorize MASK -o GRAPH --for-editor\n",
            file=sys.stderr,
        )


# ── Frontend wiring ───────────────────────────────────────────────


class _EditorApp:
    """Owns the napari viewer + the layers + the input bindings.

    Layer rebuilds are full-tear-down/re-add for simplicity.  Sub-
    second on a 4000-node graph; if you need it snappier on huge
    graphs we can incrementalise later.
    """

    def __init__(
        self,
        editor: GraphEditor,
        default_save_path: Path,
        tile_positions_path: str | Path | None = None,
        video_dirs: list | None = None,
        video_pattern: str | None = None,
    ):
        import napari
        from qtpy.QtWidgets import (
            QPushButton, QLabel, QVBoxLayout, QWidget, QMessageBox,
        )
        self._napari = napari
        self._QMessageBox = QMessageBox
        self.editor = editor
        self.default_save_path = default_save_path
        # Optional video overlay — feature-flagged, initialized
        # lazily in _build_dock so failure to load doesn't kill the app.
        self._video_overlay = None
        self._video_config = None
        if (tile_positions_path is not None
                and video_dirs is not None
                and video_pattern is not None):
            self._video_config = dict(
                tile_positions_path=tile_positions_path,
                video_dirs=video_dirs,
                video_pattern=video_pattern,
            )
        self._image_h = (editor.mask.shape[0]
                          if editor.mask is not None
                          else self._infer_height_from_graph())
        self.viewer = napari.Viewer(title=f"net3 edit · {default_save_path.name}")
        # Base layers: built on graph mutation only; selection visuals
        # live in small overlay layers (selected_nodes / selected_edges)
        # so toggling selection is O(|selected|), independent of the
        # base graph size.
        self._nodes_layer = None
        self._edges_layer = None
        self._sel_nodes_overlay = None
        self._sel_edges_overlay = None
        self._rubber_band_layer = None  # (legacy, kept for fallback)
        # Tool layer for rectangle drag-select.  Goes into napari's
        # 'add_rectangle' mode when Rect-Select mode is active — that
        # automatically suppresses pan (which is the default left-drag
        # behaviour and can't be cancelled from a mouse_drag_callback).
        self._rect_tool_layer = None
        self._rect_tool_handler_blocked = False
        # Mode state
        self._mode: str = "select"   # "select" | "rect" | "add" | "move"
        self._move_target = None     # node currently armed for Move
        # Show degree-2 intermediate path nodes?  Off by default — at
        # 30k+ nodes, drawing every intermediate slows every edit by
        # an order of magnitude.  Toggle with `i` or the dock checkbox.
        self._show_intermediates: bool = False
        # Stable ordering for the points layer — backend node ids in
        # whatever order we built the points array.  Used to map napari
        # selection indices back to graph node ids.
        self._node_index_order: List = []
        # edge layer index → canonical (u,v) — kept for any future
        # incremental ops on the base layer.
        self._edge_index_order: List[Tuple] = []
        self._edge_pair_to_index: dict = {}
        # Cache node positions (graph coords) by id, populated on
        # node-layer build, used to draw the selection overlay quickly.
        self._node_xy_by_id: dict = {}

        self._build_layers()
        self._build_dock(QPushButton, QLabel, QVBoxLayout, QWidget)
        self._bind_inputs()
        self._refresh_status()

    # ── Build / refresh layers ──────────────────────────────────

    def _infer_height_from_graph(self) -> int:
        ys = [d.get("y", 0) for _, d in self.editor.graph.nodes(data=True)]
        return int(max(ys) + 10) if ys else 1000

    def _build_layers(self) -> None:
        if self.editor.mask is not None:
            self.viewer.add_image(
                self.editor.mask, name="mask",
                colormap="gray", opacity=1.0,
            )
        self._refresh_edges_layer()
        self._refresh_nodes_layer()
        self._refresh_selection_overlays()
        self.viewer.reset_view()

    def _node_positions_and_colors(self):
        """Base nodes layer — coloured only by degree, NOT by
        selection.  Selection visuals live in a small overlay layer
        rebuilt independently so selection toggles stay O(1).

        When `_show_intermediates` is False (default), degree-2 nodes
        are skipped from the drawn layer — they still live in the
        graph and `_node_xy_by_id` so selection overlays / hit-test
        remain correct, but tens of thousands of intermediate markers
        don't get re-uploaded to the GPU on every edit.
        """
        positions = []
        face_colors = []
        sizes = []
        ids: List = []
        xy_by_id: dict = {}
        G = self.editor.graph
        show_intermediates = getattr(self, "_show_intermediates", False)
        for n, d in G.nodes(data=True):
            x = d.get("x"); y = d.get("y")
            if x is None or y is None:
                continue
            xy_by_id[n] = (float(x), float(y))
            deg = G.degree(n)
            if deg == 2 and not show_intermediates:
                continue
            positions.append([self._image_h - float(y), float(x)])
            ids.append(n)
            is_true_tip = bool(d.get("true_tip"))
            if is_true_tip:
                face_colors.append(NODE_COLOR_TRUE_TIP)
                sizes.append(NODE_SIZE_TRUE_TIP)
            elif deg >= 3:
                face_colors.append(NODE_COLOR_JUNCTION)
                sizes.append(NODE_SIZE_DEFAULT)
            elif deg == 1:
                face_colors.append(NODE_COLOR_TIP)
                sizes.append(NODE_SIZE_DEFAULT)
            else:
                face_colors.append(NODE_COLOR_UNSELECTED)
                sizes.append(NODE_SIZE_DEFAULT)
        self._node_index_order = ids
        self._node_xy_by_id = xy_by_id
        return (np.array(positions) if positions
                else np.zeros((0, 2))), face_colors, sizes

    def _refresh_nodes_layer(self) -> None:
        """Update the nodes Points layer in place (created once).  Never
        remove/re-add it: that changes layer order, which is what hid the
        graph under a loaded tile video."""
        positions, colors, sizes = self._node_positions_and_colors()
        if self._nodes_layer is not None and self._nodes_layer in self.viewer.layers:
            L = self._nodes_layer
            L.selected_data = set()
            L.data = positions
            if len(positions):
                L.face_color = colors
                L.size = sizes
            return
        self._nodes_layer = self.viewer.add_points(
            positions,
            name="nodes",
            face_color=colors if len(positions) else "white",
            size=sizes if len(positions) else 5,
            symbol="o",
        )

    def _refresh_edges_layer(self) -> None:
        """Base edges layer — uniform orange, never modified for
        selection.  Selected edges are drawn yellow on top in a small
        overlay (see `_refresh_selection_overlays`).

        Each edge is drawn from its `path` polyline (straight segment
        when absent) as line segments in ONE napari Vectors layer.  Per-
        edge segment arrays are cached, so after an edit only changed
        edges are recomputed and the layer's data is swapped in place
        (fast on 100k+ edges; a Shapes layer with one shape per edge
        took seconds).  The layer is created once and never removed, so
        its position in the layer stack — below nodes, above any tile
        video — is preserved.
        """
        G = self.editor.graph
        canon = GraphEditor._canonical_edge
        H = self._image_h
        cache = getattr(self, "_edge_seg_cache", None)
        if cache is None:
            cache = self._edge_seg_cache = {}
        new_cache, chunks, edge_pairs = {}, [], []
        for u, v, d in G.edges(data=True):
            pair = canon(u, v)
            path = d.get("path")
            nu, nv = G.nodes[u], G.nodes[v]
            key = (id(path), nu.get("x"), nu.get("y"), nv.get("x"), nv.get("y"))
            hit = cache.get(pair)
            if hit is not None and hit[0] == key:
                segs = hit[1]
            else:
                pts = None
                if path is not None and len(path) >= 2:
                    pa = np.asarray(path, dtype=float)
                    if pa.ndim == 2 and pa.shape[1] >= 2:
                        pts = pa[:, :2]
                if pts is None:
                    pts = np.array([[float(nu.get("x", 0.0)), float(nu.get("y", 0.0))],
                                    [float(nv.get("x", 0.0)), float(nv.get("y", 0.0))]])
                # path stores (x, y_math); napari wants (row, col)
                rc = np.column_stack([H - pts[:, 1], pts[:, 0]])
                segs = np.stack([rc[:-1], rc[1:] - rc[:-1]], axis=1)
            new_cache[pair] = (key, segs)
            chunks.append(segs)
            edge_pairs.append(pair)
        self._edge_seg_cache = new_cache
        self._edge_index_order = edge_pairs
        self._edge_pair_to_index = {p: i for i, p in enumerate(edge_pairs)}
        data = np.concatenate(chunks) if chunks else np.zeros((0, 2, 2))
        if self._edges_layer is not None and self._edges_layer in self.viewer.layers:
            self._edges_layer.data = data
            return
        self._edges_layer = self.viewer.add_vectors(
            data,
            edge_color=EDGE_COLOR,
            edge_width=EDGE_WIDTH,
            opacity=0.85,
            name="edges",
        )
        # Move edges to just above the mask so node markers render on top.
        self.viewer.layers.move(
            self.viewer.layers.index(self._edges_layer),
            1 if len(self.viewer.layers) > 1 else 0,
        )

    def _refresh_all(self) -> None:
        """Full rebuild — only on graph mutation (delete, add, etc).
        Selection toggles use `_refresh_selection_overlays` instead."""
        self._refresh_edges_layer()
        self._refresh_nodes_layer()
        self._refresh_selection_overlays()
        self._refresh_status()

    def _refresh_selection_overlays(self) -> None:
        """Rebuild the small overlay layers that show selected nodes /
        selected edges.  Cost is O(|selected|), not O(|graph|), so
        selection toggles remain instant even at 20k edges."""
        # --- selected edges overlay ---
        sel_e = self.editor.selected_edges
        seg = []
        for (u, v) in sel_e:
            pu = self._node_xy_by_id.get(u)
            pv = self._node_xy_by_id.get(v)
            if pu is None or pv is None:
                continue
            seg.append([
                [self._image_h - pu[1], pu[0]],
                [self._image_h - pv[1], pv[0]],
            ])
        if self._sel_edges_overlay is not None:
            try:
                self.viewer.layers.remove(self._sel_edges_overlay)
            except (KeyError, ValueError):
                pass
            self._sel_edges_overlay = None
        if seg:
            self._sel_edges_overlay = self.viewer.add_shapes(
                seg,
                shape_type="line",
                edge_color=EDGE_COLOR_SELECTED,
                edge_width=EDGE_WIDTH_SELECTED,
                face_color="transparent",
                opacity=0.95,
                name="selected edges",
            )

        # --- selected nodes overlay ---
        sel_n = self.editor.selected
        positions = []
        for n in sel_n:
            pos = self._node_xy_by_id.get(n)
            if pos is None:
                continue
            positions.append([self._image_h - pos[1], pos[0]])
        if self._sel_nodes_overlay is not None:
            try:
                self.viewer.layers.remove(self._sel_nodes_overlay)
            except (KeyError, ValueError):
                pass
            self._sel_nodes_overlay = None
        if positions:
            self._sel_nodes_overlay = self.viewer.add_points(
                np.asarray(positions),
                face_color=NODE_COLOR_SELECTED,
                size=NODE_SIZE_SELECTED,
                symbol="o",
                name="selected nodes",
            )

    # ── Build the dock widget ───────────────────────────────────

    def _build_dock(self, QPushButton, QLabel, QVBoxLayout, QWidget) -> None:
        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet(
            "padding: 6px; font-family: monospace; font-size: 11px;")

        layout = QVBoxLayout()
        layout.addWidget(self._status_lbl)

        # Mode toggles — checkable buttons that switch what left-drag
        # / left-click does.  Only one mode is on at any time.
        mode_label = QLabel("<b>Mode</b>")
        mode_label.setStyleSheet("padding: 2px 6px;")
        layout.addWidget(mode_label)
        self._mode_btns: dict = {}

        def _mode_btn(key: str, label: str, tooltip: str):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setToolTip(tooltip)
            b.clicked.connect(lambda _=False, k=key: self._set_mode(k))
            layout.addWidget(b)
            self._mode_btns[key] = b
            return b

        _mode_btn("select", "Select  (key: 1)",
                   "Default mode.  Left click toggles the nearest "
                   "node or edge; left drag pans.")
        _mode_btn("rect", "Region-Select  (key: 2)",
                   "Draw a region with the shape chosen below; every "
                   "node inside (and every edge whose midpoint is inside) "
                   "is selected — then d deletes.  Hold Shift when "
                   "finishing the shape to add to the selection.")
        from qtpy.QtWidgets import QComboBox as _QCB, QHBoxLayout as _QHB, QWidget as _QW
        _row = _QW(); _rl = _QHB(_row); _rl.setContentsMargins(16, 0, 0, 0)
        _rl.addWidget(QLabel("shape:"))
        self._region_shape_combo = _QCB()
        for lab, m in (("Rectangle (drag)", "add_rectangle"),
                       ("Ellipse (drag)", "add_ellipse"),
                       ("Lasso (click, trace, click)", "add_polygon_lasso"),
                       ("Polygon (click vertices, dbl-click)", "add_polygon")):
            self._region_shape_combo.addItem(lab, userData=m)
        self._region_shape_combo.currentIndexChanged.connect(
            lambda _i: self._set_mode("rect") if self._mode == "rect" else None)
        _rl.addWidget(self._region_shape_combo, stretch=1)
        layout.addWidget(_row)
        _mode_btn("add", "Add Node  (key: 3)",
                   "Left click creates a new node (snapped to the "
                   "local distance-transform peak if a distance "
                   "map is loaded).")
        _mode_btn("move", "Move Node  (key: 4)",
                   "Left click a node to pick it up, then left click "
                   "the new position to place it.  Every incident edge "
                   "is replaced with a straight-line 2-point path.")
        # Default mode highlighted.
        self._mode_btns["select"].setChecked(True)

        action_label = QLabel("<b>Actions</b>")
        action_label.setStyleSheet("padding: 6px 6px 2px 6px;")
        layout.addWidget(action_label)

        def _btn(label: str, tooltip: str, callback):
            b = QPushButton(label)
            b.setToolTip(tooltip)
            b.clicked.connect(callback)
            layout.addWidget(b)
            return b

        _btn("Delete selected  (d)",
              "Remove every selected node (and its incident edges) "
              "AND every selected edge (keeping endpoints).",
              self._action_delete)
        _btn("Delete vessel  (Shift-D)",
              "Delete the whole vessel through each selected edge or "
              "mid-vessel node: every edge and degree-2 node between its "
              "two end nodes (junctions / tips), which are kept.",
              self._action_delete_vessel)
        from qtpy.QtWidgets import QSpinBox as _QSB, QHBoxLayout as _QHB2, QWidget as _QW2
        _crow = _QW2(); _cl = _QHB2(_crow); _cl.setContentsMargins(0, 0, 0, 0)
        _cb = QPushButton("Select disconnected pieces")
        _cb.setToolTip(
            "Select every connected component except the largest whose "
            "total centreline length is at most 'max px' (0 = all of "
            "them).  Review the yellow selection, then press d.")
        _cb.clicked.connect(self._action_select_components)
        _cl.addWidget(_cb, stretch=1)
        _cl.addWidget(QLabel("max px:"))
        self._comp_len_spin = _QSB(); self._comp_len_spin.setRange(0, 1000000)
        self._comp_len_spin.setSingleStep(100); self._comp_len_spin.setValue(500)
        _cl.addWidget(self._comp_len_spin)
        layout.addWidget(_crow)
        _btn("Connect 2 selected  (e)",
              "Add an edge between the exactly-two selected nodes.",
              self._action_connect)
        _btn("Select cycles  (m)",
              "Highlight every node that participates in a cycle.",
              self._action_cycles)
        _btn("Streamline  (n)",
              "Collapse every degree-2 redundant node by merging edges.",
              self._action_streamline)

        from qtpy.QtWidgets import QHBoxLayout, QDoubleSpinBox
        _mj_row = QHBoxLayout()
        _mj_row.addWidget(QLabel("Merge junctions ≤ (px):"))
        self._merge_junction_spin = QDoubleSpinBox()
        self._merge_junction_spin.setDecimals(1)
        self._merge_junction_spin.setMinimum(0.5)
        self._merge_junction_spin.setMaximum(100.0)
        self._merge_junction_spin.setSingleStep(0.5)
        self._merge_junction_spin.setValue(3.0)
        self._merge_junction_spin.setToolTip(
            "Distance threshold (px) for the 'Merge close junctions' "
            "op below.  Default 3 px handles typical vectorize "
            "triple-point artifacts (fat blob → N degree-3 nodes "
            "clustered at one triangle centre).")
        _mj_row.addWidget(self._merge_junction_spin)
        _mj_widget = QWidget(); _mj_widget.setLayout(_mj_row)
        layout.addWidget(_mj_widget)

        _btn("Merge close junctions",
              "Union-find every degree-≥3 node pair within the "
              "threshold above and contract each cluster to a single "
              "node at the centroid.  Radius = max of members.  Incident "
              "edges rewired; parallel edges deduplicated (max radius, "
              "min length).",
              self._action_merge_close_junctions)
        _btn("Clear selection  (c)",
              "Drop the current node selection.",
              self._action_clear)
        _btn("Undo  (z)",
              "Revert the most recent mutation.",
              self._action_undo)
        _btn("Save  (s)",
              f"Save graph to {self.default_save_path}.",
              self._action_save)

        # Tip-marking workflow: mark a few real anchors, prune the rest.
        _btn("Toggle true-tip flag  (t)",
              "Mark every selected node as a 'true tip' (magenta) so "
              "it survives Prune Unmarked Tips.  Toggles off if already "
              "marked.  Flag applies to any node, not just degree-1.",
              self._action_toggle_true_tip)
        _btn("Prune unmarked tips  (Shift-P)",
              "Delete every degree-1 node that does NOT carry the "
              "true-tip flag.  Cascades — after deleting the current "
              "unmarked tips, any node that just became degree-1 (and "
              "isn't marked) is also deleted, until convergence.  Peels "
              "off whole spur chains in one press.  Useful for cleaning "
              "up triangulation artifacts after marking the ~4 "
              "anatomical anchors by hand.",
              self._action_prune_unmarked_tips)

        # Trim-length spinbox for shorten-tip
        from qtpy.QtWidgets import QSpinBox, QHBoxLayout
        _shorten_row = QHBoxLayout()
        _shorten_row.addWidget(QLabel("Shorten trim (px):"))
        self._shorten_trim_spin = QSpinBox()
        self._shorten_trim_spin.setMinimum(1)
        self._shorten_trim_spin.setMaximum(500)
        self._shorten_trim_spin.setValue(15)
        self._shorten_trim_spin.setToolTip(
            "Arc-length (in mask-pixel units) to trim off the tip end.")
        _shorten_row.addWidget(self._shorten_trim_spin)
        _shorten_widget = QWidget()
        _shorten_widget.setLayout(_shorten_row)
        layout.addWidget(_shorten_widget)

        _btn("Shorten selected tips  (Shift-S)",
              "For each selected degree-1 node, walk inward along its "
              "incident edge by the trim distance (px) and place the "
              "tip at the new position.  Path truncates + edge length "
              "updates + tip radius resamples from the mask EDT.  "
              "Preserves anatomical presence without inheriting the "
              "vessel's tapered-end R.",
              self._action_shorten_selected_tips)

        # Gap bridging
        from qtpy.QtWidgets import (QDoubleSpinBox, QCheckBox)
        gap_label = QLabel("<b>Gap bridging</b>")
        gap_label.setStyleSheet("padding: 6px 6px 2px 6px;")
        layout.addWidget(gap_label)

        _gap_row = QHBoxLayout()
        _gap_row.addWidget(QLabel("D_max (px):"))
        self._gap_dmax_spin = QDoubleSpinBox()
        self._gap_dmax_spin.setDecimals(1)
        self._gap_dmax_spin.setMinimum(1.0)
        self._gap_dmax_spin.setMaximum(500.0)
        self._gap_dmax_spin.setSingleStep(1.0)
        self._gap_dmax_spin.setValue(15.0)
        self._gap_dmax_spin.setToolTip(
            "Maximum Euclidean distance (px) between leaf-node pairs to "
            "consider as gap-bridge candidates.")
        _gap_row.addWidget(self._gap_dmax_spin)
        _gap_widget = QWidget(); _gap_widget.setLayout(_gap_row)
        layout.addWidget(_gap_widget)

        self._gap_diff_cc_chk = QCheckBox("Different components only")
        self._gap_diff_cc_chk.setChecked(False)
        self._gap_diff_cc_chk.setToolTip(
            "When on, only pair leaves that live in different connected "
            "components of the graph.  Off by default — many editable "
            "graphs are one big component.")
        layout.addWidget(self._gap_diff_cc_chk)

        _topo_row = QHBoxLayout()
        _topo_row.addWidget(QLabel("Min topo hops:"))
        self._gap_topo_min_spin = QSpinBox()
        self._gap_topo_min_spin.setMinimum(0)
        self._gap_topo_min_spin.setMaximum(10000)
        self._gap_topo_min_spin.setValue(15)
        self._gap_topo_min_spin.setToolTip(
            "For same-component pairs, require the graph-hop distance "
            "between the two leaves to exceed this before treating them "
            "as a bridge candidate.  0 = no topology filter (any close "
            "leaf pair counts, even ones already connected via a 1-hop "
            "detour).")
        _topo_row.addWidget(self._gap_topo_min_spin)
        _topo_widget = QWidget(); _topo_widget.setLayout(_topo_row)
        layout.addWidget(_topo_widget)

        self._gap_leaf_to_any_chk = QCheckBox(
            "Also propose leaf → trunk (T-junctions)")
        self._gap_leaf_to_any_chk.setChecked(True)
        self._gap_leaf_to_any_chk.setToolTip(
            "Also consider pairs where a leaf is close to an interior "
            "node of another vessel (not a leaf).  Uses a stricter "
            "D_max (see below).  Off by default because it fires "
            "more aggressively.")
        layout.addWidget(self._gap_leaf_to_any_chk)

        _lta_row = QHBoxLayout()
        _lta_row.addWidget(QLabel("Leaf→trunk D_max (px):"))
        self._gap_dmax_lta_spin = QDoubleSpinBox()
        self._gap_dmax_lta_spin.setDecimals(1)
        self._gap_dmax_lta_spin.setMinimum(1.0)
        self._gap_dmax_lta_spin.setMaximum(500.0)
        self._gap_dmax_lta_spin.setSingleStep(1.0)
        self._gap_dmax_lta_spin.setValue(15.0)
        self._gap_dmax_lta_spin.setToolTip(
            "Maximum Euclidean distance for the leaf → interior-node "
            "(T-junction) case only.  Ignored when the checkbox above "
            "is off.  Default is ~0.6× the leaf-leaf D_max.")
        _lta_row.addWidget(self._gap_dmax_lta_spin)
        _lta_widget = QWidget(); _lta_widget.setLayout(_lta_row)
        layout.addWidget(_lta_widget)

        _btn("Find gap bridges",
              "Detect leaf pairs closer than D_max that are topologically "
              "far.  Renders candidates as yellow lines on a preview "
              "overlay.  Nothing is added to the graph until you press "
              "'Bridge all shown'.",
              self._action_find_gaps)
        _btn("Bridge all shown",
              "Add an edge for every candidate currently in the preview.  "
              "Single undo step covers the whole batch.",
              self._action_bridge_all_shown)
        _btn("Clear gap preview",
              "Remove the preview overlay without touching the graph.",
              self._action_clear_gap_preview)

        # State for the preview overlay
        self._gap_preview_layer = None
        self._gap_candidates: list = []

        # View toggles
        view_label = QLabel("<b>View</b>")
        view_label.setStyleSheet("padding: 6px 6px 2px 6px;")
        layout.addWidget(view_label)

        self._intermediates_btn = QPushButton("Show intermediates  (i)")
        self._intermediates_btn.setCheckable(True)
        self._intermediates_btn.setChecked(self._show_intermediates)
        self._intermediates_btn.setToolTip(
            "Toggle visibility of degree-2 path nodes.  Off by "
            "default for graphs with many intermediates so editing "
            "stays responsive; selection / hit-testing still works "
            "for hidden nodes."
        )
        self._intermediates_btn.clicked.connect(
            self._toggle_intermediates)
        layout.addWidget(self._intermediates_btn)

        # ── Optional per-tile video overlay ──
        if self._video_config is not None:
            self._build_video_overlay_ui(layout, QPushButton, QLabel)

        layout.addStretch(1)

        widget = QWidget()
        widget.setLayout(layout)
        # Wrap in a scroll area so the panel remains usable when it grows
        # taller than the napari window (many controls stacked vertically).
        from qtpy.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        self.viewer.window.add_dock_widget(
            scroll, name="net3 edit", area="right",
        )

    def _build_video_overlay_ui(self, layout, QPushButton, QLabel):
        """Add tile-video overlay controls to the dock.  Only called
        when the caller passed tile-positions + video-dir + pattern
        to run_editor."""
        from qtpy.QtWidgets import QComboBox, QHBoxLayout, QWidget
        from .video_overlay import (
            VideoOverlayManager, load_tile_positions)
        try:
            positions = load_tile_positions(
                self._video_config['tile_positions_path'])
        except Exception as e:
            print(f"[net3 edit] failed to load tile positions "
                  f"({self._video_config['tile_positions_path']}): {e}")
            return
        if not positions:
            print("[net3 edit] tile-positions JSON contained no usable entries")
            return
        mask_shape = None
        try:
            if self.editor.mask is not None:
                mask_shape = tuple(int(x) for x in self.editor.mask.shape[:2])
        except Exception:
            mask_shape = None
        self._video_overlay = VideoOverlayManager(
            self.viewer, positions,
            self._video_config['video_dirs'],
            self._video_config['video_pattern'],
            mask_shape=mask_shape)

        layout.addWidget(QLabel("<b>Tile video overlay</b>"))
        row = QWidget()
        rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("Tile:"))
        self._video_tile_combo = QComboBox()
        for tid in self._video_overlay.available_tile_ids():
            self._video_tile_combo.addItem(str(tid), userData=tid)
        rl.addWidget(self._video_tile_combo, stretch=1)
        layout.addWidget(row)

        load_btn = QPushButton("Load tile video")
        load_btn.setToolTip(
            "Load the selected tile's video as a napari Image layer at "
            "its mosaic position.  Use the built-in time slider to scrub "
            "frames.  Loading a different tile removes the previous one.")
        load_btn.clicked.connect(self._load_selected_tile_video)
        layout.addWidget(load_btn)

        unload_btn = QPushButton("Unload current video")
        unload_btn.clicked.connect(self._unload_tile_video)
        layout.addWidget(unload_btn)

        # Tile boundary + label toggles (helpful for verifying video
        # placement against the mosaic).  Mirrors the plexusflow viewer's
        # "Show tile boundaries" / "Show tile numbers" checkboxes.
        from qtpy.QtWidgets import QCheckBox
        self._tile_bounds_chk = QCheckBox("Show tile boundaries")
        self._tile_bounds_chk.setToolTip(
            "Draw a yellow outline at every tile's mosaic footprint "
            "(computed from the tile-positions affine).")
        self._tile_bounds_chk.toggled.connect(
            lambda c: self._video_overlay.set_show_tile_boundaries(bool(c)))
        layout.addWidget(self._tile_bounds_chk)
        self._tile_labels_chk = QCheckBox("Show tile numbers")
        self._tile_labels_chk.setToolTip(
            "Overlay the tile id at each tile's centre.")
        self._tile_labels_chk.toggled.connect(
            lambda c: self._video_overlay.set_show_tile_numbers(bool(c)))
        layout.addWidget(self._tile_labels_chk)

        print(f"[net3 edit] video overlay ready: "
              f"{len(positions)} tiles in positions JSON, "
              f"pattern={self._video_config['video_pattern']!r}, "
              f"dirs={[str(d) for d in self._video_config['video_dirs']]}")

    def _load_selected_tile_video(self):
        if self._video_overlay is None:
            return
        tid = self._video_tile_combo.currentData()
        if tid is None:
            return
        self._video_overlay.load_tile(int(tid))

    def _unload_tile_video(self):
        if self._video_overlay is None:
            return
        self._video_overlay.unload()

    # ── Key bindings + mouse ────────────────────────────────────

    def _bind_inputs(self) -> None:
        v = self.viewer

        @v.bind_key("d", overwrite=True)
        def _(viewer):
            self._action_delete()

        @v.bind_key("Shift-D", overwrite=True)
        def _(viewer):
            self._action_delete_vessel()

        @v.bind_key("e", overwrite=True)
        def _(viewer):
            self._action_connect()

        @v.bind_key("m", overwrite=True)
        def _(viewer):
            self._action_cycles()

        @v.bind_key("n", overwrite=True)
        def _(viewer):
            self._action_streamline()

        @v.bind_key("c", overwrite=True)
        def _(viewer):
            self._action_clear()

        @v.bind_key("z", overwrite=True)
        def _(viewer):
            self._action_undo()

        @v.bind_key("s", overwrite=True)
        def _(viewer):
            self._action_save()

        @v.bind_key("i", overwrite=True)
        def _(viewer):
            self._toggle_intermediates()

        @v.bind_key("t", overwrite=True)
        def _(viewer):
            self._action_toggle_true_tip()

        @v.bind_key("Shift-P", overwrite=True)
        def _(viewer):
            self._action_prune_unmarked_tips()

        @v.bind_key("Shift-S", overwrite=True)
        def _(viewer):
            self._action_shorten_selected_tips()

        # Mode shortcuts: 1 / 2 / 3 to switch.
        @v.bind_key("1", overwrite=True)
        def _(viewer):
            self._set_mode("select")

        @v.bind_key("2", overwrite=True)
        def _(viewer):
            self._set_mode("rect")

        @v.bind_key("3", overwrite=True)
        def _(viewer):
            self._set_mode("add")

        @v.bind_key("4", overwrite=True)
        def _(viewer):
            self._set_mode("move")

        # Mouse model:
        #   left click            : toggle nearest node, fall back to nearest edge
        #   Shift + left click    : create new node at click (snap-to-centerline)
        #   right click + drag    : rectangle select  (replaces current selection)
        #   right click + Shift   : rectangle select, additive
        #   left drag             : pan (napari default — we don't intercept)
        #
        # Why right-click for rect-select: napari's left-drag is bound
        # to pan and intercepts any modifier we'd try to use to
        # distinguish (Control/Alt/Meta).  Right-click is unbound by
        # default, so our generator gets the events cleanly.
        @v.mouse_drag_callbacks.append
        def _on_press(viewer, event):
            yield from self._mouse_handler(event)

    def _click_tolerance(self) -> float:
        """Click tolerance scaled to the current camera zoom so the
        same on-screen radius works at different zoom levels."""
        try:
            zoom = float(self.viewer.camera.zoom)
            if zoom > 0:
                return max(2.0, NEW_NODE_TOLERANCE_FALLBACK / zoom)
        except Exception:
            pass
        return NEW_NODE_TOLERANCE_FALLBACK

    # ── Mouse press/drag generator ──────────────────────────────

    def _event_graph_xy(self, event) -> Optional[tuple]:
        """Pull (graph_x, graph_y) out of a napari mouse event in
        canvas coords.  Returns None if the event has no position."""
        pos = event.position
        if pos is None or len(pos) < 2:
            return None
        row, col = float(pos[-2]), float(pos[-1])
        return col, self._image_h - row

    def _mouse_handler(self, event):
        """Mode-aware mouse handler for Select / Add-Node modes.

        Rect-Select is NOT handled here — it uses napari's built-in
        Shapes 'add_rectangle' mode, which intercepts the drag at the
        layer level (so pan is suppressed automatically).  See
        `_set_mode` and `_on_rect_drawn`.
        """
        if event.button != 1:
            return
        if self._mode == "rect":
            # Hand the drag to napari's add_rectangle.  Returning
            # without yielding lets napari's default layer handlers
            # see the event.
            return
        start = self._event_graph_xy(event)
        if start is None:
            return
        shift = "Shift" in event.modifiers
        last = start
        moved = False
        yield
        while event.type == "mouse_move":
            now = self._event_graph_xy(event)
            if now is not None:
                dx = now[0] - start[0]
                dy = now[1] - start[1]
                if (dx * dx + dy * dy) > (DRAG_THRESHOLD_PX * DRAG_THRESHOLD_PX):
                    moved = True
                last = now
            yield
        end = last
        if not moved:
            self._handle_click(end[0], end[1], shift=shift)

    def _handle_click(self, gx: float, gy: float, shift: bool) -> None:
        tol = self._click_tolerance()
        # Move mode: first click on a node arms it, second click on empty
        # canvas places it at the new position.  If a node is already
        # armed (self._move_target set), the click places it and clears
        # the arm.
        if self._mode == "move":
            armed = getattr(self, "_move_target", None)
            if armed is None:
                # Pick up: click on a node to select as move target.
                node_hit = self.editor.node_at(gx, gy, tolerance=tol)
                if node_hit is None:
                    self._announce(
                        "Move mode: click a node to pick it up, then click "
                        "the new position.")
                    return
                self._move_target = node_hit
                self._announce(
                    f"picked up node {node_hit} — click to place at new "
                    f"position (or press 1 to cancel).")
                self._refresh_status()
                return
            # Place at (gx, gy).  Uses snap-to-centerline if the distance
            # map is loaded (same behaviour as Add Node).
            ok = self.editor.move_node(
                armed, gx, gy, snap_window=SNAP_WINDOW_DEFAULT)
            if ok:
                nx = self.editor.graph.nodes[armed].get('x', gx)
                ny = self.editor.graph.nodes[armed].get('y', gy)
                self._announce(
                    f"moved node {armed} to ({nx:.1f}, {ny:.1f}); incident "
                    f"edge paths replaced with straight lines.")
            else:
                self._announce(
                    f"move failed — node {armed} no longer exists.")
            self._move_target = None
            # A move touches node coords AND edge paths, so refresh
            # everything.
            self._refresh_edges_layer()
            self._refresh_nodes_layer()
            self._refresh_selection_overlays()
            self._refresh_status()
            # Auto-revert to Select so the next click doesn't accidentally
            # start another move.  Press `4` to move again.
            self._set_mode("select")
            return
        # Add-Node mode (or Shift+click in any mode) creates a new node.
        if self._mode == "add" or shift:
            sticky = shift  # Shift+click is one-shot; Add Mode-1-shot too
            n = self.editor.add_node(gx, gy, snap_window=SNAP_WINDOW_DEFAULT)
            r = self.editor.graph.nodes[n].get("radius", 0.0)
            self._announce(
                f"created node {n} at ({gx:.1f}, {gy:.1f})  r={r:.1f}")
            # Add-node only mutates nodes -- no edges change -- so skip
            # the (very expensive) edges-layer rebuild that _refresh_all
            # would otherwise trigger.  At 39k edges this was the
            # ~5 s per-edit bottleneck on the 15-somite mosaic.
            self._refresh_nodes_layer()
            self._refresh_selection_overlays()
            # Auto-revert to Select after one add so the next click
            # doesn't accidentally drop another node.  Press `3` (or
            # the Add Node button) to add again.
            if not sticky:
                self._set_mode("select")
            else:
                self._refresh_status()
            return
        # Plain click → try node first (more specific), fall back to edge.
        node_hit = self.editor.node_at(gx, gy, tolerance=tol)
        if node_hit is not None:
            self.editor.toggle_selection(node_hit)
            in_ = node_hit in self.editor.selected
            self._announce(
                f"node {node_hit} {'selected' if in_ else 'deselected'}")
            self._refresh_selection_overlays()
            self._refresh_status()
            return
        edge_hit = self.editor.edge_at(gx, gy, tolerance=tol)
        if edge_hit is not None:
            u, v = edge_hit
            self.editor.toggle_edge_selection(u, v)
            sel = (u, v) in self.editor.selected_edges
            self._announce(
                f"edge ({u},{v}) {'selected' if sel else 'deselected'}")
            # Overlay rebuild is O(|selected|), not O(|graph|), so this
            # stays instant at 20k edges.
            self._refresh_selection_overlays()
            self._refresh_status()
            return
        self._announce(
            "(click missed — Shift-click to create a node)")
        self._refresh_status()

    # ── Rubber-band rectangle (live during Ctrl+drag) ──────────

    def _show_rubber_band(self, start, end) -> None:
        """Update / create a transient Shapes layer with a single
        rectangle covering the drag region.  Uses napari (row, col)
        coords; we convert from graph (x, y) here."""
        r0 = self._image_h - start[1]
        r1 = self._image_h - end[1]
        c0, c1 = start[0], end[0]
        rect = np.array([
            [min(r0, r1), min(c0, c1)],
            [min(r0, r1), max(c0, c1)],
            [max(r0, r1), max(c0, c1)],
            [max(r0, r1), min(c0, c1)],
        ])
        if self._rubber_band_layer is None:
            self._rubber_band_layer = self.viewer.add_shapes(
                [rect],
                shape_type="rectangle",
                edge_color=RUBBER_BAND_COLOR,
                edge_width=RUBBER_BAND_WIDTH,
                face_color="transparent",
                opacity=0.7,
                name="rubber band",
            )
        else:
            try:
                self._rubber_band_layer.data = [rect]
            except Exception:
                # Fallback: rebuild.
                self._hide_rubber_band()
                self._show_rubber_band(start, end)

    def _hide_rubber_band(self) -> None:
        if self._rubber_band_layer is not None:
            try:
                self.viewer.layers.remove(self._rubber_band_layer)
            except (KeyError, ValueError):
                pass
            self._rubber_band_layer = None

    # ── Actions ─────────────────────────────────────────────────

    def _announce(self, msg: str) -> None:
        self._last_action = msg

    def _set_mode(self, mode: str) -> None:
        """Switch input mode.  Updates which dock button is highlighted
        and which mouse behavior the handler uses.

        Rect-Select uses napari's built-in Shapes 'add_rectangle' mode
        on a dedicated tool layer — when that layer is active and in
        add_rectangle mode, napari swallows the left-drag (no pan) and
        draws a live rectangle preview itself.
        """
        if mode not in ("select", "rect", "add", "move"):
            return
        # Cancel any pending move-arm when leaving move mode.
        if mode != "move" and getattr(self, "_move_target", None) is not None:
            self._move_target = None
        self._mode = mode
        for k, b in self._mode_btns.items():
            try:
                b.setChecked(k == mode)
            except Exception:
                pass
        self._ensure_rect_tool_layer()
        if mode == "rect" and self._rect_tool_layer is not None:
            try:
                combo = getattr(self, "_region_shape_combo", None)
                self._rect_tool_layer.mode = (combo.currentData() if combo is not None
                                              else "add_rectangle")
                self.viewer.layers.selection.active = self._rect_tool_layer
            except Exception as e:
                print(f"  rect mode setup failed: {e!r}")
        else:
            # Restore the rect-tool layer to pan_zoom so pan works
            # normally.  Make a non-shapes layer active so subsequent
            # clicks aren't intercepted by the (now-inactive) tool.
            if self._rect_tool_layer is not None:
                try:
                    self._rect_tool_layer.mode = "pan_zoom"
                except Exception:
                    pass
            # Prefer the nodes layer as active in Select / Add modes.
            try:
                if self._nodes_layer is not None:
                    self.viewer.layers.selection.active = self._nodes_layer
            except Exception:
                pass
        label = {"select": "Select",
                  "rect": "Region-Select",
                  "add": "Add Node",
                  "move": "Move Node"}[mode]
        self._announce(f"mode → {label}")
        self._refresh_status()

    def _ensure_rect_tool_layer(self) -> None:
        """Lazy-create the rectangle-drawing tool layer + register a
        layer-level mouse_drag_callback that fires AFTER napari's
        built-in add_rectangle generator finishes the drag.

        Earlier versions listened for the layer's `events.set_data`
        signal, but that fires DURING napari's add_rectangle generator
        (between `layer.add(...)` and `layer.selected_data = {-1}`).
        Clearing the layer from inside that signal left
        napari trying to index `[-1]` of an empty list → IndexError.

        Using a mouse_drag_callback on the layer itself lets us
        process the just-finished rectangle from a clean state, and
        clear it for the next drag without re-entering napari's
        active generator.
        """
        if self._rect_tool_layer is not None:
            return
        try:
            self._rect_tool_layer = self.viewer.add_shapes(
                name="rect tool",
                edge_color=RUBBER_BAND_COLOR,
                edge_width=2,
                face_color="transparent",
                opacity=0.5,
            )

            # Process a finished shape (any type) once napari has fully
            # committed it: defer out of the data-event so clearing the
            # layer can't re-enter napari's drawing generator.
            from qtpy.QtCore import QTimer

            def _on_data(event):
                if str(getattr(event, "action", "")) == "added":
                    QTimer.singleShot(0, self._on_region_drawn)

            self._rect_tool_layer.events.data.connect(_on_data)

        except Exception as e:
            print(f"  rect tool layer init failed: {e!r}")
            self._rect_tool_layer = None

    def _on_region_drawn(self) -> None:
        """Select everything inside the shape just drawn on the region
        tool layer (rectangle, ellipse, lasso or polygon), then clear it."""
        layer = self._rect_tool_layer
        if layer is None or self._mode != "rect" or not len(layer.data):
            return
        verts = np.asarray(layer.data[-1], float)[:, -2:]   # (row, col)
        stype = layer.shape_type[-1]
        if stype == "ellipse" and len(verts) == 4:
            c = verts.mean(axis=0)
            a = 0.5 * (verts[1] + verts[2]) - c
            b = 0.5 * (verts[2] + verts[3]) - c
            t = np.linspace(0, 2 * np.pi, 96, endpoint=False)[:, None]
            verts = c + np.cos(t) * a + np.sin(t) * b
        poly_xy = np.column_stack([verts[:, 1], self._image_h - verts[:, 0]])
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtWidgets import QApplication
            shift = bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)
        except Exception:
            shift = False
        if len(poly_xy) >= 3:
            nn, ne = self.editor.select_in_polygon(poly_xy, additive=shift)
            self._announce(
                f"region-selected {nn} node(s), {ne} edge(s)"
                + ("  (additive)" if shift else "")
                + f" — total {len(self.editor.selected)} nodes, "
                  f"{len(self.editor.selected_edges)} edges; d deletes")
            self._refresh_selection_overlays()
        try:
            mode = layer.mode
            layer.data = []
            layer.mode = mode
        except Exception:
            pass
        self._refresh_status()

    def _action_select_components(self) -> None:
        mx = float(self._comp_len_spin.value())
        n_c, n_n = self.editor.select_small_components(mx)
        total = len(self.editor.component_sizes())
        self._announce(
            f"selected {n_c} disconnected piece(s) ({n_n} nodes) of "
            f"{total} components"
            + (f", each ≤ {mx:.0f} px" if mx > 0 else ", all but the largest")
            + " — press d to delete" if n_c else
            "no disconnected pieces match")
        self._refresh_selection_overlays()
        self._refresh_status()

    def _action_delete(self) -> None:
        # Delete BOTH selected nodes (with their incident edges) and
        # selected edges (keeping endpoint nodes).  Whichever the user
        # has staged, `d` removes it.
        n_nodes = self.editor.delete_selected()
        n_edges = self.editor.delete_selected_edges()
        parts = []
        if n_nodes:
            parts.append(f"{n_nodes} node(s)")
        if n_edges:
            parts.append(f"{n_edges} edge(s) only")
        msg = ("deleted " + " + ".join(parts)) if parts else "nothing selected"
        self._announce(msg)
        self._refresh_all()

    def _action_connect(self) -> None:
        ok = self.editor.connect_selected_pair()
        if ok:
            self._announce("added edge between the 2 selected nodes")
            self._refresh_all()
        else:
            self._announce("connect needs exactly 2 selected nodes "
                            "(and edge must not already exist)")
        self._refresh_status()

    def _action_delete_vessel(self) -> None:
        n_ves, n_nodes = self.editor.delete_selected_vessels()
        if not n_ves:
            self._announce("delete vessel: select an edge or a mid-vessel "
                           "(degree-2) node first")
            self._refresh_status()
            return
        self._announce(f"deleted {n_ves} vessel(s) ({n_nodes} interior "
                       f"node(s)); end nodes kept — z to undo")
        self._refresh_all()

    def _action_find_gaps(self) -> None:
        d_max = float(self._gap_dmax_spin.value())
        diff_cc = bool(self._gap_diff_cc_chk.isChecked())
        topo_min_raw = int(self._gap_topo_min_spin.value())
        topo_min = topo_min_raw if topo_min_raw > 0 else None
        allow_lta = bool(self._gap_leaf_to_any_chk.isChecked())
        d_lta = float(self._gap_dmax_lta_spin.value())
        cands = self.editor.find_gap_candidates(
            d_max=d_max,
            require_different_components=diff_cc,
            topo_min=topo_min,
            allow_leaf_to_any=allow_lta,
            d_max_leaf_to_any=d_lta)
        self._gap_candidates = cands
        self._render_gap_preview(cands)
        topo_tag = f", topo>{topo_min}" if topo_min else ""
        cc_tag = "diff-CC" if diff_cc else "any-CC"
        lta_tag = f", +leaf→trunk D≤{d_lta:g}" if allow_lta else ""
        self._announce(f"found {len(cands)} gap-bridge candidate(s) "
                        f"(D_max={d_max:g}, {cc_tag}{topo_tag}{lta_tag})")

    def _render_gap_preview(self, candidates) -> None:
        # Rasterize candidate bridges as an RGBA image overlay.
        from skimage.draw import line_aa
        H = self._image_h
        # Prefer mask width; else fall back to max x in graph.
        if self.editor.mask is not None:
            W = int(self.editor.mask.shape[1])
        else:
            xs = [float(self.editor.graph.nodes[n].get("x", 0.0))
                  for n in self.editor.graph.nodes]
            W = int(max(xs) + 1) if xs else H
        canvas = np.zeros((H, W, 4), dtype=np.uint8)
        # yellow fill, opaque
        for (u, v, _d) in candidates:
            if u not in self.editor.graph or v not in self.editor.graph:
                continue
            xu = float(self.editor.graph.nodes[u].get("x", 0.0))
            yu = float(self.editor.graph.nodes[u].get("y", 0.0))
            xv = float(self.editor.graph.nodes[v].get("x", 0.0))
            yv = float(self.editor.graph.nodes[v].get("y", 0.0))
            r0, c0 = int(round(H - yu)), int(round(xu))
            r1, c1 = int(round(H - yv)), int(round(xv))
            r0 = max(0, min(H - 1, r0)); r1 = max(0, min(H - 1, r1))
            c0 = max(0, min(W - 1, c0)); c1 = max(0, min(W - 1, c1))
            rr, cc, val = line_aa(r0, c0, r1, c1)
            a = (val * 255).astype(np.uint8)
            canvas[rr, cc, 0] = 255           # R
            canvas[rr, cc, 1] = 220           # G  → yellow
            canvas[rr, cc, 2] = 0             # B
            canvas[rr, cc, 3] = np.maximum(canvas[rr, cc, 3], a)
        # Widen lines a bit by dilating alpha with a 3x3 max filter.
        from scipy.ndimage import maximum_filter
        canvas[..., 3] = maximum_filter(canvas[..., 3], size=3)

        if self._gap_preview_layer is not None:
            try:
                self.viewer.layers.remove(self._gap_preview_layer)
            except (KeyError, ValueError):
                pass
            self._gap_preview_layer = None
        if not candidates:
            return
        self._gap_preview_layer = self.viewer.add_image(
            canvas, rgb=True, name="gap-bridge preview", opacity=1.0)

    def _action_bridge_all_shown(self) -> None:
        if not self._gap_candidates:
            self._announce("no gap candidates to bridge — press "
                            "'Find gap bridges' first")
            return
        n = self.editor.bridge_gaps(self._gap_candidates)
        self._gap_candidates = []
        # Clear the preview and rebuild everything now that edges landed.
        if self._gap_preview_layer is not None:
            try:
                self.viewer.layers.remove(self._gap_preview_layer)
            except (KeyError, ValueError):
                pass
            self._gap_preview_layer = None
        self._refresh_all()
        self._announce(f"bridged {n} gap(s)")

    def _action_clear_gap_preview(self) -> None:
        self._gap_candidates = []
        if self._gap_preview_layer is not None:
            try:
                self.viewer.layers.remove(self._gap_preview_layer)
            except (KeyError, ValueError):
                pass
            self._gap_preview_layer = None
        self._announce("cleared gap preview")

    def _action_merge_close_junctions(self) -> None:
        d_max = float(self._merge_junction_spin.value())
        n = self.editor.merge_close_junctions(d_max=d_max)
        if n:
            self._refresh_all()
        self._announce(f"merged {n} junction node(s) at ≤{d_max:g} px")

    def _action_cycles(self) -> None:
        n = self.editor.select_cycles()
        cycles = self.editor.find_cycles()
        self._announce(f"highlighted {n} new node(s) from "
                        f"{len(cycles)} cycle(s)")
        self._refresh_selection_overlays()
        self._refresh_status()

    def _action_streamline(self) -> None:
        n = self.editor.streamline()
        self._announce(f"streamlined: removed {n} degree-2 node(s)")
        self._refresh_all()

    def _action_clear(self) -> None:
        self.editor.clear_selection()
        self._announce("selection cleared")
        self._refresh_selection_overlays()
        self._refresh_status()

    def _action_undo(self) -> None:
        ok = self.editor.undo()
        self._announce("undo applied" if ok else "(nothing to undo)")
        if ok:
            self._refresh_all()
        self._refresh_status()

    def _action_save(self) -> None:
        self.editor.save(self.default_save_path)
        self._announce(f"saved → {self.default_save_path}")
        try:
            self._QMessageBox.information(
                self.viewer.window._qt_window,
                "net3 edit · saved",
                f"Saved graph to:\n{self.default_save_path}",
            )
        except Exception:
            pass
        self._refresh_status()

    def _action_toggle_true_tip(self) -> None:
        """If any selected node is already a true-tip, unmark all selected;
        otherwise flag all selected."""
        if not self.editor.selected:
            self._announce("no selection to flag")
            return
        already = {n for n in self.editor.selected
                    if self.editor.graph.nodes[n].get("true_tip")}
        if already == set(self.editor.selected):
            n = self.editor.unmark_selected_as_true_tips()
            self._announce(f"cleared true-tip on {n} node(s)")
        else:
            n = self.editor.mark_selected_as_true_tips()
            self._announce(f"marked {n} node(s) as true tips")
        # Deselect the flagged nodes so the next click starts fresh.
        self.editor.selected.clear()
        self._refresh_nodes_layer()
        self._refresh_selection_overlays()
        self._refresh_status()

    def _action_prune_unmarked_tips(self) -> None:
        n_flagged = len(self.editor.true_tip_nodes())
        n = self.editor.prune_unmarked_tips()
        self._announce(
            f"pruned {n} unmarked tip(s)  ({n_flagged} true-tip(s) preserved)")
        self._refresh_edges_layer()
        self._refresh_nodes_layer()
        self._refresh_selection_overlays()
        self._refresh_status()

    def _action_shorten_selected_tips(self) -> None:
        trim = int(self._shorten_trim_spin.value()) if getattr(
            self, "_shorten_trim_spin", None) is not None else 15
        n = self.editor.shorten_selected_tips(float(trim))
        self._announce(f"shortened {n} tip(s) by {trim} px")
        self._refresh_edges_layer()
        self._refresh_nodes_layer()
        self._refresh_selection_overlays()
        self._refresh_status()

    def _toggle_intermediates(self) -> None:
        self._show_intermediates = not self._show_intermediates
        if getattr(self, "_intermediates_btn", None) is not None:
            self._intermediates_btn.setChecked(self._show_intermediates)
        self._announce(
            f"intermediates {'shown' if self._show_intermediates else 'hidden'}"
        )
        self._refresh_nodes_layer()
        self._refresh_selection_overlays()
        self._refresh_status()

    # ── Status panel ────────────────────────────────────────────

    def _refresh_status(self) -> None:
        last = getattr(self, "_last_action", "(no action yet)")
        mode_label = {"select": "Select",
                       "rect": "Region-Select",
                       "add": "Add Node",
                       "move": "Move Node"}.get(self._mode, self._mode)
        text = (
            f"<b>mode:</b> {mode_label}<br>"
            f"<b>graph:</b> {self.editor.n_nodes} nodes, "
            f"{self.editor.n_edges} edges<br>"
            f"<b>selected:</b> {len(self.editor.selected)} nodes, "
            f"{len(self.editor.selected_edges)} edges<br>"
            f"<b>undo depth:</b> {len(self.editor._undo_stack)}<br>"
            f"<br><b>last:</b> {last}<br>"
            f"<br><b>save target:</b><br>{self.default_save_path}"
        )
        if self._status_lbl is not None:
            self._status_lbl.setText(text)

    # ── Run loop ────────────────────────────────────────────────

    def run(self) -> None:
        self._napari.run()
