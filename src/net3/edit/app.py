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
) -> GraphEditor:
    """Open the editor on the given graph + optional mask / distance
    map.  Blocks until the napari window is closed.  Returns the
    (possibly-edited) :class:`GraphEditor` instance.

    Save target: ``save_path`` if provided, otherwise overwrite
    ``graph_path`` on Ctrl-S / button click (with a confirmation
    dialog).
    """
    editor = GraphEditor.from_gpickle(
        graph_path,
        mask_path=mask_path,
        distance_map_path=distance_map_path,
    )
    _maybe_warn_sparse(editor)
    app = _EditorApp(editor,
                      default_save_path=Path(save_path or graph_path))
    app.run()
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
    ):
        import napari
        from qtpy.QtWidgets import (
            QPushButton, QLabel, QVBoxLayout, QWidget, QMessageBox,
        )
        self._napari = napari
        self._QMessageBox = QMessageBox
        self.editor = editor
        self.default_save_path = default_save_path
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
        self._mode: str = "select"   # "select" | "rect" | "add"
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
        rebuilt independently so selection toggles stay O(1)."""
        positions = []
        face_colors = []
        sizes = []
        ids: List = []
        xy_by_id: dict = {}
        G = self.editor.graph
        for n, d in G.nodes(data=True):
            x = d.get("x"); y = d.get("y")
            if x is None or y is None:
                continue
            positions.append([self._image_h - float(y), float(x)])
            ids.append(n)
            xy_by_id[n] = (float(x), float(y))
            deg = G.degree(n)
            if deg >= 3:
                face_colors.append(NODE_COLOR_JUNCTION)
            elif deg == 1:
                face_colors.append(NODE_COLOR_TIP)
            else:
                face_colors.append(NODE_COLOR_UNSELECTED)
            sizes.append(NODE_SIZE_DEFAULT)
        self._node_index_order = ids
        self._node_xy_by_id = xy_by_id
        return (np.array(positions) if positions
                else np.zeros((0, 2))), face_colors, sizes

    def _refresh_nodes_layer(self) -> None:
        positions, colors, sizes = self._node_positions_and_colors()
        if self._nodes_layer is not None:
            try:
                self.viewer.layers.remove(self._nodes_layer)
            except (KeyError, ValueError):
                pass
            self._nodes_layer = None
        if len(positions) == 0:
            return
        self._nodes_layer = self.viewer.add_points(
            positions,
            name="nodes",
            face_color=colors,
            size=sizes,
            symbol="o",
        )

    def _refresh_edges_layer(self) -> None:
        """Base edges layer — uniform orange, never modified for
        selection.  Selected edges are drawn yellow on top in a small
        overlay (see `_refresh_selection_overlays`)."""
        segments = []
        edge_pairs: List[Tuple] = []
        G = self.editor.graph
        canon = GraphEditor._canonical_edge
        for u, v in G.edges:
            xu = float(G.nodes[u].get("x", 0.0))
            yu = float(G.nodes[u].get("y", 0.0))
            xv = float(G.nodes[v].get("x", 0.0))
            yv = float(G.nodes[v].get("y", 0.0))
            segments.append([
                [self._image_h - yu, xu],
                [self._image_h - yv, xv],
            ])
            edge_pairs.append(canon(u, v))
        self._edge_index_order = edge_pairs
        self._edge_pair_to_index = {p: i for i, p in enumerate(edge_pairs)}
        if self._edges_layer is not None:
            try:
                self.viewer.layers.remove(self._edges_layer)
            except (KeyError, ValueError):
                pass
            self._edges_layer = None
        if not segments:
            return
        self._edges_layer = self.viewer.add_shapes(
            segments,
            shape_type="line",
            edge_color=EDGE_COLOR,
            edge_width=EDGE_WIDTH,
            face_color="transparent",
            opacity=0.85,
            name="edges",
        )
        # Move edges to bottom so node markers render on top.
        self.viewer.layers.move(
            self.viewer.layers.index(self._edges_layer), 1,
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
        _mode_btn("rect", "Rect-Select  (key: 2)",
                   "Left drag draws a rubber-band rectangle and "
                   "selects every node inside on release.  "
                   "Hold Shift for additive.")
        _mode_btn("add", "Add Node  (key: 3)",
                   "Left click creates a new node (snapped to the "
                   "local distance-transform peak if a distance "
                   "map is loaded).")
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
        _btn("Connect 2 selected  (e)",
              "Add an edge between the exactly-two selected nodes.",
              self._action_connect)
        _btn("Select cycles  (m)",
              "Highlight every node that participates in a cycle.",
              self._action_cycles)
        _btn("Streamline  (n)",
              "Collapse every degree-2 redundant node by merging edges.",
              self._action_streamline)
        _btn("Clear selection  (c)",
              "Drop the current node selection.",
              self._action_clear)
        _btn("Undo  (z)",
              "Revert the most recent mutation.",
              self._action_undo)
        _btn("Save  (s)",
              f"Save graph to {self.default_save_path}.",
              self._action_save)
        layout.addStretch(1)

        widget = QWidget()
        widget.setLayout(layout)
        self.viewer.window.add_dock_widget(
            widget, name="net3 edit", area="right",
        )

    # ── Key bindings + mouse ────────────────────────────────────

    def _bind_inputs(self) -> None:
        v = self.viewer

        @v.bind_key("d", overwrite=True)
        def _(viewer):
            self._action_delete()

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
        # Add-Node mode (or Shift+click in any mode) creates a new node.
        if self._mode == "add" or shift:
            n = self.editor.add_node(gx, gy, snap_window=SNAP_WINDOW_DEFAULT)
            r = self.editor.graph.nodes[n].get("radius", 0.0)
            self._announce(
                f"created node {n} at ({gx:.1f}, {gy:.1f})  r={r:.1f}")
            self._refresh_all()
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
        if mode not in ("select", "rect", "add"):
            return
        self._mode = mode
        for k, b in self._mode_btns.items():
            try:
                b.setChecked(k == mode)
            except Exception:
                pass
        self._ensure_rect_tool_layer()
        if mode == "rect" and self._rect_tool_layer is not None:
            try:
                self._rect_tool_layer.mode = "add_rectangle"
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
                  "rect": "Rect-Select",
                  "add": "Add Node"}[mode]
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

            @self._rect_tool_layer.mouse_drag_callbacks.append
            def _on_layer_drag(layer, event):
                yield from self._rect_layer_drag(layer, event)

        except Exception as e:
            print(f"  rect tool layer init failed: {e!r}")
            self._rect_tool_layer = None

    def _rect_layer_drag(self, layer, event):
        """Generator: runs alongside napari's add_rectangle to read
        the finished rectangle and apply the selection."""
        if self._mode != "rect" or event.button != 1:
            return
        start = self._event_graph_xy(event)
        if start is None:
            return
        shift = "Shift" in event.modifiers
        yield  # let napari's add_rectangle press handler run
        while event.type == "mouse_move":
            yield
        # mouse_release — napari just committed the rectangle, safe to read.
        end = self._event_graph_xy(event)
        if end is None:
            return
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        moved = (dx * dx + dy * dy) > (DRAG_THRESHOLD_PX * DRAG_THRESHOLD_PX)
        if moved:
            self.editor.select_in_rect(
                start[0], start[1], end[0], end[1], additive=shift,
            )
            self._announce(
                f"rectangle-selected: {len(self.editor.selected)} node(s)"
                + ("  (additive)" if shift else ""))
            self._refresh_selection_overlays()
        # Always clean up the napari-drawn rectangle so the next drag
        # starts from an empty layer.  Safe here because napari's
        # add_rectangle generator has already returned.
        try:
            layer.data = []
        except Exception:
            pass
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
        else:
            self._announce("connect needs exactly 2 selected nodes "
                            "(and edge must not already exist)")
        self._refresh_all()

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
        self._refresh_all()

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

    # ── Status panel ────────────────────────────────────────────

    def _refresh_status(self) -> None:
        last = getattr(self, "_last_action", "(no action yet)")
        mode_label = {"select": "Select",
                       "rect": "Rect-Select",
                       "add": "Add Node"}.get(self._mode, self._mode)
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
