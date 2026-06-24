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
from typing import List, Optional

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
EDGE_WIDTH = 1.2
NEW_NODE_TOLERANCE_FALLBACK = 8.0


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
        self._nodes_layer = None
        self._edges_layer = None
        # Stable ordering for the points layer — backend node ids in
        # whatever order we built the points array.  Used to map napari
        # selection indices back to graph node ids.
        self._node_index_order: List = []

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
        self._refresh_nodes_layer()
        self._refresh_edges_layer()
        self.viewer.reset_view()

    def _node_positions_and_colors(self):
        positions = []
        face_colors = []
        sizes = []
        ids: List = []
        G = self.editor.graph
        for n, d in G.nodes(data=True):
            x = d.get("x"); y = d.get("y")
            if x is None or y is None:
                continue
            positions.append([self._image_h - float(y), float(x)])
            ids.append(n)
            deg = G.degree(n)
            if n in self.editor.selected:
                face_colors.append(NODE_COLOR_SELECTED)
                sizes.append(NODE_SIZE_SELECTED)
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
        segments = []
        G = self.editor.graph
        for u, v in G.edges:
            xu = float(G.nodes[u].get("x", 0.0))
            yu = float(G.nodes[u].get("y", 0.0))
            xv = float(G.nodes[v].get("x", 0.0))
            yv = float(G.nodes[v].get("y", 0.0))
            segments.append([
                [self._image_h - yu, xu],
                [self._image_h - yv, xv],
            ])
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
        self._refresh_edges_layer()
        self._refresh_nodes_layer()
        self._refresh_status()

    # ── Build the dock widget ───────────────────────────────────

    def _build_dock(self, QPushButton, QLabel, QVBoxLayout, QWidget) -> None:
        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet(
            "padding: 6px; font-family: monospace; font-size: 11px;")

        layout = QVBoxLayout()
        layout.addWidget(self._status_lbl)

        def _btn(label: str, tooltip: str, callback):
            b = QPushButton(label)
            b.setToolTip(tooltip)
            b.clicked.connect(callback)
            layout.addWidget(b)
            return b

        _btn("Delete selected  (d)",
              "Remove every selected node and its edges.",
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

        # Click handling: left-click toggles nearest-node selection;
        # shift+left-click on empty canvas creates a new node.
        @v.mouse_drag_callbacks.append
        def _on_click(viewer, event):
            if event.button != 1:  # only left-click
                return
            pos = event.position
            if pos is None or len(pos) < 2:
                return
            row, col = float(pos[-2]), float(pos[-1])
            graph_x = col
            graph_y = self._image_h - row
            tol = self._click_tolerance()
            hit = self.editor.node_at(graph_x, graph_y, tolerance=tol)
            shift = "Shift" in event.modifiers
            if hit is None:
                if shift:
                    self.editor.add_node(graph_x, graph_y)
                    self._announce(f"created node at ({graph_x:.1f}, "
                                    f"{graph_y:.1f})")
                    self._refresh_all()
                else:
                    self._announce("(click missed — try closer to a node, "
                                    "or Shift-click to create one)")
                    self._refresh_status()
            else:
                self.editor.toggle_selection(hit)
                action = ("selected" if hit in self.editor.selected
                          else "deselected")
                self._announce(f"node {hit} {action}")
                self._refresh_nodes_layer()
                self._refresh_status()

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

    # ── Actions ─────────────────────────────────────────────────

    def _announce(self, msg: str) -> None:
        self._last_action = msg

    def _action_delete(self) -> None:
        n = self.editor.delete_selected()
        self._announce(f"deleted {n} node(s)")
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
        self._refresh_nodes_layer()
        self._refresh_status()

    def _action_streamline(self) -> None:
        n = self.editor.streamline()
        self._announce(f"streamlined: removed {n} degree-2 node(s)")
        self._refresh_all()

    def _action_clear(self) -> None:
        self.editor.clear_selection()
        self._announce("selection cleared")
        self._refresh_nodes_layer()
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
        text = (
            f"<b>graph:</b> {self.editor.n_nodes} nodes, "
            f"{self.editor.n_edges} edges<br>"
            f"<b>selected:</b> {len(self.editor.selected)}<br>"
            f"<b>undo depth:</b> {len(self.editor._undo_stack)}<br>"
            f"<br><b>last:</b> {last}<br>"
            f"<br><b>save target:</b><br>{self.default_save_path}"
        )
        if self._status_lbl is not None:
            self._status_lbl.setText(text)

    # ── Run loop ────────────────────────────────────────────────

    def run(self) -> None:
        self._napari.run()
