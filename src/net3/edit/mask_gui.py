"""
napari-based frontend for net3's MaskEditor.

Requires the ``[gui]`` install extra (``pip install net3[gui]``).  The
backend in :mod:`net3.edit.mask` is pure-Python; this module only
wires up napari layers + a small dock panel.

Public entry point
------------------
    from net3.edit.mask_gui import open_mask_editor
    open_mask_editor(mask="mask.tif")                          # bare
    open_mask_editor(mask="mask.tif", image="raw.tif")          # + reference
    open_mask_editor(mask="mask.tif", probs="probs.tif")        # + threshold panel
    open_mask_editor(mask=arr, reference_layers=[...])          # advanced

Reference-layer extension hook
------------------------------
``reference_layers`` accepts a list of dicts, each describing an extra
``napari.layers.Image``-shaped layer to add underneath the mask:

    reference_layers=[
        {"data": tile_video, "name": "tile 14", "affine": aff_14,
         "colormap": "gray", "contrast_limits": (0, 4095),
         "visible": False, "blending": "additive"},
        ...
    ]

Only ``data`` is required.  Everything else defers to napari's defaults.
Time-varying data (T×H×W) works out of the box — napari's built-in time
slider syncs across all layers with a matching T dim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .mask import MaskEditor, _load_image_2d, _load_probs_2d


def _coerce_to_array(x):
    """Accept an ndarray or a path; return an ndarray."""
    if isinstance(x, (str, Path)):
        return _load_image_2d(Path(x))
    return np.asarray(x)


def _coerce_probs(x):
    if isinstance(x, (str, Path)):
        return _load_probs_2d(Path(x))
    return np.asarray(x, dtype=np.float32)


def open_mask_editor(
    mask,
    image=None,
    probs=None,
    reference_layers: Optional[Sequence[dict]] = None,
    on_ready=None,
    save_path: Optional[str | Path] = None,
    window_title: Optional[str] = None,
) -> MaskEditor:
    """Open the mask editor.  Blocks until the napari window is closed.

    Parameters
    ----------
    mask : ndarray | str | Path
        2-D mask.  Required.
    image : ndarray | str | Path, optional
        2-D grayscale reference (shown underneath the mask).
    probs : ndarray | str | Path, optional
        2-D probability map in [0, 1].  Enables the threshold + cleanup
        panel and the Reset button.
    reference_layers : list of dict, optional
        Extra napari image-layer kwargs to add underneath.  See module
        docstring for schema.  Used e.g. by PerTileFlow to overlay
        per-tile videos with affine transforms.
    on_ready : callable(viewer, editor), optional
        Hook run after all default layers + dock panels are built.
        Use to inject custom dock widgets (e.g. a tile-loader dropdown
        for on-demand video overlays) that need access to the live
        napari ``Viewer`` and the :class:`MaskEditor` state.  Anything
        the callback adds becomes part of the running session; returning
        None is fine.
    save_path : str | Path, optional
        Default target for the Save Mask button.  Falls back to the
        original ``mask`` path if it was a path, else the CWD.
    window_title : str, optional
        Napari window title.

    Returns
    -------
    MaskEditor
        The (possibly-edited) editor.  ``.mask`` is the final state.
    """
    if save_path is None and isinstance(mask, (str, Path)):
        save_path = Path(mask)
    elif save_path is not None:
        save_path = Path(save_path)

    mask_arr = _coerce_to_array(mask)
    image_arr = _coerce_to_array(image) if image is not None else None
    probs_arr = _coerce_probs(probs) if probs is not None else None

    editor = MaskEditor(mask=mask_arr, image=image_arr, probs=probs_arr)

    if window_title is None:
        window_title = f"net3 edit-mask · {save_path.name if save_path else 'unsaved'}"

    app = _MaskEditorApp(
        editor=editor,
        reference_layers=list(reference_layers) if reference_layers else [],
        save_path=save_path,
        window_title=window_title,
    )
    if on_ready is not None:
        on_ready(app.viewer, editor)
    app.run()
    return editor


# ── Frontend ──────────────────────────────────────────────────────


class _MaskEditorApp:
    """Owns the napari viewer + layers + input bindings for MaskEditor."""

    def __init__(
        self,
        editor: MaskEditor,
        reference_layers: Sequence[dict],
        save_path: Optional[Path],
        window_title: str,
    ):
        import napari
        self._napari = napari
        self.editor = editor
        self.save_path = save_path
        self._image_h, self._image_w = editor.mask.shape

        self.viewer = napari.Viewer(title=window_title)
        self._prob_layer = None
        self._image_layer = None
        self._mask_layer = None
        self._threshold_widget = None
        self._median_widget = None
        self._reset_widget = None
        self._save_widget = None

        self._build_layers(reference_layers)
        self._build_dock()
        self._bind_keys()

    # ── Layers ─────────────────────────────────────────────────

    def _build_layers(self, reference_layers: Sequence[dict]) -> None:
        # Bottom to top: extras → image → probs → mask
        for spec in reference_layers:
            spec = dict(spec)
            data = spec.pop("data")
            self.viewer.add_image(data, **spec)

        if self.editor.image is not None:
            self._image_layer = self.viewer.add_image(
                self.editor.image,
                name="image",
                colormap="gray",
                opacity=1.0,
                blending="translucent",
            )

        if self.editor.probs is not None:
            self._prob_layer = self.viewer.add_image(
                self.editor.probs,
                name="probs",
                colormap="magma",
                contrast_limits=[0.0, 1.0],
                opacity=0.5,
                blending="additive",
                visible=False,
            )

        self._mask_layer = self.viewer.add_labels(
            self.editor.mask.astype(np.uint8),
            name="mask",
            opacity=0.4,
        )
        self.viewer.reset_view()

    def _sync_backend_from_labels(self) -> None:
        """Pull the (possibly-painted) Labels-layer data into the backend."""
        if self._mask_layer is None:
            return
        self.editor.mask = (self._mask_layer.data > 0)

    def _sync_labels_from_backend(self) -> None:
        """Push the backend mask into the Labels layer."""
        if self._mask_layer is None:
            return
        self._mask_layer.data = self.editor.mask.astype(np.uint8)
        # napari's Labels layer sometimes keeps a stale render after
        # data reassignment (same shape+dtype); force a repaint.
        try:
            self._mask_layer.refresh()
        except Exception:
            pass

    # ── Dock panel ─────────────────────────────────────────────

    def _build_dock(self) -> None:
        from magicgui import magicgui

        has_probs = self.editor.probs is not None

        if has_probs:
            @magicgui(
                show_probs={"widget_type": "CheckBox", "value": False,
                            "label": "Show probabilities"},
                show_mask={"widget_type": "CheckBox", "value": True,
                           "label": "Show mask"},
                threshold={"widget_type": "FloatSlider", "min": 0.0,
                            "max": 1.0, "step": 0.01, "value": 0.5,
                            "label": "Threshold"},
                remove_small={"widget_type": "SpinBox", "min": 0,
                              "max": 1_000_000, "step": 500, "value": 0,
                              "label": "Remove objects < (px)"},
                fill_holes={"widget_type": "SpinBox", "min": 0,
                            "max": 1_000_000, "step": 500, "value": 0,
                            "label": "Fill holes < (px)"},
                lock_mask={"widget_type": "CheckBox", "value": False,
                           "label": "🔒 Lock mask (preserve edits)"},
                auto_call=True,
            )
            def threshold_widget(
                show_probs: bool = False,
                show_mask: bool = True,
                threshold: float = 0.5,
                remove_small: int = 0,
                fill_holes: int = 0,
                lock_mask: bool = False,
            ):
                if self._prob_layer is not None:
                    self._prob_layer.visible = show_probs
                self._mask_layer.visible = show_mask
                lock_changed = (lock_mask != self.editor.locked)
                self.editor.locked = lock_mask
                if not lock_mask and show_mask and not lock_changed:
                    self._sync_backend_from_labels()
                    self.editor.threshold(threshold, remove_small, fill_holes)
                    self._sync_labels_from_backend()

            self._threshold_widget = threshold_widget
            self.viewer.window.add_dock_widget(
                threshold_widget, name="Threshold + cleanup", area="right",
            )

        @magicgui(
            radius={"widget_type": "SpinBox", "min": 1, "max": 50,
                    "value": 3, "label": "Radius"},
            call_button="Apply median filter",
        )
        def median_widget(radius: int = 3):
            self._sync_backend_from_labels()
            n_before = self.editor.n_foreground
            self.editor.apply_median_filter(radius=radius)
            n_after = self.editor.n_foreground
            self._sync_labels_from_backend()
            if self._threshold_widget is not None:
                self._threshold_widget.lock_mask.value = True
            print(f"[net3 edit-mask] median filter (radius={radius}, "
                  f"size={2*radius+1}): {n_before:,} → {n_after:,} "
                  f"vessel px ({n_after - n_before:+,})")

        self._median_widget = median_widget
        self.viewer.window.add_dock_widget(
            median_widget, name="Median filter", area="right",
        )

        if has_probs:
            @magicgui(call_button="Reset from probabilities")
            def reset_widget():
                thresh = (self._threshold_widget.threshold.value
                          if self._threshold_widget else 0.5)
                rs = (self._threshold_widget.remove_small.value
                       if self._threshold_widget else 0)
                fh = (self._threshold_widget.fill_holes.value
                       if self._threshold_widget else 0)
                self._sync_backend_from_labels()
                self.editor.reset_from_probabilities(thresh, rs, fh)
                self._sync_labels_from_backend()
                if self._threshold_widget is not None:
                    self._threshold_widget.lock_mask.value = False

            self._reset_widget = reset_widget
            self.viewer.window.add_dock_widget(
                reset_widget, name="Reset", area="right",
            )

        @magicgui(call_button="Save mask")
        def save_widget():
            self._save()

        self._save_widget = save_widget
        self.viewer.window.add_dock_widget(
            save_widget, name="Save", area="right",
        )

    # ── Keybindings ─────────────────────────────────────────────

    def _bind_keys(self) -> None:
        v = self.viewer

        @v.bind_key("e", overwrite=True)
        def paint(_):
            self._enter_paint_mode()

        @v.bind_key("x", overwrite=True)
        def erase(_):
            self._enter_erase_mode()

        @v.bind_key("Escape", overwrite=True)
        def pan(_):
            if self._mask_layer is not None:
                self._mask_layer.mode = "pan_zoom"

        @v.bind_key("s", overwrite=True)
        def save(_):
            self._save()

        @v.bind_key("u", overwrite=True)
        def undo(_):
            self._sync_backend_from_labels()
            if self.editor.undo():
                self._sync_labels_from_backend()

    def _enter_paint_mode(self) -> None:
        if self._mask_layer is None:
            return
        self._sync_backend_from_labels()
        self.editor.push_undo()
        self._mask_layer.visible = True
        self.viewer.layers.selection.active = self._mask_layer
        self._mask_layer.mode = "paint"
        self._mask_layer.selected_label = 1
        self.editor.locked = True
        if self._threshold_widget is not None:
            self._threshold_widget.lock_mask.value = True
            self._threshold_widget.show_mask.value = True

    def _enter_erase_mode(self) -> None:
        if self._mask_layer is None:
            return
        self._sync_backend_from_labels()
        self.editor.push_undo()
        self._mask_layer.visible = True
        self.viewer.layers.selection.active = self._mask_layer
        self._mask_layer.mode = "erase"
        self.editor.locked = True
        if self._threshold_widget is not None:
            self._threshold_widget.lock_mask.value = True
            self._threshold_widget.show_mask.value = True

    # ── Save ────────────────────────────────────────────────────

    def _save(self) -> None:
        target = self.save_path or (Path.cwd() / "mask_edited.tif")
        self._sync_backend_from_labels()
        out = self.editor.save_mask(target)
        n = self.editor.n_foreground
        print(f"[net3 edit-mask] saved → {out}  ({n:,} vessel px)")

    # ── Run ─────────────────────────────────────────────────────

    def run(self) -> None:
        self._napari.run()
