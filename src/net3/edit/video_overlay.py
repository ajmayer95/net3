"""Optional per-tile video overlay for the graph editor.

Feature-flagged: the editor works fine without it.  Enable by passing
``--tile-positions`` + ``--video-dir`` + ``--video-pattern`` to
``net3 edit``.

Design
------
- Videos are loaded lazily — one tile at a time, on demand.  A 700-frame
  700×640 TIFF stack is ~600 MB in RAM; loading all 84 tiles up front
  would exhaust memory on most laptops.
- Napari displays the loaded video as an Image layer with a 2D affine
  that places it at the tile's mosaic position + scales it to the
  tile's mosaic footprint.  Napari's built-in time slider scrubs frames.
- Only one tile's video is shown at a time; loading a new tile
  automatically removes the previous overlay.

Tile-position JSON format
-------------------------
A dict mapping tile-id (str or int) to a dict with keys::

    {
      "1": {"top_left_x": 0, "top_left_y": 0,
              "tile_width": 640, "tile_height": 480},
      "2": {"top_left_x": 640, "top_left_y": 0, ...},
      ...
    }

Compatible with PerTileFlow's ``tile_positions_manual.json``.  Extra
keys (rotation, affine, etc.) are ignored.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_tile_positions(path: str | Path) -> Dict[int, dict]:
    """Parse a tile-positions JSON.  Returns ``{tile_id: {top_left_x,
    top_left_y, [scale_x, scale_y], [tile_width, tile_height]}}``.

    Accepts two common schemas:

    1. **Simple / flat**: top-level dict maps tile-id → position info::

           {"1": {"top_left_x": 0, "top_left_y": 0,
                    "tile_width": 640, "tile_height": 480}, ...}

    2. **PerTileFlow-style nested**: outer dict has a ``tiles`` key
       whose value is the per-tile mapping, with ``translate_x/y`` +
       ``scale_x/y`` (no explicit width/height — scaling is per-pixel)::

           {"reference_vid": 6,
            "tiles": {"6": {"translate_x": -593.0, "translate_y": -2494.2,
                             "scale_x": 1.0016, "scale_y": 1.0038}, ...},
            "relative_offsets": {...}}

    For the PerTileFlow schema, ``translate_x/y`` may be negative
    (arbitrary reference origin).  We normalize by subtracting the
    minimum so mosaic display coords start at (0, 0).
    """
    p = Path(path)
    with open(p) as f:
        raw = json.load(f)
    # Unwrap PerTileFlow nesting
    if (isinstance(raw, dict) and 'tiles' in raw
            and isinstance(raw['tiles'], dict)):
        raw = raw['tiles']
    out: Dict[int, dict] = {}
    for k, v in raw.items():
        try:
            tid = int(k)
        except (ValueError, TypeError):
            continue
        if not isinstance(v, dict):
            continue
        # Position (either schema; prefer top_left_ names, fall back to translate_)
        tx = v.get('top_left_x')
        if tx is None:
            tx = v.get('translate_x') or v.get('x') or v.get('origin_x') or 0
        ty = v.get('top_left_y')
        if ty is None:
            ty = v.get('translate_y') or v.get('y') or v.get('origin_y') or 0
        entry: dict = dict(top_left_x=float(tx), top_left_y=float(ty))
        # PerTileFlow: per-pixel scale factors
        if 'scale_x' in v:
            entry['scale_x'] = float(v['scale_x'])
        if 'scale_y' in v:
            entry['scale_y'] = float(v['scale_y'])
        # Simple: explicit tile size in mosaic px
        w = v.get('tile_width') or v.get('width') or v.get('w')
        h = v.get('tile_height') or v.get('height') or v.get('h')
        if w is not None:
            entry['tile_width'] = float(w)
        if h is not None:
            entry['tile_height'] = float(h)
        # Full 3x3 affine (plexusflow / rotated stitching). Off-diagonal
        # elements encode rotation — scale_x/y alone won't reproduce the
        # tile placement when tiles are rotated.
        A = v.get('affine_matrix')
        if A is not None:
            try:
                import numpy as _np
                entry['affine_matrix'] = _np.asarray(A, dtype=float)
            except Exception:
                pass
        out[tid] = entry
    # Global tile-offset normalization: shift so the leftmost *FOOTPRINT
    # CORNER* (not the translate) sits at (0, 0). With rotated tiles the
    # translate is the raw-frame origin, but the tile's actual bbox
    # extends into negative coords via the rotation.  Using the min
    # translate under-shifts and leaves the tile grid overhanging into
    # negative x/y relative to a fuse output that was cropped to the
    # positive-coord bbox.
    if out:
        import numpy as _np
        # Standard raw-frame corner set; scaled at video-load time
        RAW_H, RAW_W = 704.0, 640.0
        corners = _np.array([[0, 0], [0, RAW_W],
                              [RAW_H, RAW_W], [RAW_H, 0]], dtype=float)
        all_y = []; all_x = []
        for e in out.values():
            A = e.get('affine_matrix')
            if A is not None:
                w = corners @ A[:2, :2].T + A[:2, 2]
                all_y += list(w[:, 0]); all_x += list(w[:, 1])
            else:
                sy = float(e.get('scale_y', 1.0)); sx = float(e.get('scale_x', 1.0))
                ty = float(e['top_left_y']);       tx = float(e['top_left_x'])
                all_y += [ty, ty + RAW_H * sy]
                all_x += [tx, tx + RAW_W * sx]
        min_y = min(all_y); min_x = min(all_x)
        # Always shift: the fused canvas starts at the footprint's min corner
        # whether the stitch coordinates are negative (21S) or all positive
        # (27S: tiles at ~+10,000 px landed that far off when this ran only
        # for negative minima).
        if min_x != 0 or min_y != 0:
            for e in out.values():
                e['top_left_x'] -= min_x
                e['top_left_y'] -= min_y
                A = e.get('affine_matrix')
                if A is not None:
                    A2 = A.copy()
                    A2[0, 2] -= min_y
                    A2[1, 2] -= min_x
                    e['affine_matrix'] = A2
    return out


def resolve_video_path(video_dirs: List[Path], pattern: str,
                        tile_id: int) -> Optional[Path]:
    """Resolve a tile-id to a video file across multiple candidate dirs.

    ``pattern`` is a filename pattern like ``"vid{id}.tif"`` or
    ``"10x 250fps loc{id}_C001H001S0001"``.  ``{id}`` and ``{vid}`` are
    both substituted with ``tile_id``.  If the resolved pattern has no
    extension, common video/image extensions are probed.  If the pattern
    matches a directory (folder of frames), returns that directory.

    Returns the first hit across ``video_dirs`` in order.
    """
    substituted = pattern.replace('{id}', str(tile_id)).replace(
        '{vid}', str(tile_id))
    has_ext = bool(re.search(r'\.[A-Za-z0-9]{2,5}$', substituted))
    candidates: List[str] = ([substituted] if has_ext else
                              [substituted + ext for ext in
                               ('.tif', '.tiff', '.TIF', '.TIFF',
                                '.mp4', '.avi', '.mov', '')])
    for vdir in video_dirs:
        vdir = Path(vdir)
        if not vdir.is_dir():
            continue
        # 1) Literal file / directory hit
        for c in candidates:
            path = vdir / c
            if path.exists():
                return path
        # 2) Glob fallback — respect glob metacharacters (*, ?, [) in the
        # pattern. Try the substituted string as-is, then with common
        # extensions appended.
        globs = [substituted] if has_ext else \
                [substituted + ext for ext in
                 ('', '.tif', '.tiff', '.TIF', '.TIFF',
                  '.mp4', '.avi', '.mov')]
        for g in globs:
            if any(ch in g for ch in ('*', '?', '[')):
                matches = sorted(vdir.glob(g))
                if matches:
                    return matches[0]
    return None


def load_video(path: Path) -> "np.ndarray | None":
    """Load a video file (TIFF stack) or a folder of frame TIFFs.
    Returns a (T, H, W) or (T, H, W, C) ndarray, or None on failure.
    """
    import numpy as np
    if path.is_dir():
        # Folder of frames — glob TIFFs, sort, stack.
        import tifffile
        frames = sorted(path.glob('*.tif')) + sorted(path.glob('*.tiff'))
        if not frames:
            return None
        stacks = [tifffile.imread(str(f)) for f in frames]
        arr = np.stack(stacks, axis=0)
        return arr
    # Single-file TIFF stack
    if path.suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        return tifffile.imread(str(path))
    # Video container — use pyav or opencv
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f if f.ndim == 2 else
                           cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        cap.release()
        if not frames:
            return None
        import numpy as np
        return np.stack(frames, axis=0)
    except ImportError:
        return None


def compute_tile_transform(tile: dict, video_shape: Tuple[int, int],
                            tile_raw_shape: Optional[Tuple[int, int]] = None,
                            ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return ``(scale_yx, translate_yx)`` for napari's ``scale`` +
    ``translate`` kwargs.

    Scale-derivation priority:
      1. Explicit ``scale_x`` / ``scale_y`` (PerTileFlow schema — per-pixel
         scale factor calibrated against **raw** video pixels).
      2. Fall back to ``tile_width`` / ``tile_height`` divided by video
         shape (simple schema — explicit mosaic footprint).
      3. Fall back to 1.0 (identity) if neither is present.

    Downsample compensation (PerTileFlow bundles ship 2×-downsampled
    videos while their tile_positions_manual.json still uses scales
    calibrated against 704×640 raw acquisitions): if ``tile_raw_shape``
    is provided and larger than ``video_shape``, the scale is multiplied
    by the ratio so the video occupies its intended mosaic footprint
    instead of only the top-left quarter.
    """
    H_vid, W_vid = video_shape
    if 'scale_y' in tile:
        sy = float(tile['scale_y'])
    elif tile.get('tile_height') is not None:
        sy = float(tile['tile_height']) / float(H_vid)
    else:
        sy = 1.0
    if 'scale_x' in tile:
        sx = float(tile['scale_x'])
    elif tile.get('tile_width') is not None:
        sx = float(tile['tile_width']) / float(W_vid)
    else:
        sx = 1.0
    if tile_raw_shape is not None:
        H_raw, W_raw = tile_raw_shape
        if H_raw and H_vid and W_raw and W_vid:
            sy *= float(H_raw) / float(H_vid)
            sx *= float(W_raw) / float(W_vid)
    tx = float(tile.get('top_left_x', 0))
    ty = float(tile.get('top_left_y', 0))
    return ((sy, sx), (ty, tx))


# Back-compat alias for anything that imported the old name.
def compute_tile_affine(tile, video_shape):
    """Deprecated: returns a 2×3 affine.  Use ``compute_tile_transform``
    for napari's scale+translate kwargs (cleaner + dimension-robust)."""
    import numpy as np
    (sy, sx), (ty, tx) = compute_tile_transform(tile, video_shape)
    return np.array([[sy, 0.0, ty], [0.0, sx, tx]])


class VideoOverlayManager:
    """Owns the currently-loaded video layer.  Handles load/unload +
    tile selection.  Instantiate once per _EditorApp; call
    ``load_tile(tile_id)`` when the user picks a tile.
    """

    def __init__(self, viewer, tile_positions: Dict[int, dict],
                  video_dirs: List[Path], video_pattern: str,
                  tile_raw_shape: Optional[Tuple[int, int]] = (704, 640),
                  image_h: Optional[int] = None,
                  mask_shape: Optional[Tuple[int, int]] = None):
        """
        Parameters
        ----------
        tile_raw_shape : (H, W) or None
            Canonical raw acquisition frame size (default 704×640 for
            PerTileFlow 10× yolk-sac data).  Loaded videos smaller
            than this get their scale multiplied by the downsample
            factor so downsampled bundles occupy their full intended
            footprint on the mosaic.  Set to None to disable the
            compensation (e.g. for datasets where the tile-positions
            JSON was calibrated against the loaded video pixels
            directly).
        """
        self.viewer = viewer
        self.tile_positions = tile_positions
        self.video_dirs = [Path(d) for d in video_dirs]
        self.video_pattern = video_pattern
        self.tile_raw_shape = tile_raw_shape
        self.image_h = image_h  # currently unused (kept for future y-flip)
        # If mask_shape is given, compute the pad offset between the
        # tile-position coord frame (which now starts at footprint (0,0))
        # and the mask's own (0, 0). The fuse process typically pads the
        # canvas symmetrically around the footprint bbox → the leftmost
        # tile corner in mask-pixel coords sits at (pad_top, pad_left).
        self._pad_yx = (0.0, 0.0)
        if mask_shape is not None and tile_positions:
            import numpy as _np
            H_raw = float((tile_raw_shape or (704, 640))[0])
            W_raw = float((tile_raw_shape or (704, 640))[1])
            corners = _np.array([[0, 0], [0, W_raw],
                                  [H_raw, W_raw], [H_raw, 0]], dtype=float)
            all_y = []; all_x = []
            for e in tile_positions.values():
                A = e.get('affine_matrix')
                if A is not None:
                    w = corners @ A[:2, :2].T + A[:2, 2]
                    all_y += list(w[:, 0]); all_x += list(w[:, 1])
            if all_y and all_x:
                foot_h = max(all_y) - min(all_y)
                foot_w = max(all_x) - min(all_x)
                pad_y = max(0.0, (mask_shape[0] - foot_h) / 2.0)
                pad_x = max(0.0, (mask_shape[1] - foot_w) / 2.0)
                self._pad_yx = (pad_y, pad_x)
                print(f"[net3 edit] mask fuse-pad offset detected: "
                      f"(+{pad_y:.0f} row, +{pad_x:.0f} col)")
        self._current_layer = None
        self._current_tile: Optional[int] = None
        # Optional tile-outline + tile-label overlays (built lazily)
        self._boundary_layer = None
        self._label_layer = None

    def load_tile(self, tile_id: int, colormap: str = 'gray',
                   opacity: float = 1.0,
                   dim_mask_to: float = 0.35) -> bool:
        """Load `tile_id`'s video as an Image layer, remove any prior
        overlay.  Returns True on success.

        Contrast + blending match the plexusflow viewer:
          * Video is stretched to a 1%/99% percentile window and quantised
            to uint8 so the ~11-bit CMOS raw uses the full napari display
            range (raw uint16 lands mostly in the black end of gray).
          * ``opacity=0.7`` + ``blending='translucent'`` — mask remains
            visible through the video.  Drop the layer's opacity slider
            further for a lighter overlay.
        """
        tile = self.tile_positions.get(tile_id)
        if tile is None:
            print(f"[net3 edit] tile {tile_id} not in tile-positions JSON")
            return False
        path = resolve_video_path(
            self.video_dirs, self.video_pattern, tile_id)
        if path is None:
            print(f"[net3 edit] no video for tile {tile_id} "
                  f"(pattern={self.video_pattern!r}, "
                  f"dirs={[str(d) for d in self.video_dirs]})")
            return False
        print(f"[net3 edit] loading tile {tile_id} from {path.name} …")
        arr = load_video(path)
        if arr is None or arr.size == 0:
            print(f"[net3 edit] failed to load video: {path}")
            return False
        # Compute a 1%/99% percentile contrast window from a sparse sample.
        # We ALWAYS derive limits (even for pre-quantized uint8 realtime
        # videos), because napari's default (0, dtype-max) renders low-
        # dynamic-range CMOS captures as nearly-black.
        import numpy as _np
        sample = arr[::max(1, arr.shape[0] // 20)]
        _lo, _hi = _np.percentile(sample, (1.0, 99.0))
        if _hi <= _lo: _hi = _lo + 1.0
        contrast_lo, contrast_hi = float(_lo), float(_hi)
        # spatial shape = last two dims (works for both grey and RGB)
        spatial = arr.shape[-2:]
        # If the tile has a full 3x3 affine (plexusflow positions.json —
        # includes rotation), build a napari 4D affine directly.  Otherwise
        # fall back to scale+translate from the diagonal.
        A_tile = tile.get('affine_matrix')
        import numpy as _np
        use_affine = False
        affine44 = None
        if A_tile is not None and self.tile_raw_shape is not None:
            H_raw, W_raw = self.tile_raw_shape
            H_vid, W_vid = spatial
            # Pixel-scale correction: video may be pre-downsampled relative
            # to the raw acquisition the JSON's scales are calibrated to.
            ds_y = H_raw / float(H_vid) if H_vid else 1.0
            ds_x = W_raw / float(W_vid) if W_vid else 1.0
            S_pixel = _np.diag([ds_y, ds_x])
            M = A_tile[:2, :2] @ S_pixel   # 2×2 rotation·scale in mosaic coords
            t_yx = A_tile[:2, 2].astype(float)
            # Shift into mask-pixel coords by adding the fuse pad.
            t_yx = t_yx + _np.array(self._pad_yx)
            # Compose the graph's y-flip: net3 renders graph nodes at
            # (image_h - y). To land the video in the same coord system,
            # negate the y-row of M and set t_yx[0] = image_h - t_yx[0].
            if self.image_h is not None:
                M = M.copy()
                M[0, :] = -M[0, :]
                t_yx = t_yx.copy()
                t_yx[0] = float(self.image_h) - t_yx[0]
            affine44 = _np.eye(arr.ndim + 1)
            # Spatial axes are the LAST two — for (T,H,W) that's dims 1,2;
            # for (H,W) that's dims 0,1.
            spat = slice(arr.ndim - 2, arr.ndim)
            affine44[spat, spat] = M
            affine44[spat.start:spat.stop, -1] = t_yx
            use_affine = True
        if not use_affine:
            (sy, sx), (ty, tx) = compute_tile_transform(
                tile, spatial, tile_raw_shape=self.tile_raw_shape)
            if arr.ndim >= 3:
                scale_kw = (1.0, sy, sx); trans_kw = (0.0, ty, tx)
            else:
                scale_kw = (sy, sx); trans_kw = (ty, tx)
        # Remove previous overlay if present
        self.unload()
        try:
            kwargs = dict(name=f"tile {tile_id} video",
                          colormap=colormap, opacity=opacity,
                          contrast_limits=(contrast_lo, contrast_hi),
                          blending='translucent')
            if use_affine:
                # Mimic the plexusflow viewer's pattern exactly: set the
                # affine POST add_image with scale/translate zeroed. Some
                # napari versions apply the `affine=` kwarg alongside
                # non-identity defaults for scale/translate, giving a
                # subtly wrong composite transform.
                self._current_layer = self.viewer.add_image(arr, **kwargs)
                try:
                    self._current_layer.scale = tuple([1.0] * arr.ndim)
                    self._current_layer.translate = tuple([0.0] * arr.ndim)
                    self._current_layer.affine = affine44
                except Exception as e:
                    print(f"[net3 edit] setting affine post-add_image failed: {e}")
            else:
                kwargs['scale'] = scale_kw
                kwargs['translate'] = trans_kw
                self._current_layer = self.viewer.add_image(arr, **kwargs)
        except Exception as e:
            print(f"[net3 edit] add_image failed: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            return False
        self._current_tile = tile_id
        # Dim the mask layer (if any) so the video is visible against a
        # binary 0/255 mask. Original opacity is captured on first dim
        # and restored on unload.
        try:
            for lyr in self.viewer.layers:
                if lyr.name == "mask":
                    if getattr(self, '_mask_orig_opacity', None) is None:
                        self._mask_orig_opacity = float(lyr.opacity)
                    lyr.opacity = float(dim_mask_to)
                    break
        except Exception as e:
            print(f"[net3 edit] mask dim skipped: {e}")
        # Place the video JUST ABOVE the mask, but BELOW every graph
        # layer (edges/nodes/labels/etc.) so those stay visible on top.
        # Napari layer order: index 0 = bottom, -1 = top.
        try:
            idx = self.viewer.layers.index(self._current_layer)
            # find the mask's index (fallback to 0 if not found)
            mask_idx = 0
            for i, lyr in enumerate(self.viewer.layers):
                if lyr.name == "mask":
                    mask_idx = i; break
            # Target index = mask_idx + 1 (right above the mask). If the
            # video's already there, no-op.
            target = mask_idx + 1
            if idx != target:
                # `move` uses source index (after remove) → clamp target
                self.viewer.layers.move(idx, target)
        except Exception:
            pass
        # Compute the tile footprint in mosaic coords for camera-zoom.
        h, w = spatial
        if use_affine:
            corners = _np.array([[0, 0], [h, 0], [0, w], [h, w]], dtype=float)
            world = (M @ corners.T).T + t_yx
            r0, r1 = float(world[:, 0].min()), float(world[:, 0].max())
            c0, c1 = float(world[:, 1].min()), float(world[:, 1].max())
        else:
            r0, c0 = ty, tx
            r1 = r0 + h * sy
            c1 = c0 + w * sx
        try:
            self.viewer.camera.center = (0.5 * (r0 + r1), 0.5 * (c0 + c1))
            viewport_span = max(r1 - r0, c1 - c0) * 1.4
            self.viewer.camera.zoom = 800.0 / max(viewport_span, 1.0)
        except Exception as e:
            print(f"[net3 edit] camera recenter skipped: {e}")
        print(f"[net3 edit] tile {tile_id} loaded: "
              f"shape={arr.shape}, dtype={arr.dtype}, "
              f"placed at row=[{r0:.0f},{r1:.0f}] col=[{c0:.0f},{c1:.0f}]"
              f"{'  (affine w/ rotation)' if use_affine else ''}")
        return True

    def _tile_footprint(self, tile: dict) -> Optional["np.ndarray"]:
        """Return the four (y, x) corners of a tile in mosaic coords,
        y-flipped by `self.image_h` if set (so outlines match the graph)."""
        import numpy as _np
        H_raw, W_raw = (self.tile_raw_shape if self.tile_raw_shape
                         else (1.0, 1.0))
        corners = _np.array([[0, 0], [0, W_raw],
                              [H_raw, W_raw], [H_raw, 0]], dtype=float)
        A = tile.get('affine_matrix')
        if A is not None:
            M = A[:2, :2]; t = A[:2, 2]
            world = corners @ M.T + t
        else:
            sx = float(tile.get('scale_x', 1.0))
            sy = float(tile.get('scale_y', 1.0))
            tx = float(tile['top_left_x']); ty = float(tile['top_left_y'])
            M = _np.diag([sy, sx])
            world = corners @ M.T + _np.array([ty, tx])
        if self.image_h is not None:
            world = world.copy()
            world[:, 0] = float(self.image_h) - world[:, 0]
        # Shift by the fuse-pad so outlines land inside the mask canvas.
        world = world + _np.array(self._pad_yx)
        return world

    def set_show_tile_boundaries(self, show: bool) -> None:
        """Toggle a yellow-outline polygon per tile at its mosaic footprint."""
        if self._boundary_layer is not None:
            try: self.viewer.layers.remove(self._boundary_layer)
            except Exception: pass
            self._boundary_layer = None
        if not show or not self.tile_positions:
            return
        rects = []
        for tid in sorted(self.tile_positions):
            fp = self._tile_footprint(self.tile_positions[tid])
            if fp is not None: rects.append(fp)
        if not rects: return
        self._boundary_layer = self.viewer.add_shapes(
            rects, shape_type='polygon',
            edge_color='yellow', edge_width=2.0,
            face_color='transparent',
            name='Tile boundaries', opacity=0.7)

    def set_show_tile_numbers(self, show: bool) -> None:
        """Toggle a text label at the centre of each tile with its id."""
        import numpy as _np
        if self._label_layer is not None:
            try: self.viewer.layers.remove(self._label_layer)
            except Exception: pass
            self._label_layer = None
        if not show or not self.tile_positions:
            return
        pts, labels = [], []
        for tid in sorted(self.tile_positions):
            fp = self._tile_footprint(self.tile_positions[tid])
            if fp is None: continue
            c = fp.mean(axis=0)
            pts.append([float(c[0]), float(c[1])])
            labels.append(str(tid))
        if not pts: return
        self._label_layer = self.viewer.add_points(
            _np.array(pts), size=0, name='Tile numbers',
            text={'string': labels, 'size': 14,
                  'color': 'yellow', 'anchor': 'center'})

    def unload(self):
        """Remove the current video overlay, if any."""
        if self._current_layer is not None:
            try:
                self.viewer.layers.remove(self._current_layer)
            except (ValueError, KeyError):
                pass
            self._current_layer = None
            self._current_tile = None
        # Restore the mask's original opacity, if we dimmed it.
        try:
            orig = getattr(self, '_mask_orig_opacity', None)
            if orig is not None:
                for lyr in self.viewer.layers:
                    if lyr.name == "mask":
                        lyr.opacity = float(orig); break
                self._mask_orig_opacity = None
        except Exception:
            pass

    def available_tile_ids(self) -> List[int]:
        return sorted(self.tile_positions.keys())

    @property
    def current_tile(self) -> Optional[int]:
        return self._current_tile
