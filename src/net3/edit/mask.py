"""
MaskEditor — GUI-free backend for interactive mask editing.

Holds a mutable binary mask plus an optional reference image (2-D
grayscale) and optional probability map (2-D float in [0, 1]).  Exposes
the operations the napari frontend wires buttons/keys to: threshold
regeneration, small-object removal, hole filling, median smoothing,
undo, and save.

Paint/erase live in the napari frontend — those directly mutate the
labels layer's ndarray, which is `self.mask`.  Between paint strokes,
push a snapshot with `push_undo()` if you want it undoable.

Everything here is testable without a GUI — see
tests/test_edit_mask.py.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np


def _load_image_2d(path: Path) -> np.ndarray:
    """Load a 2-D grayscale image (PNG/TIFF) as uint8."""
    from PIL import Image
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.uint8)


def _load_probs_2d(path: Path) -> np.ndarray:
    """Load a probability map (TIFF/PNG) as float32 in [0, 1].

    HDF5 / ilastik outputs are not handled here — those are
    dataset-specific and should be converted to a plain ndarray by
    the caller.
    """
    from PIL import Image
    im = np.asarray(Image.open(path))
    if im.dtype == np.uint8:
        return im.astype(np.float32) / 255.0
    if im.dtype == np.uint16:
        return im.astype(np.float32) / 65535.0
    return im.astype(np.float32)


def _to_bool_mask(arr: np.ndarray) -> np.ndarray:
    """Coerce an image to a 2-D bool mask."""
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {arr.shape}")
    return arr.astype(bool)


class MaskEditor:
    """Stateful wrapper around a binary mask for interactive editing.

    Parameters
    ----------
    mask : np.ndarray
        2-D binary mask.  Any dtype; nonzero → foreground.  Modified
        in-place by every op; restored from snapshots by ``undo()``.
    image : np.ndarray, optional
        2-D grayscale reference image (same shape as ``mask``).  Not
        used by any backend op; the frontend displays it underneath.
    probs : np.ndarray, optional
        2-D probability map in ``[0, 1]``.  Used by ``threshold()`` and
        ``reset_from_probabilities()``.  Same shape as ``mask``.

    Notes
    -----
    Undo depth is capped at ``MAX_UNDO_DEPTH`` snapshots.  Each op that
    changes ``self.mask`` calls ``push_undo()`` before mutating.
    """

    MAX_UNDO_DEPTH = 20

    def __init__(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
    ):
        self.mask = _to_bool_mask(mask)
        self.image = image
        self.probs = None
        if probs is not None:
            self.set_probs(probs)
        self.locked = False
        self._undo: deque = deque(maxlen=self.MAX_UNDO_DEPTH)

    # ── Loading ────────────────────────────────────────────────────

    @classmethod
    def from_paths(
        cls,
        mask_path: str | Path,
        image_path: Optional[str | Path] = None,
        probs_path: Optional[str | Path] = None,
    ) -> "MaskEditor":
        """Construct from files (PIL-readable formats only).

        For HDF5 / ilastik probability outputs, load the array yourself
        and pass it via ``probs=`` to the normal constructor.
        """
        mask = _load_image_2d(Path(mask_path))
        image = _load_image_2d(Path(image_path)) if image_path else None
        probs = _load_probs_2d(Path(probs_path)) if probs_path else None
        return cls(mask=mask, image=image, probs=probs)

    def set_probs(self, probs: np.ndarray) -> None:
        """Attach or replace the probability map."""
        if probs.ndim != 2:
            raise ValueError(f"probs must be 2-D, got shape {probs.shape}")
        if probs.shape != self.mask.shape:
            raise ValueError(
                f"probs shape {probs.shape} != mask shape {self.mask.shape}"
            )
        self.probs = probs.astype(np.float32, copy=False)

    # ── Undo ──────────────────────────────────────────────────────

    def push_undo(self) -> None:
        """Snapshot the current mask onto the undo stack."""
        self._undo.append(self.mask.copy())

    def undo(self) -> bool:
        """Restore the previous mask.  Returns True if something popped."""
        if not self._undo:
            return False
        self.mask = self._undo.pop()
        return True

    # ── Threshold + cleanup ───────────────────────────────────────

    def threshold(
        self,
        t: float,
        remove_small: int = 0,
        fill_holes: int = 0,
    ) -> np.ndarray:
        """Regenerate the mask from ``self.probs > t`` with cleanup.

        Fails silently (returns unchanged mask) if ``probs`` is None or
        ``self.locked`` is True.

        Parameters
        ----------
        t : float
            Threshold in ``[0, 1]``.
        remove_small : int
            Drop connected components smaller than this many pixels.
            0 disables.
        fill_holes : int
            Fill holes smaller than this area (in pixels).  0 disables.
        """
        if self.probs is None or self.locked:
            return self.mask
        self.push_undo()
        m = self.probs > t
        if remove_small > 0:
            from skimage.morphology import remove_small_objects
            m = remove_small_objects(m, min_size=remove_small, connectivity=2)
        if fill_holes > 0:
            from skimage.morphology import remove_small_holes
            m = remove_small_holes(m, area_threshold=fill_holes, connectivity=2)
        self.mask = m
        return self.mask

    def apply_median_filter(self, radius: int = 3) -> np.ndarray:
        """Smooth the mask with a square median filter (side = 2r+1).

        Auto-locks after the smoothing so a downstream threshold change
        doesn't clobber the smoothing.
        """
        from scipy.ndimage import median_filter
        self.push_undo()
        size = 2 * int(radius) + 1
        self.mask = median_filter(self.mask.astype(np.uint8), size=size).astype(bool)
        self.locked = True
        return self.mask

    def remove_small_objects(self, min_size: int) -> np.ndarray:
        """Drop connected components smaller than ``min_size`` pixels."""
        from skimage.morphology import remove_small_objects
        self.push_undo()
        self.mask = remove_small_objects(self.mask, min_size=int(min_size), connectivity=2)
        return self.mask

    def fill_small_holes(self, area_threshold: int) -> np.ndarray:
        """Fill holes smaller than ``area_threshold`` pixels."""
        from skimage.morphology import remove_small_holes
        self.push_undo()
        self.mask = remove_small_holes(
            self.mask, area_threshold=int(area_threshold), connectivity=2
        )
        return self.mask

    def reset_from_probabilities(
        self,
        t: float = 0.5,
        remove_small: int = 0,
        fill_holes: int = 0,
    ) -> np.ndarray:
        """Unlock and regenerate the mask from probabilities."""
        if self.probs is None:
            return self.mask
        self.locked = False
        return self.threshold(t, remove_small, fill_holes)

    # ── Paint / erase (programmatic) ──────────────────────────────

    def set_pixels(self, rows, cols, value: bool = True) -> None:
        """Programmatically set a set of mask pixels.  Snapshots undo."""
        self.push_undo()
        self.mask[rows, cols] = bool(value)

    # ── I/O ───────────────────────────────────────────────────────

    def save_mask(self, path: str | Path) -> Path:
        """Write the mask as uint8×255 TIFF."""
        import tifffile
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(path), (self.mask.astype(np.uint8) * 255))
        return path

    # ── Introspection ─────────────────────────────────────────────

    @property
    def n_foreground(self) -> int:
        return int(self.mask.sum())

    @property
    def foreground_fraction(self) -> float:
        return float(self.mask.mean())
