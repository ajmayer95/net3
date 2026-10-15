"""Backend tests for MaskEditor — pure ndarray operations, no GUI."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from net3.edit import MaskEditor


# ── Fixtures ──────────────────────────────────────────────────────


def _toy_mask() -> np.ndarray:
    """20×20 mask: a 5-wide vertical stripe + a 3-pixel speck.

    Foreground layout:
        cols 5..9 across rows 2..17 → 5×16 = 80 vessel pixels
        rows 15..17, cols 15..17    → 3×3  = 9 speck pixels

    Total foreground: 89 pixels.
    """
    m = np.zeros((20, 20), dtype=bool)
    m[2:18, 5:10] = True
    m[15:18, 15:18] = True
    return m


def _toy_probs() -> np.ndarray:
    """20×20 float in [0, 1] shaped like the toy mask (blurred edges).

    Peaks at the stripe (0.9) and the speck (0.6), background 0.05.
    """
    p = np.full((20, 20), 0.05, dtype=np.float32)
    p[2:18, 5:10] = 0.9
    p[15:18, 15:18] = 0.6
    return p


# ── Construction ─────────────────────────────────────────────────


def test_construct_from_bool_mask():
    m = _toy_mask()
    me = MaskEditor(m)
    assert me.mask.dtype == np.bool_
    assert me.n_foreground == 89
    assert me.image is None
    assert me.probs is None
    assert me.locked is False


def test_construct_from_uint8_mask_coerces_to_bool():
    m = _toy_mask().astype(np.uint8) * 255
    me = MaskEditor(m)
    assert me.mask.dtype == np.bool_
    assert me.n_foreground == 89


def test_construct_rejects_non_2d():
    with pytest.raises(ValueError):
        MaskEditor(np.zeros((3, 20, 20), dtype=bool))


def test_construct_with_probs_shape_check():
    m = _toy_mask()
    with pytest.raises(ValueError):
        MaskEditor(m, probs=np.zeros((10, 10), dtype=np.float32))


def test_construct_with_probs_non_2d():
    m = _toy_mask()
    with pytest.raises(ValueError):
        MaskEditor(m, probs=np.zeros((3, 20, 20), dtype=np.float32))


# ── Threshold + cleanup ──────────────────────────────────────────


def test_threshold_no_probs_returns_unchanged():
    me = MaskEditor(_toy_mask())
    m0 = me.mask.copy()
    me.threshold(0.5)
    np.testing.assert_array_equal(me.mask, m0)


def test_threshold_basic():
    me = MaskEditor(_toy_mask(), probs=_toy_probs())
    me.threshold(0.5)
    # At t=0.5, stripe (0.9) and speck (0.6) both survive
    assert me.n_foreground == 89


def test_threshold_high_cutoff_drops_speck():
    me = MaskEditor(_toy_mask(), probs=_toy_probs())
    me.threshold(0.7)
    # At t=0.7, only stripe (0.9) survives; speck (0.6) drops
    assert me.n_foreground == 80


def test_threshold_remove_small_drops_speck_at_low_cutoff():
    me = MaskEditor(_toy_mask(), probs=_toy_probs())
    me.threshold(0.5, remove_small=20)
    # Speck is only 9 pixels < 20 → dropped
    assert me.n_foreground == 80


def test_threshold_when_locked_is_noop():
    me = MaskEditor(_toy_mask(), probs=_toy_probs())
    me.locked = True
    m0 = me.mask.copy()
    me.threshold(0.7)
    np.testing.assert_array_equal(me.mask, m0)


def test_fill_holes():
    m = np.zeros((20, 20), dtype=bool)
    m[2:18, 2:18] = True
    m[9:12, 9:12] = False    # 3×3 hole
    me = MaskEditor(m)
    n_before = me.n_foreground
    me.fill_small_holes(area_threshold=20)
    assert me.n_foreground == n_before + 9


def test_remove_small_objects():
    me = MaskEditor(_toy_mask())
    me.remove_small_objects(min_size=20)
    # Speck (9 px) dropped, stripe (80 px) kept
    assert me.n_foreground == 80


def test_median_filter_smooths_and_locks():
    me = MaskEditor(_toy_mask())
    assert me.locked is False
    me.apply_median_filter(radius=1)
    assert me.locked is True
    # A stripe minus corner-smoothed speck should have fewer or equal fg pixels
    assert me.n_foreground <= 89


# ── Undo ─────────────────────────────────────────────────────────


def test_undo_restores_previous_mask():
    me = MaskEditor(_toy_mask())
    m0 = me.mask.copy()
    me.remove_small_objects(min_size=20)
    assert me.n_foreground == 80
    assert me.undo() is True
    np.testing.assert_array_equal(me.mask, m0)


def test_undo_when_empty_returns_false():
    me = MaskEditor(_toy_mask())
    assert me.undo() is False


def test_undo_chain():
    me = MaskEditor(_toy_mask(), probs=_toy_probs())
    m0 = me.mask.copy()
    me.threshold(0.5)                          # step 1
    m1 = me.mask.copy()
    me.remove_small_objects(min_size=20)       # step 2
    assert me.undo() is True                   # → back to step 1
    np.testing.assert_array_equal(me.mask, m1)
    assert me.undo() is True                   # → back to step 0
    np.testing.assert_array_equal(me.mask, m0)


def test_undo_stack_capped():
    me = MaskEditor(_toy_mask())
    for _ in range(me.MAX_UNDO_DEPTH + 5):
        me.push_undo()
    assert len(me._undo) == me.MAX_UNDO_DEPTH


# ── Reset from probabilities ─────────────────────────────────────


def test_reset_unlocks_and_regenerates():
    me = MaskEditor(_toy_mask(), probs=_toy_probs())
    me.apply_median_filter(radius=1)
    assert me.locked is True
    me.reset_from_probabilities(t=0.5)
    assert me.locked is False
    assert me.n_foreground == 89


def test_reset_no_probs_returns_unchanged():
    me = MaskEditor(_toy_mask())
    m0 = me.mask.copy()
    me.reset_from_probabilities(t=0.5)
    np.testing.assert_array_equal(me.mask, m0)


# ── Set pixels (programmatic paint) ──────────────────────────────


def test_set_pixels_paints_and_undoable():
    me = MaskEditor(np.zeros((10, 10), dtype=bool))
    rr, cc = np.mgrid[3:6, 3:6]
    me.set_pixels(rr.ravel(), cc.ravel(), value=True)
    assert me.n_foreground == 9
    assert me.undo()
    assert me.n_foreground == 0


# ── I/O ──────────────────────────────────────────────────────────


def test_save_and_reload(tmp_path: Path):
    me = MaskEditor(_toy_mask())
    out = me.save_mask(tmp_path / "mask.tif")
    assert out.exists()
    reloaded = tifffile.imread(out)
    assert reloaded.dtype == np.uint8
    assert reloaded.max() == 255
    assert (reloaded > 0).sum() == 89


def test_from_paths_reads_mask(tmp_path: Path):
    m = _toy_mask().astype(np.uint8) * 255
    p = tmp_path / "mask.tif"
    tifffile.imwrite(str(p), m)
    me = MaskEditor.from_paths(p)
    assert me.mask.shape == (20, 20)
    assert me.n_foreground == 89


def test_from_paths_reads_image_and_probs(tmp_path: Path):
    m = _toy_mask().astype(np.uint8) * 255
    img = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
    probs_uint8 = (_toy_probs() * 255).astype(np.uint8)
    for name, arr in [("mask.tif", m), ("img.tif", img), ("probs.tif", probs_uint8)]:
        tifffile.imwrite(str(tmp_path / name), arr)
    me = MaskEditor.from_paths(
        tmp_path / "mask.tif",
        image_path=tmp_path / "img.tif",
        probs_path=tmp_path / "probs.tif",
    )
    assert me.image is not None and me.image.shape == (20, 20)
    assert me.probs is not None and me.probs.shape == (20, 20)
    # probs got scaled to [0, 1]
    assert me.probs.max() <= 1.0
    assert me.probs.max() > 0.5


# ── Introspection ────────────────────────────────────────────────


def test_foreground_fraction():
    me = MaskEditor(_toy_mask())
    assert me.foreground_fraction == pytest.approx(89 / 400)
