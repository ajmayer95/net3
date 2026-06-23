"""Smoke test: synthetic mask round-trips through vectorize() and produces
a non-empty graph with the expected attribute schema."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from net3 import vectorize, save_graph, load_graph


def _draw_y_shape(shape=(400, 400), radius=8):
    """Draw a simple Y-shaped mask: a central junction with three arms."""
    H, W = shape
    img = np.zeros(shape, dtype=np.uint8)
    cy, cx = H // 2, W // 2
    # Three arm endpoints arranged at 120° intervals around the center.
    angles = [np.pi / 2, np.pi / 2 + 2 * np.pi / 3,
              np.pi / 2 + 4 * np.pi / 3]
    arm_len = min(H, W) // 3
    for theta in angles:
        ey = int(cy - arm_len * np.sin(theta))
        ex = int(cx + arm_len * np.cos(theta))
        # Bresenham-ish: rasterise a thick line by interpolating.
        n = max(abs(ey - cy), abs(ex - cx)) + 1
        ys = np.linspace(cy, ey, n).astype(int)
        xs = np.linspace(cx, ex, n).astype(int)
        for y, x in zip(ys, xs):
            yl = max(0, y - radius); yh = min(H, y + radius + 1)
            xl = max(0, x - radius); xh = min(W, x + radius + 1)
            img[yl:yh, xl:xh] = 1
    return img * 255


def test_vectorize_synthetic_y(tmp_path):
    mask_path = tmp_path / "y.png"
    Image.fromarray(_draw_y_shape()).save(mask_path)

    G = vectorize(
        str(mask_path),
        min_feature_size=200,    # small synthetic features
        prune_order=2,
        verbose=False,
    )

    assert G.number_of_nodes() > 0, "graph has no nodes"
    assert G.number_of_edges() > 0, "graph has no edges"
    # Y-shape has one junction + three tips; allow some slack from the
    # triangulation, but we should not be ending up with hundreds of
    # disconnected pieces or zero edges.
    assert G.number_of_edges() < 500

    # Schema spot checks.
    any_node = next(iter(G.nodes(data=True)))[1]
    for key in ("x", "y", "radius"):
        assert key in any_node, f"node missing '{key}'"
    any_edge = next(iter(G.edges(data=True)))[2]
    for key in ("radius", "weight"):
        assert key in any_edge, f"edge missing '{key}'"

    # Save / load round-trip.
    out = tmp_path / "g.gpickle"
    save_graph(G, str(out))
    G2 = load_graph(str(out))
    assert G2.number_of_nodes() == G.number_of_nodes()
    assert G2.number_of_edges() == G.number_of_edges()
