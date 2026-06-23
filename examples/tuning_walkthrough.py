"""
Tuning walkthrough: synthesise three masks (clean, noisy, gapped), show
how default-flag vectorisation handles each, and demonstrate which knobs
fix each failure mode.

Run from the repo root:
    python examples/tuning_walkthrough.py

Produces three PNG side-by-side comparisons in the current directory:
    tuning_01_clean_baseline.png      — baseline: clean mask, defaults
    tuning_02_noisy_spurs.png         — noisy mask, --prune-dangling sweep
    tuning_03_gapped_breaks.png       — gapped mask: defaults vs --bridge-gaps

Each panel shows the input mask overlaid with the recovered graph
(nodes red, edges orange).  Failure mode → flag-to-touch.

A note on `--prune-order`: PerTileFlow's default of 5 is tuned for
masks with thousands of triangles (real microscopy).  On small synthetic
masks like the ones below, even `--prune-order 2` removes the whole
topology.  This walkthrough uses `--prune-order 0` throughout — the
right default for small masks.  For real data start at 3-5 and bump up
only if you see spurs that --prune-dangling alone can't remove.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from net3 import vectorize


# ── Mask generators ────────────────────────────────────────────────────

def draw_tcross(shape=(400, 600), thickness=8):
    """Three-armed "Y" with sinusoidal centerlines (no straight edges,
    no 90° corners).  Delaunay-based vectorisation is degenerate on
    axis-aligned rectangles — perfectly collinear boundary points and
    right-angle corners give too few triangles per branch, so the
    visualised graph ends up with long straight-line edges that cut
    diagonally through the mask.  Real vessel masks have wavy
    boundaries that naturally produce dense triangulation; this helper
    mimics that with a low-frequency sinusoid on each arm's centerline.

    Topology: one central junction + three tips."""
    H, W = shape
    img = np.zeros(shape, dtype=np.uint8)
    cy, cx = H // 2, W // 2
    r = thickness // 2
    # Three arms radiating from (cy, cx) at 120° intervals, each with a
    # small sinusoidal wobble on the centerline so the constrained
    # Delaunay finds plenty of triangles along the arm.
    # Rotated 15° off-axis so no arm is purely vertical/horizontal —
    # the Delaunay degeneracy on axis-aligned boundaries makes the
    # vectorisation collapse otherwise.  Wobble amplitude > thickness/2
    # so the wave clearly perturbs the boundary, not just the centerline.
    import cv2
    rng = np.random.default_rng(7)
    tilt = np.pi / 12
    angles = [np.pi / 2 + tilt,
              np.pi / 2 + 2 * np.pi / 3 + tilt,
              np.pi / 2 + 4 * np.pi / 3 + tilt]
    arm_len = int(min(H, W) * 0.42)
    # Draw each arm as a thick polyline through ~12 randomly-perturbed
    # control points.  Each segment between control points has a slight
    # bend, so no portion of the arm is locally straight at the
    # triangulation's resolution.  This forces dense triangulation
    # along every arm, regardless of axis orientation.
    n_ctrl = 12
    jitter_amp = 6.0
    for theta in angles:
        s_pts = np.linspace(0, arm_len, n_ctrl)
        # Per-control-point perpendicular jitter.
        jitter = rng.normal(0, 1, n_ctrl) * jitter_amp
        jitter[0] = 0.0          # anchor at the junction
        perp = np.array([-np.sin(theta), np.cos(theta)])
        pts = []
        for s_i, j_i in zip(s_pts, jitter):
            y_i = cy - s_i * np.sin(theta) + j_i * perp[0]
            x_i = cx + s_i * np.cos(theta) + j_i * perp[1]
            pts.append([int(x_i), int(y_i)])  # cv2 uses (x, y)
        pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=False,
                       color=1, thickness=thickness,
                       lineType=cv2.LINE_AA)
    return (img > 0).astype(np.uint8) * 255


def add_salt_noise(mask, n_specks=60, speck_radius=2, seed=0):
    """Sprinkle small foreground specks — segmentation noise."""
    rng = np.random.default_rng(seed)
    H, W = mask.shape
    out = mask.copy()
    for _ in range(n_specks):
        cy = rng.integers(speck_radius, H - speck_radius)
        cx = rng.integers(speck_radius, W - speck_radius)
        out[cy - speck_radius:cy + speck_radius + 1,
            cx - speck_radius:cx + speck_radius + 1] = 255
    return out


def add_attached_specks(mask, n_specks=10, speck_radius=8, seed=1):
    """Glue medium specks against the existing foreground.

    Picks foreground-boundary pixels at random and places a speck just
    outside the boundary, touching it.  These specks ARE big enough to
    survive `--min-feature-size 200`, so they become real spurs in the
    graph — exactly the failure mode `--prune-dangling` is designed to
    fix.
    """
    from scipy.ndimage import binary_dilation
    rng = np.random.default_rng(seed)
    H, W = mask.shape
    out = mask.copy()
    fg = mask > 0
    # Boundary candidates: dilated FG minus FG itself = a 1-px ring
    # just outside the existing structure.
    boundary = binary_dilation(fg, iterations=speck_radius + 2) & (~fg)
    ys, xs = np.where(boundary)
    if len(ys) == 0:
        return out
    for _ in range(n_specks):
        idx = rng.integers(0, len(ys))
        cy, cx = ys[idx], xs[idx]
        out[max(0, cy - speck_radius):cy + speck_radius + 1,
            max(0, cx - speck_radius):cx + speck_radius + 1] = 255
    return out


def add_gap(mask, center=None, half_size=14):
    """Knock out a small square along one arm — broken segmentation."""
    H, W = mask.shape
    if center is None:
        # Punch a hole partway down the "up" arm (the straight-vertical
        # branch in the default Y geometry).
        center = (int(H * 0.32), W // 2)
    cy, cx = center
    out = mask.copy()
    out[max(0, cy - half_size):cy + half_size + 1,
        max(0, cx - half_size):cx + half_size + 1] = 0
    return out


# ── Render helper ──────────────────────────────────────────────────────

def render_panel(ax, mask, G, title):
    """imshow(mask) + nx overlay; node coords are (x, image_height − y)."""
    H = mask.shape[0]
    ax.imshow(mask, cmap='gray', origin='upper')
    if G.number_of_nodes() > 0:
        pos = {n: (d['x'], H - d['y']) for n, d in G.nodes(data=True)}
        nx.draw_networkx_edges(G, pos, ax=ax,
                                edge_color='orange', width=1.6)
        nx.draw_networkx_nodes(G, pos, ax=ax,
                                node_color='red', node_size=22)
    ax.set_title(f'{title}\n'
                  f'{G.number_of_nodes()} nodes, '
                  f'{G.number_of_edges()} edges',
                  fontsize=10)
    ax.set_axis_off()


def save_mask(mask, path):
    Image.fromarray(mask).save(path)


def vec(path, **kw):
    """Vectorise quietly with small-mask defaults.

    Uses `remove_redundant='none'` so every intermediate triangle-center
    node is preserved.  That keeps the orange-edge polyline visually
    tracing the white vessel pixels — the abstract straight-line edges
    that come out of `remove_redundant='all'` are mathematically correct
    but visually misleading on synthetic shapes with right-angle
    branches.  For real data, the default `remove_redundant='all'` is
    fine if you only care about topology.
    """
    kw.setdefault('min_feature_size', 200)
    kw.setdefault('prune_order', 0)
    kw.setdefault('remove_redundant', 'none')
    kw.setdefault('verbose', False)
    return vectorize(str(path), **kw)


# ── Walkthrough panels ────────────────────────────────────────────────

def panel_01_clean(tmp):
    """Baseline: clean wavy Y with default flags."""
    m = draw_tcross()
    p = tmp / 'mask_clean.png'; save_mask(m, p)
    G = vec(p)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    render_panel(ax, m, G,
                  'Clean wavy Y · defaults\n'
                  '(--min-feature-size 200, --prune-order 0)')
    fig.tight_layout()
    fig.savefig('tuning_01_clean_baseline.png', dpi=130)
    plt.close(fig)
    print(f"  panel 1: tuning_01_clean_baseline.png  "
          f"({G.number_of_nodes()}n / {G.number_of_edges()}e)")


def panel_02_noisy(tmp):
    """Noisy mask, sweep --prune-dangling-min-length."""
    m = add_attached_specks(draw_tcross(), n_specks=12, speck_radius=7,
                              seed=2)
    p = tmp / 'mask_noisy.png'; save_mask(m, p)
    sweep = [
        (False, 0,  'defaults\n→ attached specks become real spurs'),
        (True,  40, '+ --prune-dangling-min-length 40\n'
                    '→ short spurs gone, real branches kept'),
        (True,  300, '+ --prune-dangling-min-length 300\n'
                     '→ too aggressive, eats real branches'),
    ]
    fig, axes = plt.subplots(1, len(sweep), figsize=(6 * len(sweep), 5.5))
    counts = []
    for ax, (use_prune, ml, title) in zip(axes, sweep):
        G = vec(p, prune_dangling=use_prune, prune_dangling_min_length=ml)
        render_panel(ax, m, G, title)
        counts.append((G.number_of_nodes(), G.number_of_edges()))
    fig.suptitle('--prune-dangling-min-length is image-scale dependent: '
                  'set it above the longest noise spur, '
                  'well below the shortest real branch.',
                  fontsize=11)
    fig.tight_layout()
    fig.savefig('tuning_02_noisy_spurs.png', dpi=130)
    plt.close(fig)
    print(f"  panel 2: tuning_02_noisy_spurs.png  "
          f"(sweep " + " → ".join(f"{n}n/{e}e" for n, e in counts) + ")")


def panel_03_gapped(tmp):
    """Gapped trunk → disconnected → --bridge-gaps reconnects."""
    m = add_gap(draw_tcross())
    p = tmp / 'mask_gapped.png'; save_mask(m, p)
    G_default = vec(p)
    G_bridged = vec(p, bridge_gaps=22)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    render_panel(axes[0], m, G_default,
                  'Gapped mask · defaults\n'
                  '→ arm broken at the gap, smaller half '
                  'dropped by keep_largest_component')
    render_panel(axes[1], m, G_bridged,
                  '+ --bridge-gaps 22\n'
                  '→ morphological closing reconnects the arm')
    fig.suptitle('Fix: --bridge-gaps closes gaps BEFORE small-object '
                  'filtering so the segments stay connected.',
                  fontsize=11)
    fig.tight_layout()
    fig.savefig('tuning_03_gapped_breaks.png', dpi=130)
    plt.close(fig)
    print(f"  panel 3: tuning_03_gapped_breaks.png  "
          f"(default {G_default.number_of_nodes()}n/"
          f"{G_default.number_of_edges()}e  →  "
          f"bridged {G_bridged.number_of_nodes()}n/"
          f"{G_bridged.number_of_edges()}e)")


def panel_04_min_feature_sweep(tmp):
    """Sweep --min-feature-size on a noisy mask."""
    m = add_salt_noise(draw_tcross(), n_specks=200, speck_radius=4)
    p = tmp / 'mask_clean2.png'; save_mask(m, p)
    sweep = [20, 200, 1500, 8000]
    fig, axes = plt.subplots(1, len(sweep), figsize=(5 * len(sweep), 5.5))
    for ax, mfs in zip(axes, sweep):
        try:
            G = vec(p, min_feature_size=mfs)
            render_panel(ax, m, G, f'--min-feature-size {mfs}')
        except Exception as e:
            ax.imshow(m, cmap='gray', origin='upper')
            ax.text(0.5, 0.5,
                     f'--min-feature-size {mfs}\nfailed:\n{type(e).__name__}',
                     transform=ax.transAxes, ha='center', va='center',
                     color='red', fontsize=10)
            ax.set_axis_off()
    fig.suptitle('Sweep: --min-feature-size on a noisy wavy Y\n'
                  '(too small → noise creeps in;  too large → real '
                  'features get dropped)',
                  fontsize=11)
    fig.tight_layout()
    fig.savefig('tuning_04_min_feature_sweep.png', dpi=130)
    plt.close(fig)
    print(f"  panel 4: tuning_04_min_feature_sweep.png")


def main():
    tmp = Path('.')
    print("Tuning walkthrough — generating panels...")
    panel_01_clean(tmp)
    panel_02_noisy(tmp)
    panel_03_gapped(tmp)
    print("Done.")


if __name__ == "__main__":
    main()
