"""
Quickstart: draw a Y-shaped mask, vectorise it, render the result.

Run from the repo root:
    python examples/quickstart.py
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from net3 import vectorize


def draw_y(shape=(400, 400), radius=8):
    H, W = shape
    img = np.zeros(shape, dtype=np.uint8)
    cy, cx = H // 2, W // 2
    angles = [np.pi / 2, np.pi / 2 + 2 * np.pi / 3,
              np.pi / 2 + 4 * np.pi / 3]
    arm_len = min(H, W) // 3
    for theta in angles:
        ey = int(cy - arm_len * np.sin(theta))
        ex = int(cx + arm_len * np.cos(theta))
        n = max(abs(ey - cy), abs(ex - cx)) + 1
        ys = np.linspace(cy, ey, n).astype(int)
        xs = np.linspace(cx, ex, n).astype(int)
        for y, x in zip(ys, xs):
            img[max(0, y - radius):y + radius + 1,
                max(0, x - radius):x + radius + 1] = 1
    return img * 255


def main():
    mask = draw_y()
    Image.fromarray(mask).save("example_mask.png")

    G = vectorize("example_mask.png",
                  min_feature_size=200,
                  prune_order=2)

    print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask, cmap="gray", origin="upper")
    # net3 returns y in mathematical convention (origin bottom); flip
    # back to image-row coords for overlay.
    H = mask.shape[0]
    pos = {n: (d["x"], H - d["y"]) for n, d in G.nodes(data=True)}
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="orange", width=1.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="red",
                            node_size=20)
    ax.set_title("net3: synthetic Y mask → graph")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("example_overlay.png", dpi=130)
    print("Wrote example_overlay.png")


if __name__ == "__main__":
    main()
