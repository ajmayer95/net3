# net3

Binary mask → NetworkX graph. Given a segmentation of a tubular network
(vessels, neurons, mycelia, anything one-pixel-thick after the medial
axis), `net3` extracts a graph with node coordinates, edge lengths, and
per-edge radii in pixels.

## Install

```bash
pip install -e .
```

`pip install` automatically builds the bundled Cython extension
(`C_net_functions`) — that's why `Cython` and `numpy` are listed under
`[build-system].requires` in `pyproject.toml`. The build needs a working
C compiler: on macOS `xcode-select --install` covers it, on Linux a
standard `gcc` package, on Windows the MSVC Build Tools.

If you're working in-place without a real install (e.g. on an HPC node
where `pip install -e .` is awkward), you can build just the extension:

```bash
python setup.py build_ext --inplace
PYTHONPATH=src python -c "import net3; print(net3.__version__)"
```

Other dependency worth flagging: `meshpy` wraps J. R. Shewchuk's Triangle
library. macOS / Linux ship a wheel; Windows users may need a pre-built
wheel from a third party.

## Use — Python

```python
from net3 import vectorize, save_graph

G = vectorize("mask.tif", min_feature_size=300)
print(G.number_of_nodes(), G.number_of_edges())

# Each node has (x, y, radius); each edge has (weight=length, radius).
for u, v, data in G.edges(data=True):
    print(u, v, data["radius"], data["weight"])

save_graph(G, "graph.gpickle")
```

For neuron data the relevant tweak is usually a smaller
`min_feature_size` (the default 3000 is tuned for ~3000 px² yolk-sac
vessels) — set it to the smallest connected component you actually
want to keep.

## Use — CLI

```bash
net3 mask.tif -o graph.gpickle
net3 mask.png -o graph.graphml --format graphml --invert
net3 binary_neuron.tif -o neuron_graph.gpickle --min-feature-size 200
```

`net3 --help` lists every flag.

## What it does

```
mask → distance transform → contours → constrained Delaunay
     → triangle classification (branch / junction / sleeve)
     → prune short branch triangles
     → adjacency → NetworkX graph
     → collapse degree-2 chains
     → keep largest connected component
```

Output schema:

| key | where | unit | meaning |
|---|---|---|---|
| `x`, `y` | node attr | pixels | image coords (y is flipped to mathematical convention) |
| `radius` | node attr | pixels | distance transform at that triangle's center |
| `radius` | edge attr | pixels | mean of endpoint radii |
| `weight` | edge attr | pixels | Euclidean distance between endpoints |

A separate `collapse_to_vessel_graph(G)` is available if you want one
edge per branch (instead of per inter-junction segment) — useful for
biological vessel-level analysis.

## Tuning the flags

The flags fall into two phases: **mask cleanup** (before vectorisation)
and **graph cleanup** (after). Mask cleanup is applied in this order:
`bridge-gaps → remove small objects → smoothing → distance transform`.

### Mask cleanup

| Flag | Default | What it does | When to change it |
|---|---|---|---|
| `--min-feature-size` | 3000 px | Drops any connected foreground blob smaller than N pixels. | **Always check this first.** It's the only mask-cleanup knob that affects every run. The default is tuned for ~3000 px² yolk-sac vessels at 10× / 1.7 µm/px. For neurons at higher magnification, drop to 100–500. If real features keep getting lost → too high. If noise speckles appear as graph nodes → too low. |
| `--smoothing` | off | Morphological opening then closing with a disk of radius N, then a second small-object filter. Smooths jagged mask boundaries. | Use **only** if the mask has a noisy boundary producing spurious tiny branches. Typical values 1–3. Larger values erode thin vessels — stay below half the vessel radius. |
| `--bridge-gaps` | off | Morphological closing with a disk of radius N, applied **before** the small-object filter. Bridges small gaps between disconnected pieces. | Use when segmentation breaks vessels at boundaries (tile seams, faded segments). Typical 5–15. Larger values can fuse genuinely-separate vessels — start small. |
| `--invert` | off | Swap foreground/background. | Set this if vessels are dark on light. Quick check: open the mask — if vessels are dark, add `--invert`. |

### Graph cleanup

| Flag | Default | What it does | When to change it |
|---|---|---|---|
| `--prune-order` | 5 | Iteratively peels short branch triangles from the triangulation N times before graph construction. Removes tiny spurs from boundary roughness. | Lower (1–3) preserves every twig. Higher (8–15) gives a cleaner "main vessels only" graph. 5 is the default sweet spot. |
| `--remove-redundant` | `all` | Collapses degree-2 nodes (a node with exactly two neighbors) by merging their two edges. Default `all` means *every* degree-2 node is removed, so only junctions (deg ≥ 3) and tips (deg 1) remain. | `none` if you need every intermediate vessel-path point (useful for per-segment curvature). `half` keeps every other — middle ground. For most analyses, `all`. |
| `--prune-dangling` | off | After everything else, removes degree-1 dangling tips. | Use to remove single short edges sticking out of junctions (segmentation noise). Combine with `--prune-dangling-min-length`. |
| `--prune-dangling-min-length` | 0 | Only prune dangling tips **shorter than N pixels**. | Keeps long end-branches (probably real) while removing short spurs. Sensible: a small multiple of the typical vessel radius, e.g. 30–50 px. |

### Recipe book

Clean mask, want most detail (defaults):
```bash
net3 mask.tif -o g.gpickle
```

Noisy segmentation with tile seams (mosaic-style data):
```bash
net3 mask.tif -o g.gpickle \
    --bridge-gaps 10 --smoothing 2 \
    --prune-dangling --prune-dangling-min-length 30
```

Neuron data at higher magnification, smaller features:
```bash
net3 neuron_mask.tif -o g.gpickle \
    --min-feature-size 200 --prune-order 3
```

Skeleton with every intermediate node, no collapse (for per-segment analysis):
```bash
net3 mask.tif -o g.gpickle --remove-redundant none
```

Coarse "main vessels only" graph:
```bash
net3 mask.tif -o g.gpickle \
    --min-feature-size 8000 --prune-order 12 \
    --prune-dangling --prune-dangling-min-length 80
```

### How to tune in practice

There's no auto-tuning. Iterate:

1. Run with **defaults first**.
2. Quick-visualise the result: `imshow(mask)` + `nx.draw_networkx_edges(G, pos)` (the pattern in [`examples/quickstart.py`](examples/quickstart.py)).
3. Diagnose by failure mode:
   - **Spurious tiny edges everywhere** → bump `--prune-order` or add `--prune-dangling`.
   - **Vessels broken into pieces** → add `--bridge-gaps 10`, or check whether `--invert` is needed.
   - **Real vessels missing** → `--min-feature-size` is too high, or `--smoothing` is eroding them.

Iterative visualisation is dramatically faster than guessing every flag right on the first run.

[`examples/tuning_walkthrough.py`](examples/tuning_walkthrough.py) generates four side-by-side PNG comparisons that walk you through each failure mode visually — clean baseline, spurs from segmentation noise, broken trunk reconnected via `--bridge-gaps`, and a `--min-feature-size` sweep. Run it from the repo root after install to produce reference imagery you can compare your own tuning attempts against.

### A gotcha worth knowing

`--prune-order` is *much* more aggressive than the help text suggests. It operates on triangles, not graph edges, and iteratively peels them off **before** the graph is built. On small synthetic masks (anything under a few thousand triangles) even `--prune-order 2` can wipe out the entire topology. The default 5 is tuned for full-mosaic vessel masks with tens of thousands of triangles.

Rule of thumb: for small or moderate masks, start with `--prune-order 0` and bump up only if you see spurs that `--prune-dangling --prune-dangling-min-length N` alone can't remove. For real microscopy data at the scale that PerTileFlow ships, 3–5 is fine.

## Interactive editor

For cleaning up false junctions, broken arms, and other vectorisation
artifacts by hand, net3 ships a napari-based editor. Modernised port of
[Jana Lasser's `gegui`](https://github.com/JanaLasser/network_extraction)
(GPL-3, 2015, Python 2 / NetworkX 1.x) with a separable GUI-free
backend.

```bash
pip install 'net3[gui]'            # adds napari + qtpy + PyQt5
net3 edit graph.gpickle --mask mask.tif
```

Optional flags:

- `--distance-map dm.png` — sample radius for newly-created nodes from
  a precomputed distance transform (cv2.distanceTransform).
- `--save out.gpickle` — Save target on `s` / button; defaults to
  overwriting the input.

### Recommended workflow

`net3 vectorize` defaults to `--remove-redundant all`, which gives a
clean topology graph (every node is a tip or junction). That's the
right form for downstream analysis, but it's visually confusing in the
editor — edges become straight lines connecting tips/junctions and cut
across the mask instead of tracing it.

For the editor, vectorise with **`--for-editor`** (shorthand for
`--remove-redundant none`). The polyline of short edges then traces
the mask centerline closely:

```bash
# 1. Vectorise in editor-friendly form (keeps every intermediate node)
net3 vectorize mask.tif -o graph_edit.gpickle --for-editor

# 2. Edit interactively (clean up false junctions, broken arms, ...)
net3 edit graph_edit.gpickle --mask mask.tif --save graph_edit.gpickle

# 3. Collapse to topological form for analysis: press `n` in the editor
#    before saving, OR run a second vectorise pass without --for-editor.
```

If you launch the editor on a graph that's already been streamlined,
it'll print a hint with the right command to re-vectorise.

### Controls

| Action | Key | Mouse |
|---|---|---|
| Select / deselect node | — | Click near a node |
| Create node | — | Shift + click on empty canvas |
| Delete selected | `d` | — |
| Connect two selected | `e` | — |
| Highlight cycles | `m` | — |
| Streamline (collapse degree-2) | `n` | — |
| Clear selection | `c` | — |
| Undo last edit | `z` | — |
| Save | `s` | — |

Node colors: **green** = degree-1 tip, **cyan** = degree-≥3 junction,
**red** = intermediate, **yellow** = selected.

### Backend API (no GUI)

The editing operations live in `net3.edit.GraphEditor` — a plain
Python class that wraps a NetworkX graph with selection state and an
undo stack. Use it directly in scripts or notebooks:

```python
from net3.edit import GraphEditor

ed = GraphEditor.from_gpickle("graph.gpickle")
ed.select(42)
ed.delete_selected()
ed.add_node(120.5, 80.0)
ed.connect_selected_pair()
n_collapsed = ed.streamline()
ed.undo()                  # restore the last mutation
ed.save("edited.gpickle")
```

All operations have unit tests (`tests/test_edit_core.py`) that pass
without napari installed.

## Caveats

- Radii are distance-transform values at triangle centers. Good first-pass estimate; sub-pixel-accurate diameter requires per-edge refitting downstream.
- `min_feature_size` is in pixels and is the most magnification-sensitive parameter — always reset it for new data.
- The loader auto-thresholds at 127 if the image isn't already 0/1.

## License

MIT — see [LICENSE](LICENSE).
