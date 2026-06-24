# net3

Binary mask to NetworkX graph, with an interactive napari editor.

## Install

Needs Python 3.9+ and a C compiler (`xcode-select --install` on macOS,
`build-essential` on Debian/Ubuntu, MSVC Build Tools on Windows).

```bash
mkdir -p ~/net3 && cd ~/net3
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install 'net3[gui] @ git+https://github.com/ajmayer95/net3.git'
```

```bash
net3 --version
```

## Try it

```bash
curl -L https://raw.githubusercontent.com/JanaLasser/network_extraction/master/data/binaries/tracheole1_binary.png -o test_mask.png
net3 vectorize test_mask.png -o test_graph.gpickle --min-feature-size 200 --for-editor
net3 edit test_graph.gpickle --mask test_mask.png
```

A napari window opens with the mask in greyscale and an orange polyline
along every branch. Red dots are intermediate nodes, green are tips,
cyan are junctions.

## Two forms of the graph

`net3 vectorize` produces one of two shapes depending on `--for-editor`:

- **Topological** (default): every node is a tip or junction. Edges are
  straight lines between them. Right form for graph analysis. Edges
  cut across the mask in the editor view because intermediate path
  geometry is gone.
- **Centerline** (`--for-editor`): every intermediate node kept.
  Polyline traces the mask.

To get both: vectorise with `--for-editor`, edit, then press `n`
(streamline) in the editor before saving.

## Python

```python
from net3 import vectorize, save_graph

G = vectorize("test_mask.png", min_feature_size=200)
print(G.number_of_nodes(), G.number_of_edges())

for u, v, data in G.edges(data=True):
    print(u, v, data["radius"], data["weight"])

save_graph(G, "test_graph.gpickle")
```

## Output schema

| key | where | unit | meaning |
|---|---|---|---|
| `x`, `y` | node attr | pixels | image coords (y flipped to mathematical convention) |
| `radius` | node and edge attr | pixels | distance-transform value (node) or mean of endpoint radii (edge) |
| `weight` | edge attr | pixels | Euclidean distance between endpoints |

`collapse_to_branch_graph(G)` gives one edge per branch (instead of
per inter-junction segment) with the full path geometry stored as an
edge attribute.

## Tuning the flags

The flags fall into two phases: **mask cleanup** (before vectorisation)
and **graph cleanup** (after). Mask cleanup runs in this order:
`bridge-gaps → remove small objects → smoothing → distance transform`.

### Mask cleanup

| Flag | Default | What it does |
|---|---|---|
| `--min-feature-size` | 3000 px | Drops foreground components smaller than N pixels. The most magnification-sensitive parameter. For finer-scale data, drop to 100-500. |
| `--smoothing` | off | Morphological open then close with a disk of radius N. Smooths jagged boundaries. Typical values 1-3. Larger values erode thin features. |
| `--bridge-gaps` | off | Morphological closing with a disk of radius N, before the small-object filter. Bridges small gaps between disconnected pieces. Typical 5-15. |
| `--invert` | off | Swap foreground/background. For masks where the network is dark on light. |

### Graph cleanup

| Flag | Default | What it does |
|---|---|---|
| `--prune-order` | 5 | Iteratively peels short branch triangles from the triangulation N times before the graph is built. Lower (1-3) preserves every twig. Higher (8-15) gives a cleaner main-branches graph. |
| `--remove-redundant` | `all` | Collapses degree-2 nodes by merging their two edges. `all` removes every degree-2 node. `none` keeps every intermediate point. `half` keeps every other. |
| `--prune-dangling` | off | Removes degree-1 dangling tips. Combine with `--prune-dangling-min-length`. |
| `--prune-dangling-min-length` | 0 | Only prune dangling tips shorter than N pixels. Sensible value: a few times the typical branch radius (e.g. 30-50 px). |

### Diagnosing common failures

- Spurious tiny edges everywhere: bump `--prune-order` or add `--prune-dangling --prune-dangling-min-length N`.
- Branches broken into pieces: add `--bridge-gaps 10`, or check whether `--invert` is needed.
- Real branches missing: `--min-feature-size` is too high, or `--smoothing` is eroding them.

`examples/tuning_walkthrough.py` generates visual examples of each
failure mode.

### On `--prune-order`

The default of 5 is tuned for large masks with tens of thousands of
triangles. On small masks (a few thousand triangles or fewer), even
`--prune-order 2` can wipe out the entire topology. For small or
moderate masks, start with 0 and use `--prune-dangling` to remove
spurs instead.

## Interactive editor

A napari-based editor for cleaning up false junctions, broken arms,
and other vectorisation artifacts by hand. Modernised port of
[Jana Lasser's `gegui`](https://github.com/JanaLasser/network_extraction)
(GPL-3, 2015, Python 2 / NetworkX 1.x) with a separable GUI-free
backend.

```bash
net3 edit graph.gpickle --mask mask.tif
```

Optional flags:

- `--distance-map dm.png`: sample radius for newly-created nodes from
  a precomputed distance transform.
- `--save out.gpickle`: save target on `s` or button. Defaults to
  overwriting the input.

### Workflow

1. Vectorise with `--for-editor` so the polyline traces the mask.
2. Edit interactively.
3. Press `n` before save to collapse back to topological form, or
   re-vectorise without `--for-editor` to a separate file.

### Modes

Click a mode button in the dock or press `1` / `2` / `3`.

| Mode | Key | Left-click | Left-drag |
|---|---|---|---|
| Select (default) | `1` | Toggle nearest node or edge | Pan |
| Rect-Select | `2` | Same as Select | Rubber-band rectangle, selects nodes inside on release |
| Add Node | `3` | Create new node (snapped) | Pan |

Shift + click in any mode creates a new node. Shift + drag in
Rect-Select is additive.

### Actions

| Action | Key |
|---|---|
| Delete selected nodes and edges | `d` |
| Connect two selected nodes | `e` |
| Highlight cycles | `m` |
| Streamline (collapse degree-2) | `n` |
| Clear selection | `c` |
| Undo | `z` |
| Save | `s` |

Node colors: green = degree-1 tip, cyan = degree ≥ 3 junction,
red = intermediate, yellow = selected.
Edge colors: orange = normal, yellow (thicker) = selected.

Snap-to-centerline: when a distance map is loaded, new nodes from
Shift+click are snapped to the local distance-transform peak within
a ±6 px window.

Edge editing: clicking an edge with no node in click range toggles
its selection. `d` deletes selected edges (keeping endpoints), useful
for removing a false junction between two crossing branches without
touching the branches themselves.

### Backend (no GUI)

Editing operations live in `net3.edit.GraphEditor`, a plain Python
class usable without napari. See its docstring or
`tests/test_edit_core.py`.

## Caveats

- Radii are distance-transform values at triangle centers. Good
  first-pass estimate; sub-pixel-accurate diameter requires per-edge
  refitting downstream.
- `min_feature_size` is in pixels and the most magnification-sensitive
  parameter. Always reset it for new data.
- The loader auto-thresholds at 127 if the image isn't already 0/1.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
