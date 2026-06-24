# net3

Binary mask → NetworkX graph + interactive napari-based editor.

## Install

Needs Python ≥ 3.9 + a C compiler (`xcode-select --install` on macOS;
`build-essential` on Debian/Ubuntu; MSVC Build Tools on Windows).

```bash
mkdir -p ~/net3 && cd ~/net3
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install 'net3[gui] @ git+https://github.com/ajmayer95/net3.git'
```

Test:

```bash
net3 --version
```

## Try it

Download a sample mask, vectorise it, open in the editor:

```bash
curl -L https://raw.githubusercontent.com/JanaLasser/network_extraction/master/data/binaries/tracheole1_binary.png -o test_mask.png
net3 vectorize test_mask.png -o test_graph.gpickle --min-feature-size 200 --for-editor
net3 edit test_graph.gpickle --mask test_mask.png
```

A napari window opens with the mask in greyscale + an orange polyline
tracing every branch (red dots at intermediate nodes, green at tips,
cyan at junctions). Click around, press `m` to highlight cycles, close
to exit.

That's it — net3 works on your machine.

---

Everything below is reference material for tuning parameters, the
Python API, the editor's full keyboard cheatsheet, and the developer
install. Read it when you hit a specific question, not start-to-finish.

---

## Two forms of the graph

`net3 vectorize` produces one of two shapes depending on `--for-editor`:

- **Topological** (default): every node is a tip or junction. Edges
  are straight lines between them. Right form for graph analysis;
  visually confusing in the editor because edges cut across the mask.
- **Centerline** (`--for-editor`): every intermediate node kept.
  Polyline traces the mask. Right form for the editor.

If you need both — edit the centerline form, then press `n`
(streamline) in the editor before saving to collapse to topological.

## Developer install

If you want to modify net3's source:

```bash
git clone https://github.com/ajmayer95/net3.git
cd net3
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[gui,dev]'
pytest -q
```

Edits to `src/net3/` take effect immediately. `pytest -q` reports 32
passed.

## Use from Python

Inside a Python script, notebook, or `python3` REPL — NOT pasted into a
shell. Save this to `run_me.py` and run with `python3 run_me.py`:

```python
from net3 import vectorize, save_graph

G = vectorize("test_mask.png", min_feature_size=200)
print(G.number_of_nodes(), G.number_of_edges())

for u, v, data in G.edges(data=True):
    print(u, v, data["radius"], data["weight"])

save_graph(G, "test_graph.gpickle")
```

Node attributes: `x`, `y`, `radius` (pixels).
Edge attributes: `weight` = Euclidean length (pixels), `radius` = mean
of endpoint radii.

The single most important tunable is **`min_feature_size`** — it drops
foreground components smaller than this many pixels. The default of
3000 is tuned for the canonical case the package was built for; for
finer-scale data, drop it to 100–500. See "Tuning the flags" below.

## Output schema

| key | where | unit | meaning |
|---|---|---|---|
| `x`, `y` | node attr | pixels | image coords (y is flipped to mathematical convention) |
| `radius` | node + edge attr | pixels | distance-transform value (node) or mean of endpoint radii (edge) |
| `weight` | edge attr | pixels | Euclidean distance between endpoints |

`collapse_to_branch_graph(G)` gives one edge per branch (instead of
per inter-junction segment) with the full path geometry stored as an
edge attribute.

## Tuning the flags

The flags fall into two phases: **mask cleanup** (before vectorisation)
and **graph cleanup** (after). Mask cleanup is applied in this order:
`bridge-gaps → remove small objects → smoothing → distance transform`.

### Mask cleanup

| Flag | Default | What it does | When to change it |
|---|---|---|---|
| `--min-feature-size` | 3000 px | Drops any connected foreground blob smaller than N pixels. | **Always check this first.** It's the only mask-cleanup knob that affects every run. The default is tuned for the canonical use case the package was built for; for finer-scale data drop to 100–500. If real features keep getting lost → too high. If noise speckles appear as graph nodes → too low. |
| `--smoothing` | off | Morphological opening then closing with a disk of radius N, then a second small-object filter. Smooths jagged mask boundaries. | Use **only** if the mask has a noisy boundary producing spurious tiny branches. Typical values 1–3. Larger values erode thin features — stay below half the typical feature radius. |
| `--bridge-gaps` | off | Morphological closing with a disk of radius N, applied **before** the small-object filter. Bridges small gaps between disconnected pieces. | Use when segmentation breaks the network at boundaries (tile seams, faded segments). Typical 5–15. Larger values can fuse genuinely-separate branches — start small. |
| `--invert` | off | Swap foreground/background. | Set this if the network is dark on light. Quick check: open the mask — if the network is dark, add `--invert`. |

### Graph cleanup

| Flag | Default | What it does | When to change it |
|---|---|---|---|
| `--prune-order` | 5 | Iteratively peels short branch triangles from the triangulation N times before graph construction. Removes tiny spurs from boundary roughness. | Lower (1–3) preserves every twig. Higher (8–15) gives a cleaner "main branches only" graph. 5 is the default sweet spot. |
| `--remove-redundant` | `all` | Collapses degree-2 nodes (a node with exactly two neighbors) by merging their two edges. Default `all` means *every* degree-2 node is removed, so only junctions (deg ≥ 3) and tips (deg 1) remain. | `none` if you need every intermediate point along a branch (useful for per-segment curvature). `half` keeps every other — middle ground. For most analyses, `all`. |
| `--prune-dangling` | off | After everything else, removes degree-1 dangling tips. | Use to remove single short edges sticking out of junctions (segmentation noise). Combine with `--prune-dangling-min-length`. |
| `--prune-dangling-min-length` | 0 | Only prune dangling tips **shorter than N pixels**. | Keeps long end-branches (probably real) while removing short spurs. Sensible: a small multiple of the typical branch radius, e.g. 30–50 px. |

### How to tune

Run with defaults first, visualise the result, then diagnose:

- **Spurious tiny edges everywhere** → bump `--prune-order` or add `--prune-dangling --prune-dangling-min-length N`.
- **Branches broken into pieces** → add `--bridge-gaps 10`, or check whether `--invert` is needed.
- **Real branches missing** → `--min-feature-size` is too high, or `--smoothing` is eroding them.

For fine-scale data, drop `--min-feature-size` to ~200 and `--prune-order` to 3. For noisy mask boundaries, add `--smoothing 2`. Run `examples/tuning_walkthrough.py` after install for visual examples of each failure mode.

### A gotcha worth knowing

`--prune-order` is *much* more aggressive than the help text suggests. It operates on triangles, not graph edges, and iteratively peels them off **before** the graph is built. On small masks (anything under a few thousand triangles) even `--prune-order 2` can wipe out the entire topology. The default 5 is tuned for large masks with tens of thousands of triangles.

Rule of thumb: for small or moderate masks, start with `--prune-order 0` and bump up only if you see spurs that `--prune-dangling --prune-dangling-min-length N` alone can't remove. For large, dense masks, 3–5 is fine.

## Interactive editor

For cleaning up false junctions, broken arms, and other vectorisation
artifacts by hand, net3 ships a napari-based editor. Modernised port of
[Jana Lasser's `gegui`](https://github.com/JanaLasser/network_extraction)
(GPL-3, 2015, Python 2 / NetworkX 1.x) with a separable GUI-free
backend.

If you didn't install the `[gui]` extra earlier, do it now (adds napari
+ qtpy + PyQt5):

```bash
pip install 'net3[gui] @ git+https://github.com/ajmayer95/net3.git'
```

Then open the editor on an existing graph + its source mask:

```bash
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
the mask centerline closely. Three-step workflow:

**1.** Vectorise in editor-friendly form (keeps every intermediate node):

```bash
net3 vectorize mask.tif -o graph_edit.gpickle --for-editor
```

**2.** Edit interactively (clean up false junctions, broken arms, ...):

```bash
net3 edit graph_edit.gpickle --mask mask.tif --save graph_edit.gpickle
```

**3.** Collapse to topological form for analysis: press `n` in the editor
before saving, OR run a second vectorise pass without `--for-editor`.

If you launch the editor on a graph that's already been streamlined,
it'll print a hint with the right command to re-vectorise.

### Modes

Click a mode button in the dock (or press `1` / `2` / `3`) to switch.
Only one mode is on at a time.

| Mode | Key | What left-click does | What left-drag does |
|---|---|---|---|
| **Select** (default) | `1` | Toggle nearest node OR edge | Pan |
| **Rect-Select** | `2` | Same as Select (single click still works) | Rubber-band → select nodes inside on release |
| **Add Node** | `3` | Create new node (snapped) | Pan |

Shift modifiers:
- **Shift + click** in any mode → create new node (a shortcut, doesn't require switching mode).
- **Shift + drag** in Rect-Select → additive selection.

### Actions

| Action | Key |
|---|---|
| Delete selected nodes AND edges | `d` |
| Connect two selected nodes | `e` |
| Highlight cycles | `m` |
| Streamline (collapse degree-2) | `n` |
| Clear selection | `c` |
| Undo last edit | `z` |
| Save | `s` |

Node colors: **green** = degree-1 tip, **cyan** = degree-≥3 junction,
**red** = intermediate, **yellow** = selected.
Edge colors: **orange** = normal, **yellow** (thicker) = selected.

**Snap-to-centerline**: when a distance map is loaded
(`--distance-map dm.png`), new nodes from Shift+click are snapped to
the local distance-transform peak within a ±6 px window. Drop a node
slightly off-centerline and it'll pull onto the branch midline.

**Edge editing**: clicking an edge that has no node within click
range toggles its selection. Press `d` to delete just the edge
(keeping endpoints). This is the right operation for "false junction
between two crossing branches" — delete the false edge, the two real
branches keep their structure.

### Backend API (no GUI)

All editing operations live in `net3.edit.GraphEditor` — a plain
Python class usable from scripts/notebooks without napari. See its
docstring (`help(net3.edit.GraphEditor)`) or `tests/test_edit_core.py`
for the full surface.

## Caveats

- Radii are distance-transform values at triangle centers. Good first-pass estimate; sub-pixel-accurate diameter requires per-edge refitting downstream.
- `min_feature_size` is in pixels and is the most magnification-sensitive parameter — always reset it for new data.
- The loader auto-thresholds at 127 if the image isn't already 0/1.

## License

MIT — see [LICENSE](LICENSE).
