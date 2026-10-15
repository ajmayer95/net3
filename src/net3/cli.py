"""
Command-line interface for net3.

Subcommands:
    net3 vectorize MASK -o GRAPH        (or omit the subcommand —
                                         see below for back-compat)
    net3 edit GRAPH [--mask MASK] [--distance-map DM] [--save OUT]
    net3 edit-mask MASK [--image RAW] [--probs PROBS] [--save OUT]

Back-compat: if the first non-flag argument looks like a mask file and
neither subcommand keyword is present, the parser falls back to the
old ``net3 mask.tif -o graph.gpickle`` form so existing scripts keep
working.
"""

import argparse
import sys
from pathlib import Path

from .pipeline import vectorize, save_graph
from . import __version__


# ── Shared helpers ────────────────────────────────────────────────


def _add_vectorize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "mask", type=Path,
        help=("Path to the binary mask (PNG/TIFF/JPG). Any image "
              "readable by PIL; auto-thresholded at 127 if not "
              "already 0/1."),
    )
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output graph path.")
    parser.add_argument(
        "--format",
        choices=("gpickle", "graphml", "gml", "edgelist"),
        default="gpickle",
        help="Output graph format. Default: gpickle.",
    )
    parser.add_argument(
        "--min-feature-size", type=int, default=3000,
        help=("Drop foreground components smaller than this (px). "
              "Default: 3000."),
    )
    parser.add_argument(
        "--smoothing", type=int, default=None,
        help="Optional morphological smoothing kernel size.",
    )
    parser.add_argument(
        "--bridge-gaps", type=int, default=None,
        help=("Closing radius to bridge gaps between disconnected "
              "regions. Useful for masks where vessels are cut at "
              "tile boundaries. Typical values: 5-15."),
    )
    parser.add_argument(
        "--invert", action="store_true",
        help=("Invert the mask (use when foreground is dark on light "
              "background)."),
    )
    parser.add_argument(
        "--prune-order", type=int, default=5,
        help=("Number of triangles to prune from branch tips. "
              "Default: 5."),
    )
    parser.add_argument(
        "--remove-redundant",
        choices=("all", "half", "none"),
        default="all",
        help="Remove degree-2 redundant nodes. Default: all.",
    )
    parser.add_argument(
        "--prune-dangling", action="store_true",
        help="Prune degree-1 dangling branches after graph construction.",
    )
    parser.add_argument(
        "--prune-dangling-min-length", type=float, default=0.0,
        help=("Minimum length (px) of dangling branches kept when "
              "--prune-dangling is on. Default: 0."),
    )
    parser.add_argument(
        "--merge-junctions", type=float, default=0.0,
        metavar="D_MAX",
        help=("Merge degree-3+ junction nodes within D_MAX px of each "
              "other (union-find).  Fixes the fat-vessel triple-point "
              "artifact where one physical junction produces N tightly-"
              "spaced graph nodes.  Default 0 = off; typical value 2-3 px."),
    )
    parser.add_argument(
        "--for-editor", action="store_true",
        help=("Shorthand for `--remove-redundant none`.  Keeps every "
              "intermediate triangle-center node so straight-line "
              "edges trace the mask centerline — the form the "
              "interactive editor (`net3 edit`) expects.  Overrides "
              "--remove-redundant if both are given."),
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress per-stage progress logging.",
    )


def _run_vectorize(args: argparse.Namespace) -> int:
    if not args.mask.exists():
        print(f"error: mask not found: {args.mask}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    remove_redundant = "none" if args.for_editor else args.remove_redundant
    G = vectorize(
        str(args.mask),
        min_feature_size=args.min_feature_size,
        smoothing=args.smoothing,
        bridge_gaps=args.bridge_gaps,
        invert=args.invert,
        prune_order=args.prune_order,
        remove_redundant=remove_redundant,
        prune_dangling=args.prune_dangling,
        prune_dangling_min_length=args.prune_dangling_min_length,
        merge_junctions_d_max=args.merge_junctions,
        verbose=not args.quiet,
    )
    save_graph(G, str(args.output), format=args.format)
    if not args.quiet:
        print(f"Wrote {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges → {args.output}")
    return 0


def _run_edit(args: argparse.Namespace) -> int:
    if not args.graph.exists():
        print(f"error: graph not found: {args.graph}", file=sys.stderr)
        return 2
    if args.mask is not None and not args.mask.exists():
        print(f"error: mask not found: {args.mask}", file=sys.stderr)
        return 2
    if args.distance_map is not None and not args.distance_map.exists():
        print(f"error: distance map not found: {args.distance_map}",
              file=sys.stderr)
        return 2
    # Probe the GUI deps up front so a missing `napari` / `qtpy` /
    # `PyQt5` produces a friendly install hint instead of a deep
    # traceback from inside the editor (the editor lazily imports
    # napari, so the outer `from .edit.app` succeeds even without it).
    missing = []
    for dep in ("napari", "qtpy"):
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    if missing:
        print(
            "error: net3 edit requires the [gui] extra to be installed.\n"
            "    pip install 'net3[gui]'\n"
            f"missing module(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 3
    # Optional video-overlay args — validate that all three come
    # together or none.  Missing tile-positions JSON is a hard error
    # if either video arg is passed; missing video dir is only a
    # warning (dropdown will list tiles but load will fail per tile).
    video_dirs = getattr(args, 'video_dir', None) or []
    video_pattern = getattr(args, 'video_pattern', None)
    tp_path = getattr(args, 'tile_positions', None)
    if (video_dirs or video_pattern) and tp_path is None:
        print("error: --video-dir / --video-pattern require --tile-positions",
              file=sys.stderr)
        return 2
    if tp_path is not None and not tp_path.exists():
        print(f"error: tile-positions JSON not found: {tp_path}",
              file=sys.stderr)
        return 2

    from .edit.app import run_editor
    run_editor(
        graph_path=args.graph,
        mask_path=args.mask,
        distance_map_path=args.distance_map,
        save_path=args.save,
        tile_positions_path=tp_path,
        video_dirs=video_dirs if video_dirs else None,
        video_pattern=video_pattern,
    )
    return 0


def _run_edit_mask(args: argparse.Namespace) -> int:
    if not args.mask.exists():
        print(f"error: mask not found: {args.mask}", file=sys.stderr)
        return 2
    if args.image is not None and not args.image.exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2
    if args.probs is not None and not args.probs.exists():
        print(f"error: probs not found: {args.probs}", file=sys.stderr)
        return 2
    missing = []
    for dep in ("napari", "qtpy"):
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    if missing:
        print(
            "error: net3 edit-mask requires the [gui] extra to be installed.\n"
            "    pip install 'net3[gui]'\n"
            f"missing module(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 3
    from .edit.mask_gui import open_mask_editor
    # Optional per-tile video overlay (same flags as `net3 edit`).
    on_ready = None
    video_dirs = getattr(args, 'video_dir', None) or []
    tp_path = getattr(args, 'tile_positions', None)
    video_pattern = getattr(args, 'video_pattern', None)
    if tp_path is not None and video_dirs and video_pattern is not None:
        if not tp_path.exists():
            print(f"error: tile-positions JSON not found: {tp_path}",
                  file=sys.stderr)
            return 4
        def on_ready(viewer, editor):
            from .edit.video_overlay import (
                VideoOverlayManager, load_tile_positions)
            try:
                positions = load_tile_positions(tp_path)
            except Exception as e:
                print(f"[net3 edit-mask] failed to load tile positions "
                      f"({tp_path}): {e}")
                return
            if not positions:
                print("[net3 edit-mask] tile-positions JSON contained no entries")
                return
            mask_shape = None
            try:
                if editor.mask is not None:
                    mask_shape = tuple(int(x) for x in editor.mask.shape[:2])
            except Exception:
                mask_shape = None
            mgr = VideoOverlayManager(viewer, positions,
                                       video_dirs, video_pattern,
                                       mask_shape=mask_shape)
            from qtpy.QtWidgets import (
                QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                QPushButton, QComboBox, QCheckBox)
            dock = QWidget()
            lay = QVBoxLayout(dock)
            lay.addWidget(QLabel("<b>Tile video overlay</b>"))
            row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0)
            rl.addWidget(QLabel("Tile:"))
            combo = QComboBox()
            for tid in mgr.available_tile_ids():
                combo.addItem(str(tid), userData=tid)
            rl.addWidget(combo, stretch=1); lay.addWidget(row)
            load_btn = QPushButton("Load tile video")
            load_btn.clicked.connect(
                lambda: mgr.load_tile(int(combo.currentData())))
            lay.addWidget(load_btn)
            unload_btn = QPushButton("Unload current video")
            unload_btn.clicked.connect(mgr.unload)
            lay.addWidget(unload_btn)
            b_chk = QCheckBox("Show tile boundaries")
            b_chk.toggled.connect(lambda c: mgr.set_show_tile_boundaries(bool(c)))
            lay.addWidget(b_chk)
            l_chk = QCheckBox("Show tile numbers")
            l_chk.toggled.connect(lambda c: mgr.set_show_tile_numbers(bool(c)))
            lay.addWidget(l_chk)
            lay.addStretch(1)
            viewer.window.add_dock_widget(dock, area="right",
                                          name="Tile video overlay")
            print(f"[net3 edit-mask] video overlay ready: "
                  f"{len(positions)} tiles, pattern={video_pattern!r}")
    open_mask_editor(
        mask=args.mask,
        image=args.image,
        probs=args.probs,
        save_path=args.save,
        on_ready=on_ready,
    )
    return 0


# ── Top-level parser ──────────────────────────────────────────────


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Back-compat shim: if the first arg looks like a path to a mask
    # AND no subcommand keyword is given, fall through to vectorize.
    has_subcmd = any(a in ("vectorize", "edit", "edit-mask") for a in argv[:1])
    if not has_subcmd and argv and not argv[0].startswith("-"):
        argv = ["vectorize", *argv]

    parser = argparse.ArgumentParser(
        prog="net3",
        description=("Vectorise a binary mask into a NetworkX graph, "
                     "or interactively edit an existing graph."),
    )
    parser.add_argument("--version", action="version",
                        version=f"net3 {__version__}")

    subs = parser.add_subparsers(dest="cmd", required=True)

    p_vec = subs.add_parser(
        "vectorize",
        help="Mask → graph (the default behaviour).",
        description=("Vectorise a binary mask into a NetworkX graph "
                     "(nodes carry x/y/radius, edges carry length/radius)."),
    )
    _add_vectorize_args(p_vec)
    p_vec.set_defaults(func=_run_vectorize)

    p_edit = subs.add_parser(
        "edit",
        help="Open an interactive napari editor on an existing graph.",
        description=("Open an existing .gpickle graph in net3's "
                     "interactive editor (napari-based, requires "
                     "the [gui] install extra)."),
    )
    p_edit.add_argument("graph", type=Path,
                        help="Path to a .gpickle graph to edit.")
    p_edit.add_argument("-m", "--mask", type=Path, default=None,
                        help="Optional mask image to display as background.")
    p_edit.add_argument(
        "--distance-map", type=Path, default=None,
        help=("Optional distance map (PNG/TIFF) used to sample radius "
              "for newly created nodes."),
    )
    p_edit.add_argument(
        "--save", type=Path, default=None,
        help=("Save target for the 's' / Save button (defaults to "
              "overwriting the input graph)."),
    )
    # Optional per-tile video overlay (all three must come together)
    p_edit.add_argument(
        "--tile-positions", type=Path, default=None,
        help=("Optional JSON file mapping tile-id → tile position + "
              "size on the mosaic.  Enables the video-overlay dock.  "
              "Format: dict of tile-id → {top_left_x, top_left_y, "
              "tile_width, tile_height}."),
    )
    p_edit.add_argument(
        "--video-dir", type=Path, action='append', default=None,
        help=("Directory (or dirs — repeat --video-dir) that contain "
              "per-tile videos.  Requires --tile-positions and "
              "--video-pattern."),
    )
    p_edit.add_argument(
        "--video-pattern", type=str, default=None,
        help=("Filename pattern for per-tile videos with {id} or {vid} "
              "substituted with the tile id (extension optional; "
              "common ones probed).  E.g. 'vid{id}.tif' or "
              "'10x 250fps loc{id}_C001H001S0001'.  Directories "
              "matching the pattern are treated as frame folders."),
    )
    p_edit.set_defaults(func=_run_edit)

    p_em = subs.add_parser(
        "edit-mask",
        help="Open an interactive napari editor on a binary mask.",
        description=("Open an existing mask (TIFF/PNG) in net3's "
                     "interactive mask editor (napari-based, requires "
                     "the [gui] install extra).  Optional grayscale "
                     "reference image and/or probability map enable "
                     "additional tools."),
    )
    p_em.add_argument("mask", type=Path,
                      help="Path to a binary mask (TIFF/PNG) to edit.")
    p_em.add_argument("-i", "--image", type=Path, default=None,
                      help="Optional grayscale reference image shown "
                           "underneath the mask.")
    p_em.add_argument("-p", "--probs", type=Path, default=None,
                      help="Optional probability map (TIFF/PNG, 2-D). "
                           "Enables the threshold + cleanup panel.")
    p_em.add_argument("--save", type=Path, default=None,
                      help="Save target for the 's' / Save Mask button "
                           "(defaults to overwriting the input mask).")
    # Optional per-tile video overlay (same flags as `net3 edit`)
    p_em.add_argument("--tile-positions", type=Path, default=None,
                      help="Path to a tile-positions JSON for the "
                           "per-tile video overlay (optional).")
    p_em.add_argument("--video-dir", type=Path, action='append', default=None,
                      help="Directory (repeatable) that contains per-tile "
                           "videos. Requires --tile-positions + --video-pattern.")
    p_em.add_argument("--video-pattern", type=str, default=None,
                      help="Filename pattern with {vid} placeholder, e.g. "
                           "'10x 250fps loc{vid}_*'.")
    p_em.set_defaults(func=_run_edit_mask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
