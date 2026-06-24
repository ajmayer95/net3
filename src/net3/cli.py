"""
Command-line interface for net3.

Subcommands:
    net3 vectorize MASK -o GRAPH        (or omit the subcommand —
                                         see below for back-compat)
    net3 edit GRAPH [--mask MASK] [--distance-map DM] [--save OUT]

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
    try:
        from .edit.app import run_editor
    except ImportError as exc:
        print(
            "error: net3 edit requires the [gui] extra to be installed:\n"
            "    pip install 'net3[gui]'\n"
            f"underlying ImportError: {exc}",
            file=sys.stderr,
        )
        return 3
    run_editor(
        graph_path=args.graph,
        mask_path=args.mask,
        distance_map_path=args.distance_map,
        save_path=args.save,
    )
    return 0


# ── Top-level parser ──────────────────────────────────────────────


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Back-compat shim: if the first arg looks like a path to a mask
    # AND no subcommand keyword is given, fall through to vectorize.
    has_subcmd = any(a in ("vectorize", "edit") for a in argv[:1])
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
    p_edit.set_defaults(func=_run_edit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
