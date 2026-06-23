"""
Command-line interface for net3.

Usage:
    net3 mask.tif -o graph.gpickle
    net3 mask.png -o graph.graphml --format graphml --invert
"""

import argparse
import sys
from pathlib import Path

from .pipeline import vectorize, save_graph
from . import __version__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="net3",
        description=(
            "Vectorise a binary mask into a NetworkX graph "
            "(nodes carry x/y/radius, edges carry length/radius)."
        ),
    )
    parser.add_argument(
        "mask",
        type=Path,
        help="Path to the binary mask (PNG/TIFF/JPG). Any image readable "
             "by PIL; auto-thresholded at 127 if not already 0/1.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output graph path.",
    )
    parser.add_argument(
        "--format",
        choices=("gpickle", "graphml", "gml", "edgelist"),
        default="gpickle",
        help="Output graph format. Default: gpickle.",
    )
    parser.add_argument(
        "--min-feature-size",
        type=int,
        default=3000,
        help="Drop foreground components smaller than this (px). Default: 3000.",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=None,
        help="Optional morphological smoothing kernel size.",
    )
    parser.add_argument(
        "--bridge-gaps",
        type=int,
        default=None,
        help="Closing radius to bridge gaps between disconnected regions. "
             "Useful for masks where vessels are cut at tile boundaries. "
             "Typical values: 5-15.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the mask (use when foreground is dark on light "
             "background).",
    )
    parser.add_argument(
        "--prune-order",
        type=int,
        default=5,
        help="Number of triangles to prune from branch tips. Default: 5.",
    )
    parser.add_argument(
        "--remove-redundant",
        choices=("all", "half", "none"),
        default="all",
        help="Remove degree-2 redundant nodes. Default: all.",
    )
    parser.add_argument(
        "--prune-dangling",
        action="store_true",
        help="Prune degree-1 dangling branches after graph construction.",
    )
    parser.add_argument(
        "--prune-dangling-min-length",
        type=float,
        default=0.0,
        help="Minimum length (px) of dangling branches kept when "
             "--prune-dangling is on. Default: 0.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-stage progress logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"net3 {__version__}",
    )
    args = parser.parse_args(argv)

    if not args.mask.exists():
        print(f"error: mask not found: {args.mask}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)

    G = vectorize(
        str(args.mask),
        min_feature_size=args.min_feature_size,
        smoothing=args.smoothing,
        bridge_gaps=args.bridge_gaps,
        invert=args.invert,
        prune_order=args.prune_order,
        remove_redundant=args.remove_redundant,
        prune_dangling=args.prune_dangling,
        prune_dangling_min_length=args.prune_dangling_min_length,
        verbose=not args.quiet,
    )

    save_graph(G, str(args.output), format=args.format)
    if not args.quiet:
        print(f"Wrote {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
