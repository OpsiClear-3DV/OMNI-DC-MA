# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OMNI-DC-MA contributors

"""Add depth-guided single-view points to sparse COLMAP models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from colmap_utils.densify import densify_model_from_depth  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    io = parser.add_argument_group("input/output")
    io.add_argument("--model-dir", required=True, help="Input COLMAP sparse model directory, e.g. scene/sparse/0")
    io.add_argument(
        "--depth-dir",
        required=True,
        help="Directory of dense metric depth .npy files matched by image stem",
    )
    io.add_argument("--output-model-dir", required=True, help="Output COLMAP sparse model directory")
    io.add_argument("--rgb-dir", help="Optional RGB directory for coloring added points by image pixel")
    io.add_argument("--model-ext", default="", choices=("", ".bin", ".txt"), help="Input COLMAP model format")
    io.add_argument("--output-model-ext", default=".bin", choices=(".bin", ".txt"), help="Output COLMAP model format")
    io.add_argument("--image-ext", action="append", help="RGB extensions for --rgb-dir; may repeat or comma-separate")
    io.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output model directory")
    io.add_argument("--dry-run", action="store_true", help="Compute additions but do not write a model")
    io.add_argument(
        "--only-stem",
        action="append",
        help="Only process matching image stems. May repeat or comma-separate",
    )

    density = parser.add_argument_group("density target")
    density.add_argument(
        "--cell-size",
        type=int,
        default=16,
        help="Image grid cell size in pixels. Default: %(default)s",
    )
    density.add_argument(
        "--min-points-per-cell",
        type=int,
        default=1,
        help="Target minimum existing-or-added points per cell. Default: %(default)s",
    )
    density.add_argument(
        "--points-per-cell-per-iteration",
        type=int,
        default=1,
        help="Maximum points added to one underfilled cell in one iteration. Default: %(default)s",
    )
    density.add_argument("--iterations", type=int, default=1, help="Repeat density pass this many times")
    density.add_argument("--max-points-per-image", type=int, default=0, help="Per-image addition cap. 0 means no cap")

    depth = parser.add_argument_group("depth checks")
    depth.add_argument("--min-depth", type=float, default=0.0, help="Ignore depth below this many meters. 0 disables")
    depth.add_argument("--max-depth", type=float, default=0.0, help="Ignore depth above this many meters. 0 disables")
    depth.add_argument(
        "--max-inverse-depth-error",
        dest="max_inv_depth_diff",
        type=float,
        default=0.0,
        help="Existing observed points count only when absolute inverse-depth error is within this 1/m threshold. "
        "0 disables this check.",
    )
    depth.add_argument(
        "--max-relative-inverse-depth-error",
        dest="max_inv_depth_rel_diff",
        type=float,
        default=0.25,
        help="Existing observed points count only when relative inverse-depth error is within this threshold. "
        "Default: %(default)s",
    )
    depth.add_argument(
        "--align-reference-depth-scale",
        dest="consistency_align_scale",
        action="store_true",
        default=False,
        help="Median-align reference inverse depths before checking existing-point consistency.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.cell_size < 1:
        raise ValueError("--cell-size must be >= 1")
    if args.min_points_per_cell < 1:
        raise ValueError("--min-points-per-cell must be >= 1")
    if args.points_per_cell_per_iteration < 1:
        raise ValueError("--points-per-cell-per-iteration must be >= 1")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.max_points_per_image < 0:
        raise ValueError("--max-points-per-image must be >= 0")
    if args.min_depth < 0 or args.max_depth < 0:
        raise ValueError("--min-depth and --max-depth must be >= 0")
    if args.max_depth > 0 and args.min_depth > args.max_depth:
        raise ValueError("--min-depth must be <= --max-depth")
    if args.max_inv_depth_diff < 0:
        raise ValueError("--max-inverse-depth-error must be >= 0")
    if args.max_inv_depth_rel_diff < 0:
        raise ValueError("--max-relative-inverse-depth-error must be >= 0")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args)
    stats = densify_model_from_depth(args)
    print(
        "depth-guided COLMAP point addition: "
        f"images_seen={stats.images_seen}, "
        f"images_used={stats.images_used}, "
        f"cells={stats.cells_seen}, "
        f"underfilled={stats.cells_underfilled}, "
        f"added_points={stats.added_points}, "
        f"invalid_depth={stats.skipped_invalid_depth}, "
        f"unsupported_camera={stats.skipped_unsupported_camera}, "
        f"cap_skips={stats.skipped_existing_limit}"
    )
    if args.dry_run:
        print("dry run: no model written")
    else:
        print(f"wrote COLMAP model to {args.output_model_dir}")


if __name__ == "__main__":
    main()
