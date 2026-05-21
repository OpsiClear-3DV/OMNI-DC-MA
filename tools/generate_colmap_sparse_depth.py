# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OMNI-DC-MA contributors
# ruff: noqa: E402, I001

"""Generate OMNI-DC sparse-depth .npy files from a COLMAP sparse model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from colmap_utils.filters import (  # noqa: E402
    DEFAULT_MAX_REPROJ_ERROR,
    DEFAULT_MIN_TRACK_LENGTH,
    INVERSE_DEPTH_EPS as _INVERSE_DEPTH_EPS,
    consistency_enabled,
    consistency_filter_label,
    inverse_depth_consistency_mask,
    point_is_certain,
    quality_filter_label,
    reprojection_error,
    sample_depth_map,
    track_length,
)
from colmap_utils.sparse_depth import (  # noqa: E402
    DEFAULT_IMAGE_EXTS as _DEFAULT_IMAGE_EXTS,
    depth_for_image,
    find_global_consistency_rejects,
    generate_sparse_depth_maps,
    parse_exts,
    parse_stems,
    project_consistency_candidates,
    rgb_files,
)

# Backward-compatible helper names used by tests and older local scripts.
DEFAULT_IMAGE_EXTS = _DEFAULT_IMAGE_EXTS
INVERSE_DEPTH_EPS = _INVERSE_DEPTH_EPS
_parse_exts = parse_exts
_parse_stems = parse_stems
_rgb_files = rgb_files
_track_length = track_length
_reprojection_error = reprojection_error
_point_is_certain = point_is_certain
_quality_filter_label = quality_filter_label
_consistency_enabled = consistency_enabled
_consistency_filter_label = consistency_filter_label
_sample_depth_map = sample_depth_map
_inverse_depth_consistency_mask = inverse_depth_consistency_mask
_project_consistency_candidates = project_consistency_candidates
_find_global_consistency_rejects = find_global_consistency_rejects
_depth_for_image = depth_for_image
generate = generate_sparse_depth_maps


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--model-dir", required=True, help="COLMAP sparse model directory, e.g. sparse/0")
    io_group.add_argument("--rgb-dir", required=True, help="Directory containing RGB images to match by stem")
    io_group.add_argument("--out-dir", required=True, help="Output directory for .npy sparse depth maps")
    io_group.add_argument("--model-ext", default="", choices=("", ".bin", ".txt"), help="COLMAP model format")
    io_group.add_argument(
        "--image-ext",
        action="append",
        help="Image extension to include. May be repeated or comma-separated. Defaults to common JPG/PNG forms.",
    )
    io_group.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy outputs")
    io_group.add_argument("--only-stem", action="append", help="Only process matching RGB stems. May be repeated.")
    io_group.add_argument("--limit", type=int, default=0, help="Stop after writing this many files. 0 means no limit.")

    quality_group = parser.add_argument_group("COLMAP point quality filter")
    quality_group.add_argument(
        "--min-track-length",
        type=int,
        default=DEFAULT_MIN_TRACK_LENGTH,
        help="Require COLMAP 3D points to be observed in at least this many images. Default: %(default)s.",
    )
    quality_group.add_argument(
        "--max-reproj-error",
        type=float,
        default=DEFAULT_MAX_REPROJ_ERROR,
        help="Require COLMAP point reprojection error at or below this many pixels. Default: %(default)s.",
    )
    quality_group.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable the default certain-point filter. Intended only for comparison/debugging.",
    )

    consistency_group = parser.add_argument_group("SfM vs reference-depth filter")
    consistency_group.add_argument(
        "--reference-depth-dir",
        dest="consistency_depth_dir",
        metavar="DIR",
        help="Directory of reference dense depth .npy maps matched by RGB stem. "
        "Use with an inverse-depth error threshold to reject inconsistent SfM anchors.",
    )
    consistency_group.add_argument(
        "--consistency-depth-dir",
        dest="consistency_depth_dir",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    consistency_group.add_argument(
        "--max-inverse-depth-error",
        dest="max_inv_depth_diff",
        metavar="ERROR_1_PER_M",
        type=float,
        default=0.0,
        help="Reject anchors with absolute inverse-depth error above this threshold in 1/m. "
        "Requires --reference-depth-dir. 0 disables this check.",
    )
    consistency_group.add_argument(
        "--max-inv-depth-diff",
        dest="max_inv_depth_diff",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    consistency_group.add_argument(
        "--max-relative-inverse-depth-error",
        dest="max_inv_depth_rel_diff",
        metavar="REL_ERROR",
        type=float,
        default=0.0,
        help="Reject anchors with symmetric relative inverse-depth error above this threshold. "
        "For example, 0.25 rejects points differing by more than about 25%%. "
        "Requires --reference-depth-dir. 0 disables this check.",
    )
    consistency_group.add_argument(
        "--max-inv-depth-rel-diff",
        dest="max_inv_depth_rel_diff",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    consistency_group.add_argument(
        "--align-reference-depth-scale",
        dest="consistency_align_scale",
        action="store_true",
        default=False,
        help="Median-align reference inverse depths to SfM inverse depths per image before checking errors. "
        "Useful when the reference depth maps are only relatively scaled.",
    )
    consistency_group.add_argument(
        "--consistency-align-scale",
        dest="consistency_align_scale",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    consistency_group.add_argument(
        "--drop-inconsistent-points",
        dest="consistency_drop_point_all_views",
        action="store_true",
        default=False,
        help="If any observation of a COLMAP 3D point fails the reference-depth check, reject that 3D point "
        "in all selected output views. Default removes only failing observations.",
    )
    consistency_group.add_argument(
        "--consistency-drop-point-all-views",
        "--consistency-remove-point-all-views",
        action="store_true",
        dest="consistency_drop_point_all_views",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    io_group.add_argument("--verbose", action="store_true", help="Print one line per written image")


def _validate_args(args: argparse.Namespace) -> None:
    if args.min_track_length < 1:
        raise ValueError("--min-track-length must be >= 1")
    if args.max_reproj_error <= 0:
        raise ValueError("--max-reproj-error must be > 0")
    if args.max_inv_depth_diff < 0:
        raise ValueError("--max-inverse-depth-error must be >= 0")
    if args.max_inv_depth_rel_diff < 0:
        raise ValueError("--max-relative-inverse-depth-error must be >= 0")
    if (args.max_inv_depth_diff > 0 or args.max_inv_depth_rel_diff > 0) and not args.consistency_depth_dir:
        raise ValueError("inverse-depth error thresholds require --reference-depth-dir")
    if args.consistency_drop_point_all_views and not consistency_enabled(args):
        raise ValueError(
            "--drop-inconsistent-points requires --reference-depth-dir and an inverse-depth error threshold"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args)
    generate_sparse_depth_maps(args)


if __name__ == "__main__":
    main()
