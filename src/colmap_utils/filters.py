# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OMNI-DC-MA contributors

"""Point quality and depth-consistency filters for COLMAP models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_MIN_TRACK_LENGTH = 3
DEFAULT_MAX_REPROJ_ERROR = 2.0
INVERSE_DEPTH_EPS = 1e-6


def track_length(point) -> int:
    return int(len(point.image_ids))


def reprojection_error(point) -> float:
    return float(point.error)


def point_is_certain(point, args: argparse.Namespace) -> bool:
    if args.no_quality_filter:
        return True
    return track_length(point) >= args.min_track_length and reprojection_error(point) <= args.max_reproj_error


def quality_filter_label(args: argparse.Namespace) -> str:
    if args.no_quality_filter:
        return "off"
    return f"track_length>={args.min_track_length}, reprojection_error<={args.max_reproj_error:g}px"


def consistency_enabled(args: argparse.Namespace) -> bool:
    return bool(args.consistency_depth_dir) and (args.max_inv_depth_diff > 0 or args.max_inv_depth_rel_diff > 0)


def consistency_filter_label(args: argparse.Namespace) -> str:
    if not consistency_enabled(args):
        return "off"
    checks: list[str] = []
    if args.max_inv_depth_diff > 0:
        checks.append(f"absolute inverse-depth error <= {args.max_inv_depth_diff:g} 1/m")
    if args.max_inv_depth_rel_diff > 0:
        checks.append(f"relative inverse-depth error <= {args.max_inv_depth_rel_diff:g}")
    align = "reference scale aligned" if args.consistency_align_scale else "no scale alignment"
    mode = (
        "drop whole inconsistent COLMAP points"
        if args.consistency_drop_point_all_views
        else "drop inconsistent observations only"
    )
    return f"{', '.join(checks)} against {args.consistency_depth_dir} ({align}; {mode})"


def normalize_depth_map(array: np.ndarray, path: Path) -> np.ndarray:
    depth = np.asarray(array)
    if depth.ndim == 3:
        if depth.shape[0] == 1:
            depth = depth[0]
        elif depth.shape[-1] == 1:
            depth = depth[..., 0]
        else:
            raise ValueError(f"Expected a single-channel depth map in {path}, got shape {depth.shape}")
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map in {path}, got shape {depth.shape}")
    return depth.astype(np.float32, copy=False)


def load_consistency_depth(stem: str, args: argparse.Namespace) -> np.ndarray | None:
    if not consistency_enabled(args):
        return None
    path = Path(args.consistency_depth_dir) / f"{stem}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Consistency depth map not found for {stem}: {path}")
    return normalize_depth_map(np.load(path), path)


def sample_depth_map(depth_map: np.ndarray, xys: np.ndarray, source_width: int, source_height: int) -> np.ndarray:
    ref_height, ref_width = depth_map.shape
    scale_x = (ref_width - 1) / max(source_width - 1, 1)
    scale_y = (ref_height - 1) / max(source_height - 1, 1)
    xs = np.rint(xys[:, 0] * scale_x).astype(np.int64)
    ys = np.rint(xys[:, 1] * scale_y).astype(np.int64)
    xs = np.clip(xs, 0, ref_width - 1)
    ys = np.clip(ys, 0, ref_height - 1)
    return depth_map[ys, xs].astype(np.float64, copy=False)


def aligned_ref_inverse_depth(inv_sfm: np.ndarray, inv_ref: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if not args.consistency_align_scale:
        return inv_ref
    ratios = inv_sfm / np.maximum(inv_ref, INVERSE_DEPTH_EPS)
    valid = np.isfinite(ratios) & (ratios > 0)
    if not valid.any():
        return inv_ref
    scale = float(np.median(ratios[valid]))
    if not np.isfinite(scale) or scale <= 0:
        return inv_ref
    return inv_ref * scale


def inverse_depth_consistency_mask(
    sfm_depth: np.ndarray,
    reference_depth: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    valid = (
        np.isfinite(sfm_depth)
        & np.isfinite(reference_depth)
        & (sfm_depth > 0)
        & (reference_depth > 0)
    )
    keep = np.zeros(sfm_depth.shape, dtype=bool)
    if valid.any():
        valid_indices = np.flatnonzero(valid)
        inv_sfm = 1.0 / sfm_depth[valid]
        inv_ref = 1.0 / reference_depth[valid]
        inv_ref = aligned_ref_inverse_depth(inv_sfm, inv_ref, args)
        valid_keep = np.ones(inv_sfm.shape, dtype=bool)
        diff = np.abs(inv_sfm - inv_ref)
        if args.max_inv_depth_diff > 0:
            valid_keep &= diff <= args.max_inv_depth_diff
        if args.max_inv_depth_rel_diff > 0:
            denom = np.maximum(np.maximum(np.abs(inv_sfm), np.abs(inv_ref)), INVERSE_DEPTH_EPS)
            valid_keep &= (diff / denom) <= args.max_inv_depth_rel_diff
        keep[valid_indices[valid_keep]] = True
    invalid_reference = int((~valid).sum())
    return keep, invalid_reference

