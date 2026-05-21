# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OMNI-DC-MA contributors

"""COLMAP sparse-model to per-image sparse-depth conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .filters import (
    consistency_filter_label,
    inverse_depth_consistency_mask,
    load_consistency_depth,
    point_is_certain,
    quality_filter_label,
    sample_depth_map,
)
from .io import qvec2rotmat, read_colmap_model

DEFAULT_IMAGE_EXTS = (".JPG", ".jpg", ".JPEG", ".jpeg", ".png", ".PNG")


def parse_exts(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_IMAGE_EXTS
    exts: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                exts.append(item if item.startswith(".") else f".{item}")
    return tuple(dict.fromkeys(exts))


def parse_stems(values: list[str] | None) -> set[str]:
    stems: list[str] = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                stems.append(Path(item).stem)
    return set(stems)


def rgb_files(rgb_dir: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for ext in exts:
        for path in rgb_dir.glob(f"*{ext}"):
            files.setdefault(path.stem, path)
    return files


def project_consistency_candidates(
    image,
    camera,
    points3d,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    stats = {
        "track_refs": 0,
        "missing_points": 0,
        "rejected_quality": 0,
        "kept_points": 0,
        "outside_or_behind": 0,
    }
    point_ids = np.asarray(image.point3D_ids)
    valid_track = point_ids != -1
    stats["track_refs"] = int(valid_track.sum())
    if not valid_track.any():
        empty_ids = np.asarray([], dtype=np.int64)
        empty_xy = np.empty((0, 2), dtype=np.float64)
        empty_z = np.asarray([], dtype=np.float64)
        empty_inside = np.asarray([], dtype=bool)
        return empty_ids, empty_xy, empty_z, empty_inside, stats

    ids = point_ids[valid_track]
    xys = np.asarray(image.xys, dtype=np.float64)[valid_track]
    xyz = []
    keep = []
    kept_point_ids: list[int] = []
    for idx, point_id in enumerate(ids):
        point = points3d.get(int(point_id))
        if point is None:
            stats["missing_points"] += 1
            continue
        if not point_is_certain(point, args):
            stats["rejected_quality"] += 1
            continue
        xyz.append(point.xyz)
        keep.append(idx)
        kept_point_ids.append(int(point_id))
    stats["kept_points"] = len(xyz)
    if not xyz:
        empty_ids = np.asarray([], dtype=np.int64)
        empty_xy = np.empty((0, 2), dtype=np.float64)
        empty_z = np.asarray([], dtype=np.float64)
        empty_inside = np.asarray([], dtype=bool)
        return empty_ids, empty_xy, empty_z, empty_inside, stats

    xyz_arr = np.asarray(xyz, dtype=np.float64)
    xys = xys[np.asarray(keep, dtype=np.int64)]
    rot = qvec2rotmat(image.qvec)
    cam_xyz = xyz_arr @ rot.T + np.asarray(image.tvec, dtype=np.float64)
    z = cam_xyz[:, 2]

    width = int(camera.width)
    height = int(camera.height)
    xs = np.rint(xys[:, 0]).astype(np.int64)
    ys = np.rint(xys[:, 1]).astype(np.int64)
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height) & (z > 0)
    stats["outside_or_behind"] = int(len(inside) - int(inside.sum()))
    return np.asarray(kept_point_ids, dtype=np.int64), xys, z, inside, stats


def find_global_consistency_rejects(
    work_items: list[tuple[object, str, Path]],
    cameras,
    points3d,
    args: argparse.Namespace,
) -> set[int]:
    rejected: set[int] = set()
    for image, stem, _out_path in work_items:
        camera = cameras[image.camera_id]
        point_ids, xys, z, inside, _stats = project_consistency_candidates(image, camera, points3d, args)
        if not inside.any():
            continue
        consistency_depth = load_consistency_depth(stem, args)
        inside_indices = np.flatnonzero(inside)
        reference_depth = sample_depth_map(consistency_depth, xys[inside], int(camera.width), int(camera.height))
        consistency_keep, _invalid_reference = inverse_depth_consistency_mask(z[inside], reference_depth, args)
        rejected.update(int(point_id) for point_id in point_ids[inside_indices[~consistency_keep]])
    return rejected


def depth_for_image(
    image,
    stem: str,
    camera,
    points3d,
    args: argparse.Namespace,
    global_consistency_rejects: set[int] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    height = int(camera.height)
    width = int(camera.width)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    consistency_depth = None if global_consistency_rejects is not None else load_consistency_depth(stem, args)
    stats = {
        "track_refs": 0,
        "missing_points": 0,
        "rejected_quality": 0,
        "rejected_consistency": 0,
        "invalid_consistency_depth": 0,
        "kept_points": 0,
        "outside_or_behind": 0,
        "projected": 0,
        "anchors": 0,
    }

    point_ids, xys, z, inside, candidate_stats = project_consistency_candidates(image, camera, points3d, args)
    for key, value in candidate_stats.items():
        stats[key] = value
    if len(point_ids) == 0:
        depth[~np.isfinite(depth)] = 0.0
        return depth, stats

    xs = np.rint(xys[:, 0]).astype(np.int64)
    ys = np.rint(xys[:, 1]).astype(np.int64)
    if global_consistency_rejects is not None and inside.any():
        inside_indices = np.flatnonzero(inside)
        global_keep = np.asarray(
            [int(point_id) not in global_consistency_rejects for point_id in point_ids[inside]],
            dtype=bool,
        )
        stats["rejected_consistency"] = int((~global_keep).sum())
        inside[inside_indices[~global_keep]] = False
    if consistency_depth is not None and inside.any():
        inside_indices = np.flatnonzero(inside)
        reference_depth = sample_depth_map(consistency_depth, xys[inside], width, height)
        consistency_keep, invalid_reference = inverse_depth_consistency_mask(z[inside], reference_depth, args)
        stats["invalid_consistency_depth"] = invalid_reference
        stats["rejected_consistency"] = int((~consistency_keep).sum())
        inside[inside_indices[~consistency_keep]] = False
    stats["projected"] = int(inside.sum())
    if inside.any():
        flat = ys[inside] * width + xs[inside]
        np.minimum.at(depth.ravel(), flat, z[inside].astype(np.float32))

    depth[~np.isfinite(depth)] = 0.0
    stats["anchors"] = int((depth > 0).sum())
    return depth, stats


def generate_sparse_depth_maps(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    rgb_dir = Path(args.rgb_dir)
    out_dir = Path(args.out_dir)
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    cameras, images, points3d = read_colmap_model(model_dir, args.model_ext)
    rgb_by_stem = rgb_files(rgb_dir, parse_exts(args.image_ext))
    if not rgb_by_stem:
        raise RuntimeError(f"No RGB images found in {rgb_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    only_stems = parse_stems(args.only_stem)
    work_items: list[tuple[object, str, Path]] = []
    written = 0
    skipped = 0
    valid_pixels = 0
    total_stats = {
        "track_refs": 0,
        "missing_points": 0,
        "rejected_quality": 0,
        "rejected_consistency": 0,
        "invalid_consistency_depth": 0,
        "kept_points": 0,
        "outside_or_behind": 0,
        "projected": 0,
        "anchors": 0,
    }
    print(f"quality filter: {quality_filter_label(args)}")
    print(f"depth consistency filter: {consistency_filter_label(args)}")

    for image in sorted(images.values(), key=lambda item: item.name):
        image_name = Path(image.name).name
        stem = Path(image_name).stem
        if only_stems and stem not in only_stems:
            skipped += 1
            continue
        if stem not in rgb_by_stem:
            skipped += 1
            continue
        out_path = out_dir / f"{stem}.npy"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        work_items.append((image, stem, out_path))
        if args.limit and len(work_items) >= args.limit:
            break

    global_consistency_rejects = None
    if args.consistency_drop_point_all_views:
        global_consistency_rejects = find_global_consistency_rejects(work_items, cameras, points3d, args)
        print(f"global consistency rejected COLMAP points: {len(global_consistency_rejects)}")

    for image, stem, out_path in work_items:
        camera = cameras[image.camera_id]
        depth, stats = depth_for_image(image, stem, camera, points3d, args, global_consistency_rejects)
        for key, value in stats.items():
            total_stats[key] += value
        np.save(out_path, depth.astype(np.float32, copy=False))
        count = int((depth > 0).sum())
        valid_pixels += count
        written += 1
        if args.verbose:
            print(
                f"{stem}: wrote {out_path.name} ({depth.shape[1]}x{depth.shape[0]}, "
                f"{count} anchors; tracks={stats['track_refs']}, "
                f"quality_rejected={stats['rejected_quality']}, "
                f"consistency_rejected={stats['rejected_consistency']})"
            )

    print(f"wrote {written} sparse depth maps to {out_dir} ({valid_pixels} valid pixels total, skipped {skipped})")
    print(
        "point stats: "
        f"tracks={total_stats['track_refs']}, "
        f"quality_rejected={total_stats['rejected_quality']}, "
        f"consistency_rejected={total_stats['rejected_consistency']}, "
        f"consistency_invalid={total_stats['invalid_consistency_depth']}, "
        f"kept={total_stats['kept_points']}, "
        f"projected={total_stats['projected']}, "
        f"anchors={total_stats['anchors']}"
    )
    if global_consistency_rejects is not None:
        print(f"global consistency rejected unique points: {len(global_consistency_rejects)}")
