# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OMNI-DC-MA contributors

"""Depth-guided point addition for COLMAP sparse models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from .editing import append_single_view_point
from .filters import inverse_depth_consistency_mask, normalize_depth_map, sample_depth_map
from .geometry import SUPPORTED_UNPROJECT_MODELS, unproject_depth_pixels_to_world, world_to_camera_depth
from .io import read_colmap_model, write_colmap_model
from .sparse_depth import parse_exts, parse_stems, rgb_files


@dataclass
class DensifyStats:
    images_seen: int = 0
    images_used: int = 0
    cells_seen: int = 0
    cells_underfilled: int = 0
    added_points: int = 0
    skipped_invalid_depth: int = 0
    skipped_depth_inconsistent: int = 0
    skipped_unsupported_camera: int = 0
    skipped_existing_limit: int = 0

    def add(self, other: DensifyStats) -> None:
        for key in self.__dataclass_fields__:
            setattr(self, key, getattr(self, key) + getattr(other, key))


def _depth_path_for_image(depth_dir: Path, image_name: str) -> Path:
    return depth_dir / f"{Path(image_name).stem}.npy"


def _valid_depth(depth: float, min_depth: float, max_depth: float) -> bool:
    if not np.isfinite(depth) or depth <= 0:
        return False
    if min_depth > 0 and depth < min_depth:
        return False
    return not (max_depth > 0 and depth > max_depth)


def _cell_depth_sample(
    depth_map: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    min_depth: float,
    max_depth: float,
    slot: int,
) -> tuple[float, float, float] | None:
    ref_h, ref_w = depth_map.shape
    sx = (ref_w - 1) / max(source_width - 1, 1)
    sy = (ref_h - 1) / max(source_height - 1, 1)
    rx0 = max(0, int(np.floor(x0 * sx)))
    ry0 = max(0, int(np.floor(y0 * sy)))
    rx1 = min(ref_w, int(np.ceil((x1 - 1) * sx)) + 1)
    ry1 = min(ref_h, int(np.ceil((y1 - 1) * sy)) + 1)
    if rx0 >= rx1 or ry0 >= ry1:
        return None
    crop = depth_map[ry0:ry1, rx0:rx1]
    valid = np.isfinite(crop) & (crop > 0)
    if min_depth > 0:
        valid &= crop >= min_depth
    if max_depth > 0:
        valid &= crop <= max_depth
    if not valid.any():
        return None

    offsets = (
        (0.5, 0.5),
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
        (0.5, 0.25),
        (0.5, 0.75),
        (0.25, 0.5),
        (0.75, 0.5),
    )
    ox, oy = offsets[slot % len(offsets)]
    target_x = x0 + ox * max(x1 - x0 - 1, 0)
    target_y = y0 + oy * max(y1 - y0 - 1, 0)
    target_rx = target_x * sx
    target_ry = target_y * sy
    valid_ys, valid_xs = np.nonzero(valid)
    dist2 = (valid_xs + rx0 - target_rx) ** 2 + (valid_ys + ry0 - target_ry) ** 2
    pick = int(np.argmin(dist2))
    ref_x = int(valid_xs[pick] + rx0)
    ref_y = int(valid_ys[pick] + ry0)
    u = ref_x / max(ref_w - 1, 1) * max(source_width - 1, 1)
    v = ref_y / max(ref_h - 1, 1) * max(source_height - 1, 1)
    return float(u), float(v), float(depth_map[ref_y, ref_x])


def _sample_rgb(rgb_path: Path | None, u: float, v: float, width: int, height: int) -> np.ndarray:
    if rgb_path is None:
        return np.asarray([255, 255, 255], dtype=np.uint8)
    with PILImage.open(rgb_path) as img:
        arr = np.asarray(img.convert("RGB"))
    scale_x = (arr.shape[1] - 1) / max(width - 1, 1)
    scale_y = (arr.shape[0] - 1) / max(height - 1, 1)
    x = int(np.clip(round(u * scale_x), 0, arr.shape[1] - 1))
    y = int(np.clip(round(v * scale_y), 0, arr.shape[0] - 1))
    return arr[y, x].astype(np.uint8, copy=False)


def _consistent_existing_observations(image, camera, points3d, depth_map: np.ndarray, args: argparse.Namespace):
    ids = np.asarray(image.point3D_ids)
    valid_obs = ids != -1
    if not valid_obs.any():
        return np.empty((0, 2), dtype=np.float64)
    obs_indices = np.flatnonzero(valid_obs)
    xys = np.asarray(image.xys, dtype=np.float64)[obs_indices]
    xyz = []
    keep_obs = []
    for obs_index, point_id in zip(obs_indices, ids[obs_indices], strict=False):
        point = points3d.get(int(point_id))
        if point is None:
            continue
        xyz.append(point.xyz)
        keep_obs.append(obs_index)
    if not xyz:
        return np.empty((0, 2), dtype=np.float64)
    z = world_to_camera_depth(image, np.asarray(xyz, dtype=np.float64))
    xys = np.asarray(image.xys, dtype=np.float64)[np.asarray(keep_obs, dtype=np.int64)]
    in_front = z > 0
    xys = xys[in_front]
    z = z[in_front]
    if len(z) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if args.max_inv_depth_diff <= 0 and args.max_inv_depth_rel_diff <= 0:
        return xys
    reference_depth = sample_depth_map(depth_map, xys, int(camera.width), int(camera.height))
    consistency_keep, _invalid = inverse_depth_consistency_mask(z, reference_depth, args)
    return xys[consistency_keep]


def densify_image_from_depth(
    images,
    points3d,
    *,
    image_id: int,
    camera,
    depth_map: np.ndarray,
    rgb_path: Path | None,
    args: argparse.Namespace,
    next_point_id: int,
):
    image = images[image_id]
    stats = DensifyStats(images_seen=1)
    if str(camera.model).upper() not in SUPPORTED_UNPROJECT_MODELS:
        stats.skipped_unsupported_camera = 1
        return images, points3d, next_point_id, stats

    height = int(camera.height)
    width = int(camera.width)
    existing_xys = _consistent_existing_observations(image, camera, points3d, depth_map, args)
    cell = int(args.cell_size)
    max_per_image = int(args.max_points_per_image)
    added_this_image = 0
    edited_images = images
    edited_points = points3d

    for y0 in range(0, height, cell):
        y1 = min(y0 + cell, height)
        for x0 in range(0, width, cell):
            x1 = min(x0 + cell, width)
            stats.cells_seen += 1
            in_cell = (
                (existing_xys[:, 0] >= x0)
                & (existing_xys[:, 0] < x1)
                & (existing_xys[:, 1] >= y0)
                & (existing_xys[:, 1] < y1)
            )
            count = int(in_cell.sum())
            needed = max(0, int(args.min_points_per_cell) - count)
            if needed == 0:
                continue
            stats.cells_underfilled += 1
            for slot in range(min(needed, int(args.points_per_cell_per_iteration))):
                if max_per_image > 0 and added_this_image >= max_per_image:
                    stats.skipped_existing_limit += 1
                    break
                sample = _cell_depth_sample(
                    depth_map,
                    source_width=width,
                    source_height=height,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    min_depth=float(args.min_depth),
                    max_depth=float(args.max_depth),
                    slot=count + slot,
                )
                if sample is None:
                    stats.skipped_invalid_depth += 1
                    continue
                u, v, z = sample
                if not _valid_depth(z, float(args.min_depth), float(args.max_depth)):
                    stats.skipped_invalid_depth += 1
                    continue
                xy = np.asarray([[u, v]], dtype=np.float64)
                xyz = unproject_depth_pixels_to_world(image, camera, xy, np.asarray([z], dtype=np.float64))[0]
                rgb = _sample_rgb(rgb_path, u, v, width, height)
                edited_images, edited_points, add_stats = append_single_view_point(
                    edited_images,
                    edited_points,
                    image_id=image_id,
                    point_id=next_point_id,
                    xy=[u, v],
                    xyz=xyz,
                    rgb=rgb,
                    error=0.0,
                )
                next_point_id += 1
                added_this_image += add_stats.added_points
                stats.added_points += add_stats.added_points
                existing_xys = np.vstack([existing_xys, xy])
    if added_this_image:
        stats.images_used = 1
    return edited_images, edited_points, next_point_id, stats


def densify_model_from_depth(args: argparse.Namespace) -> DensifyStats:
    cameras, images, points3d = read_colmap_model(args.model_dir, args.model_ext)
    output_dir = Path(args.output_model_dir)
    if not args.dry_run and output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output model directory is not empty: {output_dir} (use --overwrite)")

    depth_dir = Path(args.depth_dir)
    rgb_by_stem = {}
    if args.rgb_dir:
        rgb_by_stem = rgb_files(Path(args.rgb_dir), parse_exts(args.image_ext))
    only_stems = parse_stems(args.only_stem)
    next_point_id = max([0, *[int(pid) for pid in points3d]]) + 1
    total = DensifyStats()

    for _iteration in range(max(1, int(args.iterations))):
        iteration_stats = DensifyStats()
        for image_id, image in sorted(images.items(), key=lambda item: item[1].name):
            stem = Path(Path(image.name).name).stem
            if only_stems and stem not in only_stems:
                continue
            depth_path = _depth_path_for_image(depth_dir, image.name)
            if not depth_path.exists():
                continue
            depth_map = normalize_depth_map(np.load(depth_path), depth_path)
            rgb_path = rgb_by_stem.get(stem) if rgb_by_stem else None
            images, points3d, next_point_id, stats = densify_image_from_depth(
                images,
                points3d,
                image_id=int(image_id),
                camera=cameras[image.camera_id],
                depth_map=depth_map,
                rgb_path=rgb_path,
                args=args,
                next_point_id=next_point_id,
            )
            iteration_stats.add(stats)
        total.add(iteration_stats)
        if iteration_stats.added_points == 0:
            break

    if not args.dry_run:
        write_colmap_model(cameras, images, points3d, output_dir, args.output_model_ext)
    return total
