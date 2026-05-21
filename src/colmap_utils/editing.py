# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OMNI-DC-MA contributors

"""COLMAP point-cloud editing primitives.

These helpers keep COLMAP's bidirectional point/observation references in sync:
``points3D`` owns tracks, while each image stores the point id for each 2D feature.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .io import Point3D


@dataclass(frozen=True)
class EditStats:
    removed_points: int = 0
    removed_observations: int = 0
    added_points: int = 0
    added_observations: int = 0
    touched_images: int = 0


def _replace_image_point_ids(image, point3d_ids: np.ndarray):
    ids = np.asarray(point3d_ids, dtype=np.int64)
    if hasattr(image, "_replace"):
        return image._replace(point3D_ids=ids)
    image.point3D_ids = ids
    return image


def _replace_point_track(point, image_ids: list[int], point2d_idxs: list[int]):
    if hasattr(point, "_replace"):
        return point._replace(
            image_ids=np.asarray(image_ids, dtype=np.int32),
            point2D_idxs=np.asarray(point2d_idxs, dtype=np.int32),
        )
    point.image_ids = np.asarray(image_ids, dtype=np.int32)
    point.point2D_idxs = np.asarray(point2d_idxs, dtype=np.int32)
    return point


def remove_observations(
    images: Mapping[int, object],
    points3d: Mapping[int, object],
    image_point_indices: Mapping[int, Iterable[int]],
):
    """Remove selected image observations and update affected point tracks."""
    edited_images = dict(images)
    removed_pairs: set[tuple[int, int]] = set()
    removed = 0
    touched = 0
    for image_id, indices in image_point_indices.items():
        if image_id not in edited_images:
            continue
        image = edited_images[image_id]
        point_ids = np.asarray(image.point3D_ids, dtype=np.int64).copy()
        valid_indices = sorted({int(idx) for idx in indices if 0 <= int(idx) < len(point_ids)})
        if not valid_indices:
            continue
        before = point_ids[valid_indices] != -1
        removed_pairs.update(
            (int(image_id), int(idx)) for idx, was_valid in zip(valid_indices, before, strict=False) if was_valid
        )
        point_ids[valid_indices] = -1
        count = int(before.sum())
        if count:
            removed += count
            touched += 1
            edited_images[image_id] = _replace_image_point_ids(image, point_ids)

    edited_points = {}
    removed_points = 0
    for point_id, point in points3d.items():
        kept_image_ids: list[int] = []
        kept_point2d_idxs: list[int] = []
        changed = False
        for track_image_id, point2d_idx in zip(point.image_ids, point.point2D_idxs, strict=False):
            pair = (int(track_image_id), int(point2d_idx))
            if pair in removed_pairs:
                changed = True
                continue
            kept_image_ids.append(pair[0])
            kept_point2d_idxs.append(pair[1])
        if not kept_image_ids and changed:
            removed_points += 1
            continue
        edited_points[point_id] = (
            _replace_point_track(point, kept_image_ids, kept_point2d_idxs) if changed else point
        )
    return edited_images, edited_points, EditStats(
        removed_points=removed_points,
        removed_observations=removed,
        touched_images=touched,
    )


def remove_points(images: Mapping[int, object], points3d: Mapping[int, object], point_ids: Iterable[int]):
    """Remove 3D points and clear every image observation that referenced them."""
    remove_ids = {int(point_id) for point_id in point_ids}
    edited_points = {pid: point for pid, point in points3d.items() if int(pid) not in remove_ids}
    edited_images = {}
    removed_observations = 0
    touched_images = 0
    for image_id, image in images.items():
        ids = np.asarray(image.point3D_ids, dtype=np.int64).copy()
        mask = np.isin(ids, list(remove_ids))
        if mask.any():
            removed_observations += int(mask.sum())
            touched_images += 1
            ids[mask] = -1
            edited_images[image_id] = _replace_image_point_ids(image, ids)
        else:
            edited_images[image_id] = image
    stats = EditStats(
        removed_points=len(points3d) - len(edited_points),
        removed_observations=removed_observations,
        touched_images=touched_images,
    )
    return edited_images, edited_points, stats


def add_tracked_point(
    images: Mapping[int, object],
    points3d: Mapping[int, object],
    *,
    point_id: int,
    xyz,
    rgb=(255, 255, 255),
    error: float = 0.0,
    observations: Iterable[tuple[int, int]],
):
    """Add one COLMAP 3D point and wire its image observations.

    ``observations`` is an iterable of ``(image_id, point2D_idx)`` pairs. The
    target 2D feature indices must already exist in the images.
    """
    point_id = int(point_id)
    if point_id in points3d:
        raise ValueError(f"point id already exists: {point_id}")
    edited_images = dict(images)
    image_ids: list[int] = []
    point2d_idxs: list[int] = []
    touched_images: set[int] = set()
    seen_observations: set[tuple[int, int]] = set()
    for image_id_raw, point2d_idx_raw in observations:
        image_id = int(image_id_raw)
        point2d_idx = int(point2d_idx_raw)
        observation = (image_id, point2d_idx)
        if observation in seen_observations:
            raise ValueError(f"duplicate observation for point {point_id}: {image_id}:{point2d_idx}")
        seen_observations.add(observation)
        if image_id not in edited_images:
            raise KeyError(f"image id not found: {image_id}")
        image = edited_images[image_id]
        ids = np.asarray(image.point3D_ids, dtype=np.int64).copy()
        if point2d_idx < 0 or point2d_idx >= len(ids):
            raise IndexError(f"point2D index {point2d_idx} out of range for image {image_id}")
        if ids[point2d_idx] != -1:
            raise ValueError(f"image {image_id} observation {point2d_idx} already references point {ids[point2d_idx]}")
        ids[point2d_idx] = point_id
        edited_images[image_id] = _replace_image_point_ids(image, ids)
        image_ids.append(image_id)
        point2d_idxs.append(point2d_idx)
        touched_images.add(image_id)

    if not image_ids:
        raise ValueError("tracked points require at least one observation")

    edited_points = dict(points3d)
    edited_points[point_id] = Point3D(
        id=point_id,
        xyz=np.asarray(xyz, dtype=np.float64),
        rgb=np.asarray(rgb, dtype=np.uint8),
        error=float(error),
        image_ids=np.asarray(image_ids, dtype=np.int32),
        point2D_idxs=np.asarray(point2d_idxs, dtype=np.int32),
    )
    stats = EditStats(added_points=1, added_observations=len(image_ids), touched_images=len(touched_images))
    return edited_images, edited_points, stats


def validate_point_references(images: Mapping[int, object], points3d: Mapping[int, object]) -> list[str]:
    """Return human-readable model consistency problems."""
    problems: list[str] = []
    point_ids = {int(pid) for pid in points3d}
    for image_id, image in images.items():
        ids = np.asarray(image.point3D_ids)
        missing = sorted({int(pid) for pid in ids if int(pid) != -1 and int(pid) not in point_ids})
        if missing:
            problems.append(f"image {image_id} references missing points: {missing[:10]}")
    for point_id, point in points3d.items():
        for image_id, point2d_idx in zip(point.image_ids, point.point2D_idxs, strict=False):
            image_id = int(image_id)
            point2d_idx = int(point2d_idx)
            image = images.get(image_id)
            if image is None:
                problems.append(f"point {point_id} references missing image {image_id}")
                continue
            ids = np.asarray(image.point3D_ids)
            if point2d_idx < 0 or point2d_idx >= len(ids):
                problems.append(f"point {point_id} has out-of-range observation {image_id}:{point2d_idx}")
            elif int(ids[point2d_idx]) != int(point_id):
                problems.append(f"point {point_id} track mismatch at image {image_id}:{point2d_idx}")
    return problems
