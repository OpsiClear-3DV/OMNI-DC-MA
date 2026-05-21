# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OMNI-DC-MA contributors

"""Projection and unprojection helpers for COLMAP cameras."""

import numpy as np

from .io import qvec2rotmat

PINHOLE_MODELS = {"SIMPLE_PINHOLE", "PINHOLE"}
BROWN_MODELS = {"SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"}
SUPPORTED_UNPROJECT_MODELS = PINHOLE_MODELS | BROWN_MODELS


def _camera_params(camera):
    model = str(camera.model).upper()
    params = np.asarray(camera.params, dtype=np.float64)
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        return model, float(f), float(f), float(cx), float(cy), params
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        return model, float(fx), float(fy), float(cx), float(cy), params
    if model in {"SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy = params[:3]
        return model, float(f), float(f), float(cx), float(cy), params
    if model in {"OPENCV", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
        return model, float(fx), float(fy), float(cx), float(cy), params
    raise ValueError(f"unsupported camera model for depth unprojection: {camera.model}")


def _distort_normalized(model: str, x: np.ndarray, y: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if model in PINHOLE_MODELS:
        return x, y
    r2 = x * x + y * y
    if model == "SIMPLE_RADIAL":
        k1 = params[3]
        radial = 1.0 + k1 * r2
        return x * radial, y * radial
    if model == "RADIAL":
        k1, k2 = params[3:5]
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        return x * radial, y * radial
    if model == "OPENCV":
        k1, k2, p1, p2 = params[4:8]
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
    elif model == "FULL_OPENCV":
        k1, k2, p1, p2, k3, k4, k5, k6 = params[4:12]
        num = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        den = 1.0 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2
        radial = num / np.maximum(den, 1e-12)
    else:
        raise ValueError(f"unsupported camera model for distortion: {model}")
    x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_dist, y_dist


def image_to_camera_points(camera, xy: np.ndarray, depth: np.ndarray, *, iterations: int = 8) -> np.ndarray:
    """Unproject image pixels plus camera-z depth to camera coordinates."""
    xy = np.asarray(xy, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    model, fx, fy, cx, cy, params = _camera_params(camera)
    xd = (xy[:, 0] - cx) / fx
    yd = (xy[:, 1] - cy) / fy
    x = xd.copy()
    y = yd.copy()
    if model in BROWN_MODELS:
        for _ in range(iterations):
            px, py = _distort_normalized(model, x, y, params)
            x += xd - px
            y += yd - py
    return np.stack([x * depth, y * depth, depth], axis=1)


def camera_to_world(image, cam_xyz: np.ndarray) -> np.ndarray:
    """Transform COLMAP camera-coordinate points to world coordinates."""
    rot = qvec2rotmat(image.qvec)
    tvec = np.asarray(image.tvec, dtype=np.float64)
    return (np.asarray(cam_xyz, dtype=np.float64) - tvec) @ rot


def world_to_camera_depth(image, xyz: np.ndarray) -> np.ndarray:
    """Return camera-z depth for world-coordinate points in one image."""
    rot = qvec2rotmat(image.qvec)
    cam_xyz = np.asarray(xyz, dtype=np.float64) @ rot.T + np.asarray(image.tvec, dtype=np.float64)
    return cam_xyz[:, 2]


def unproject_depth_pixels_to_world(image, camera, xy: np.ndarray, depth: np.ndarray) -> np.ndarray:
    cam_xyz = image_to_camera_points(camera, xy, depth)
    return camera_to_world(image, cam_xyz)
