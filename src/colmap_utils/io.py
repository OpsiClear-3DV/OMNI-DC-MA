# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OMNI-DC-MA contributors

"""Small wrappers around the vendored COLMAP model reader/writer."""

from __future__ import annotations

from pathlib import Path

from robust_dc_protocol.read_write_colmap_model import (
    Camera,
    Image,
    Point3D,
    qvec2rotmat,
    read_model,
    write_model,
)


def is_colmap_model_dir(path: str | Path) -> bool:
    path = Path(path)
    has_bin = all((path / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin"))
    has_txt = all((path / name).is_file() for name in ("cameras.txt", "images.txt", "points3D.txt"))
    return has_bin or has_txt


def read_colmap_model(model_dir: str | Path, ext: str = ""):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"COLMAP model directory not found: {model_dir}")
    cameras, images, points3d = read_model(str(model_dir), ext)
    if cameras is None:
        raise RuntimeError(f"Could not read COLMAP model at {model_dir}")
    return cameras, images, points3d


def write_colmap_model(cameras, images, points3d, output_dir: str | Path, ext: str = ".bin"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return write_model(cameras, images, points3d, str(output_dir), ext)


__all__ = [
    "Camera",
    "Image",
    "Point3D",
    "is_colmap_model_dir",
    "qvec2rotmat",
    "read_colmap_model",
    "write_colmap_model",
]

