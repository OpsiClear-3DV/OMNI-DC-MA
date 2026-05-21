# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OMNI-DC-MA contributors

"""Generate OMNI-DC sparse-depth .npy files from a COLMAP sparse model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "robust_dc_protocol"))

from read_write_colmap_model import qvec2rotmat, read_model  # noqa: E402

DEFAULT_IMAGE_EXTS = (".JPG", ".jpg", ".JPEG", ".jpeg", ".png", ".PNG")
DEFAULT_MIN_TRACK_LENGTH = 3
DEFAULT_MAX_REPROJ_ERROR = 2.0


def _parse_exts(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_IMAGE_EXTS
    exts: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            exts.append(item if item.startswith(".") else f".{item}")
    return tuple(dict.fromkeys(exts))


def _parse_stems(values: list[str] | None) -> set[str]:
    stems: list[str] = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                stems.append(Path(item).stem)
    return set(stems)


def _rgb_files(rgb_dir: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for ext in exts:
        for path in rgb_dir.glob(f"*{ext}"):
            files.setdefault(path.stem, path)
    return files


def _track_length(point) -> int:
    return int(len(point.image_ids))


def _reprojection_error(point) -> float:
    return float(point.error)


def _point_is_certain(point, args: argparse.Namespace) -> bool:
    if args.no_quality_filter:
        return True
    return (
        _track_length(point) >= args.min_track_length
        and _reprojection_error(point) <= args.max_reproj_error
    )


def _quality_filter_label(args: argparse.Namespace) -> str:
    if args.no_quality_filter:
        return "off"
    return f"track_length>={args.min_track_length}, reprojection_error<={args.max_reproj_error:g}px"


def _depth_for_image(image, camera, points3d, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, int]]:
    height = int(camera.height)
    width = int(camera.width)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    stats = {
        "track_refs": 0,
        "missing_points": 0,
        "rejected_quality": 0,
        "kept_points": 0,
        "outside_or_behind": 0,
        "projected": 0,
        "anchors": 0,
    }

    point_ids = np.asarray(image.point3D_ids)
    valid_track = point_ids != -1
    stats["track_refs"] = int(valid_track.sum())
    if not valid_track.any():
        depth[~np.isfinite(depth)] = 0.0
        return depth, stats

    ids = point_ids[valid_track]
    xys = np.asarray(image.xys, dtype=np.float64)[valid_track]
    xyz = []
    keep = []
    for idx, point_id in enumerate(ids):
        point = points3d.get(int(point_id))
        if point is None:
            stats["missing_points"] += 1
            continue
        if not _point_is_certain(point, args):
            stats["rejected_quality"] += 1
            continue
        xyz.append(point.xyz)
        keep.append(idx)
    stats["kept_points"] = len(xyz)
    if not xyz:
        depth[~np.isfinite(depth)] = 0.0
        return depth, stats

    xyz_arr = np.asarray(xyz, dtype=np.float64)
    xys = xys[np.asarray(keep, dtype=np.int64)]
    rot = qvec2rotmat(image.qvec)
    cam_xyz = xyz_arr @ rot.T + np.asarray(image.tvec, dtype=np.float64)
    z = cam_xyz[:, 2]

    xs = np.rint(xys[:, 0]).astype(np.int64)
    ys = np.rint(xys[:, 1]).astype(np.int64)
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height) & (z > 0)
    stats["projected"] = int(inside.sum())
    stats["outside_or_behind"] = int(len(inside) - stats["projected"])
    if inside.any():
        flat = ys[inside] * width + xs[inside]
        np.minimum.at(depth.ravel(), flat, z[inside].astype(np.float32))

    depth[~np.isfinite(depth)] = 0.0
    stats["anchors"] = int((depth > 0).sum())
    return depth, stats


def generate(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    rgb_dir = Path(args.rgb_dir)
    out_dir = Path(args.out_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"COLMAP model directory not found: {model_dir}")
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    cameras, images, points3d = read_model(str(model_dir), args.model_ext)
    if cameras is None:
        raise RuntimeError(f"Could not read COLMAP model at {model_dir}")

    rgb_by_stem = _rgb_files(rgb_dir, _parse_exts(args.image_ext))
    if not rgb_by_stem:
        raise RuntimeError(f"No RGB images found in {rgb_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    only_stems = _parse_stems(args.only_stem)
    written = 0
    skipped = 0
    valid_pixels = 0
    total_stats = {
        "track_refs": 0,
        "missing_points": 0,
        "rejected_quality": 0,
        "kept_points": 0,
        "outside_or_behind": 0,
        "projected": 0,
        "anchors": 0,
    }
    print(f"quality filter: {_quality_filter_label(args)}")

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

        camera = cameras[image.camera_id]
        depth, stats = _depth_for_image(image, camera, points3d, args)
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
                f"quality_rejected={stats['rejected_quality']})"
            )
        if args.limit and written >= args.limit:
            break

    print(
        f"wrote {written} sparse depth maps to {out_dir} "
        f"({valid_pixels} valid pixels total, skipped {skipped})"
    )
    print(
        "point stats: "
        f"tracks={total_stats['track_refs']}, "
        f"quality_rejected={total_stats['rejected_quality']}, "
        f"kept={total_stats['kept_points']}, "
        f"projected={total_stats['projected']}, "
        f"anchors={total_stats['anchors']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="COLMAP sparse model directory, e.g. sparse/0")
    parser.add_argument("--rgb-dir", required=True, help="Directory containing RGB images to match by stem")
    parser.add_argument("--out-dir", required=True, help="Output directory for .npy sparse depth maps")
    parser.add_argument("--model-ext", default="", choices=("", ".bin", ".txt"), help="COLMAP model format")
    parser.add_argument(
        "--image-ext",
        action="append",
        help="Image extension to include. May be repeated or comma-separated. Defaults to common JPG/PNG forms.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy outputs")
    parser.add_argument("--only-stem", action="append", help="Only process matching RGB stems. May be repeated.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after writing this many files. 0 means no limit.")
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=DEFAULT_MIN_TRACK_LENGTH,
        help="Require COLMAP 3D points to be observed in at least this many images. Default: %(default)s.",
    )
    parser.add_argument(
        "--max-reproj-error",
        type=float,
        default=DEFAULT_MAX_REPROJ_ERROR,
        help="Require COLMAP point reprojection error at or below this many pixels. Default: %(default)s.",
    )
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable the default certain-point filter. Intended only for comparison/debugging.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print one line per written image")
    args = parser.parse_args()
    if args.min_track_length < 1:
        raise ValueError("--min-track-length must be >= 1")
    if args.max_reproj_error <= 0:
        raise ValueError("--max-reproj-error must be > 0")
    generate(args)


if __name__ == "__main__":
    main()
