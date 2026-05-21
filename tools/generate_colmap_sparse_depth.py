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
INVERSE_DEPTH_EPS = 1e-6


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


def _consistency_enabled(args: argparse.Namespace) -> bool:
    return bool(args.consistency_depth_dir) and (
        args.max_inv_depth_diff > 0 or args.max_inv_depth_rel_diff > 0
    )


def _consistency_filter_label(args: argparse.Namespace) -> str:
    if not _consistency_enabled(args):
        return "off"
    checks: list[str] = []
    if args.max_inv_depth_diff > 0:
        checks.append(f"abs_inv_diff<={args.max_inv_depth_diff:g} 1/m")
    if args.max_inv_depth_rel_diff > 0:
        checks.append(f"rel_inv_diff<={args.max_inv_depth_rel_diff:g}")
    align = "median inverse-depth scale aligned" if args.consistency_align_scale else "no scale alignment"
    mode = (
        "drop failing COLMAP points in all selected views"
        if args.consistency_drop_point_all_views
        else "drop failing observations only"
    )
    return f"{', '.join(checks)} against {args.consistency_depth_dir} ({align}; {mode})"


def _normalize_depth_map(array: np.ndarray, path: Path) -> np.ndarray:
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


def _load_consistency_depth(stem: str, args: argparse.Namespace) -> np.ndarray | None:
    if not _consistency_enabled(args):
        return None
    path = Path(args.consistency_depth_dir) / f"{stem}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Consistency depth map not found for {stem}: {path}")
    return _normalize_depth_map(np.load(path), path)


def _sample_depth_map(depth_map: np.ndarray, xys: np.ndarray, source_width: int, source_height: int) -> np.ndarray:
    ref_height, ref_width = depth_map.shape
    scale_x = (ref_width - 1) / max(source_width - 1, 1)
    scale_y = (ref_height - 1) / max(source_height - 1, 1)
    xs = np.rint(xys[:, 0] * scale_x).astype(np.int64)
    ys = np.rint(xys[:, 1] * scale_y).astype(np.int64)
    xs = np.clip(xs, 0, ref_width - 1)
    ys = np.clip(ys, 0, ref_height - 1)
    return depth_map[ys, xs].astype(np.float64, copy=False)


def _aligned_ref_inverse_depth(inv_sfm: np.ndarray, inv_ref: np.ndarray, args: argparse.Namespace) -> np.ndarray:
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


def _inverse_depth_consistency_mask(
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
        inv_ref = _aligned_ref_inverse_depth(inv_sfm, inv_ref, args)
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


def _project_consistency_candidates(
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
        if not _point_is_certain(point, args):
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


def _find_global_consistency_rejects(
    work_items: list[tuple[object, str, Path]],
    cameras,
    points3d,
    args: argparse.Namespace,
) -> set[int]:
    rejected: set[int] = set()
    for image, stem, _out_path in work_items:
        camera = cameras[image.camera_id]
        point_ids, xys, z, inside, _stats = _project_consistency_candidates(image, camera, points3d, args)
        if not inside.any():
            continue
        consistency_depth = _load_consistency_depth(stem, args)
        inside_indices = np.flatnonzero(inside)
        reference_depth = _sample_depth_map(consistency_depth, xys[inside], int(camera.width), int(camera.height))
        consistency_keep, _invalid_reference = _inverse_depth_consistency_mask(z[inside], reference_depth, args)
        rejected.update(int(point_id) for point_id in point_ids[inside_indices[~consistency_keep]])
    return rejected


def _depth_for_image(
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
    consistency_depth = None if global_consistency_rejects is not None else _load_consistency_depth(stem, args)
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

    point_ids, xys, z, inside, candidate_stats = _project_consistency_candidates(
        image,
        camera,
        points3d,
        args,
    )
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
        reference_depth = _sample_depth_map(consistency_depth, xys[inside], width, height)
        consistency_keep, invalid_reference = _inverse_depth_consistency_mask(z[inside], reference_depth, args)
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
    print(f"quality filter: {_quality_filter_label(args)}")
    print(f"depth consistency filter: {_consistency_filter_label(args)}")

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
        global_consistency_rejects = _find_global_consistency_rejects(work_items, cameras, points3d, args)
        print(f"global consistency rejected COLMAP points: {len(global_consistency_rejects)}")

    for image, stem, out_path in work_items:
        camera = cameras[image.camera_id]
        depth, stats = _depth_for_image(image, stem, camera, points3d, args, global_consistency_rejects)
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

    print(
        f"wrote {written} sparse depth maps to {out_dir} "
        f"({valid_pixels} valid pixels total, skipped {skipped})"
    )
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
    parser.add_argument(
        "--consistency-depth-dir",
        help="Optional directory of reference dense depth .npy maps matched by RGB stem. "
             "When paired with an inverse-depth threshold, projected SfM points that disagree are rejected.",
    )
    parser.add_argument(
        "--max-inv-depth-diff",
        type=float,
        default=0.0,
        help="Reject points with absolute inverse-depth disagreement above this threshold in 1/m. "
             "Requires --consistency-depth-dir. 0 disables this check.",
    )
    parser.add_argument(
        "--max-inv-depth-rel-diff",
        type=float,
        default=0.0,
        help="Reject points with symmetric relative inverse-depth disagreement above this threshold. "
             "For example, 0.25 rejects points differing by more than about 25%%. "
             "Requires --consistency-depth-dir. 0 disables this check.",
    )
    parser.add_argument(
        "--consistency-align-scale",
        action="store_true",
        help="Median-align reference inverse depths to SfM inverse depths per image before consistency checks. "
             "Useful when the reference depth maps are only relatively scaled.",
    )
    parser.add_argument(
        "--consistency-drop-point-all-views",
        "--consistency-remove-point-all-views",
        action="store_true",
        dest="consistency_drop_point_all_views",
        help="If any observation of a COLMAP 3D point fails the depth consistency check, reject that point in all "
             "selected output views. By default, only the failing observation is removed.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print one line per written image")
    args = parser.parse_args()
    if args.min_track_length < 1:
        raise ValueError("--min-track-length must be >= 1")
    if args.max_reproj_error <= 0:
        raise ValueError("--max-reproj-error must be > 0")
    if args.max_inv_depth_diff < 0:
        raise ValueError("--max-inv-depth-diff must be >= 0")
    if args.max_inv_depth_rel_diff < 0:
        raise ValueError("--max-inv-depth-rel-diff must be >= 0")
    if (args.max_inv_depth_diff > 0 or args.max_inv_depth_rel_diff > 0) and not args.consistency_depth_dir:
        raise ValueError("inverse-depth consistency thresholds require --consistency-depth-dir")
    if args.consistency_drop_point_all_views and not _consistency_enabled(args):
        raise ValueError(
            "--consistency-drop-point-all-views requires --consistency-depth-dir and an inverse-depth threshold"
        )
    generate(args)


if __name__ == "__main__":
    main()
