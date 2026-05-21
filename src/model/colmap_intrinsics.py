"""COLMAP focal-length lookup for MA-depthmap metric scaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robust_dc_protocol.read_write_colmap_model import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
)


@dataclass(frozen=True)
class ColmapFocal:
    image_name: str
    camera_model: str
    camera_width: int
    camera_height: int
    focal_px: float


def is_colmap_model_dir(path: str | Path) -> bool:
    path = Path(path)
    has_bin = (path / "cameras.bin").is_file() and (path / "images.bin").is_file()
    has_txt = (path / "cameras.txt").is_file() and (path / "images.txt").is_file()
    return has_bin or has_txt


def camera_focal_px(camera) -> float:
    """Return the horizontal focal length in COLMAP camera pixels."""
    params = np.asarray(camera.params, dtype=np.float64)
    if params.size == 0:
        raise ValueError(f"camera {getattr(camera, 'id', '?')} has no parameters")
    # MA's metric conversion uses image width / focal, so use fx for models
    # with separate fx/fy and f for the SIMPLE_* family.
    return float(params[0])


def _read_cameras_images(path: Path):
    if (path / "cameras.bin").is_file() and (path / "images.bin").is_file():
        return read_cameras_binary(str(path / "cameras.bin")), read_images_binary(str(path / "images.bin"))
    if (path / "cameras.txt").is_file() and (path / "images.txt").is_file():
        return read_cameras_text(str(path / "cameras.txt")), read_images_text(str(path / "images.txt"))
    raise FileNotFoundError(f"not a COLMAP model directory: {path}")


def _image_keys(image_name: str) -> tuple[str, ...]:
    normalized = image_name.replace("\\", "/")
    name = Path(normalized).name
    stem = Path(name).stem
    return tuple(dict.fromkeys((
        normalized.lower(),
        name.lower(),
        stem.lower(),
    )))


def load_colmap_focals(model_dir: str | Path) -> dict[str, ColmapFocal]:
    """Load per-image focal metadata from a COLMAP model directory."""
    model_dir = Path(model_dir)
    cameras, images = _read_cameras_images(model_dir)
    lookup: dict[str, ColmapFocal] = {}
    for image in images.values():
        camera = cameras[image.camera_id]
        focal = ColmapFocal(
            image_name=image.name,
            camera_model=camera.model,
            camera_width=int(camera.width),
            camera_height=int(camera.height),
            focal_px=camera_focal_px(camera),
        )
        for key in _image_keys(image.name):
            lookup.setdefault(key, focal)
    return lookup


def scaled_focal_for_image(
    lookup: dict[str, ColmapFocal] | None,
    image_path: str | Path,
    tensor_width: int,
) -> float | None:
    if not lookup:
        return None
    for key in _image_keys(str(image_path)):
        focal = lookup.get(key)
        if focal is not None:
            if focal.camera_width <= 0:
                return None
            return float(focal.focal_px) * (float(tensor_width) / float(focal.camera_width))
    return None


def _candidate_model_dirs(paths: list[str | Path]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(path)

    roots: list[Path] = []
    for raw in paths:
        p = Path(raw)
        root = p if p.is_dir() else p.parent
        roots.append(root)
        roots.extend(list(root.parents)[:4])

    for root in roots:
        add(root)
        add(root / "sparse")
        add(root / "sparse" / "0")
        add(root / "sparse" / "0_0")
        if root.is_dir():
            try:
                for child in root.iterdir():
                    if child.is_dir() and child.name.startswith("sparse"):
                        add(child)
                        add(child / "0")
                        add(child / "0_0")
            except OSError:
                pass
    return candidates


def resolve_colmap_focals(
    image_paths: list[str | Path],
    model_dir: str | Path | None = "auto",
) -> tuple[Path | None, dict[str, ColmapFocal] | None, int]:
    """Find and load the best COLMAP model for the given RGB image paths."""
    if model_dir is not None and str(model_dir).strip().lower() in {"", "none", "off", "false", "0"}:
        return None, None, 0

    explicit = model_dir is not None and str(model_dir).strip().lower() != "auto"
    candidates = [Path(model_dir)] if explicit else _candidate_model_dirs(image_paths)
    best_path: Path | None = None
    best_lookup: dict[str, ColmapFocal] | None = None
    best_matches = 0

    for candidate in candidates:
        if not is_colmap_model_dir(candidate):
            continue
        lookup = load_colmap_focals(candidate)
        matches = sum(
            1
            for image_path in image_paths
            if scaled_focal_for_image(lookup, image_path, 1) is not None
        )
        if explicit:
            return candidate, lookup, matches
        if matches > best_matches:
            best_path, best_lookup, best_matches = candidate, lookup, matches
            if matches == len(image_paths):
                break

    if explicit:
        raise FileNotFoundError(f"COLMAP model directory not found or incomplete: {model_dir}")
    return best_path, best_lookup, best_matches
