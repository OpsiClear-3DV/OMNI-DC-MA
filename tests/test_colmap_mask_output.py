from types import SimpleNamespace

import numpy as np
from PIL import Image


def test_colmap_mask_path_uses_sibling_masks_dir(tmp_path):
    from demo import _colmap_mask_path

    image_dir = tmp_path / "scene" / "images"
    rgb_path = image_dir / "frame.jpg"
    args = SimpleNamespace(demo_rgb_dir=str(image_dir), demo_colmap_mask_dir=None)

    assert _colmap_mask_path(args, rgb_path) == tmp_path / "scene" / "masks" / "frame.jpg.png"


def test_colmap_mask_writes_colmap_semantics(tmp_path):
    from demo import _write_colmap_mask

    image_dir = tmp_path / "scene" / "images"
    image_dir.mkdir(parents=True)
    rgb_path = image_dir / "frame.jpg"
    Image.new("RGB", (2, 2), "white").save(rgb_path)
    args = SimpleNamespace(demo_rgb_dir=str(image_dir), demo_colmap_mask_dir=None)

    sky = np.array([[False, True], [False, False]])
    out_path = _write_colmap_mask(args, rgb_path, sky, sky.shape)

    assert out_path == tmp_path / "scene" / "masks" / "frame.jpg.png"
    mask = np.asarray(Image.open(out_path))
    assert mask.tolist() == [[255, 0], [255, 255]]


def test_no_sky_mask_disables_mask_outputs():
    from demo import _resolve_demo_output_options

    args = SimpleNamespace(
        demo_outputs="depth,raw,vis,skymask,colmap_mask",
        save_sky_mask=True,
        save_colmap_mask=True,
        sky_mask=False,
        apply_sky_mask_to_depth=None,
        anchor_cap_factor=1.25,
    )

    outputs, save_sky, save_colmap_mask, request_sky, apply_sky = _resolve_demo_output_options(args)

    assert outputs == {"depth", "raw", "vis"}
    assert not save_sky
    assert not save_colmap_mask
    assert not request_sky
    assert not apply_sky
    assert args.anchor_cap_factor == 0.0


def test_colmap_mask_can_export_without_applying_to_depth():
    from demo import _resolve_demo_output_options

    args = SimpleNamespace(
        demo_outputs="depth,vis",
        save_sky_mask=False,
        save_colmap_mask=True,
        sky_mask=None,
        apply_sky_mask_to_depth=False,
        anchor_cap_factor=1.25,
    )

    outputs, save_sky, save_colmap_mask, request_sky, apply_sky = _resolve_demo_output_options(args)

    assert outputs == {"depth", "vis", "colmap_mask"}
    assert not save_sky
    assert save_colmap_mask
    assert request_sky
    assert not apply_sky
