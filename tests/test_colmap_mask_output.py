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
