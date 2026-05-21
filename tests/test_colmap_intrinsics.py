from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_colmap_focal_scales_to_tensor_width():
    from model.colmap_intrinsics import ColmapFocal, camera_focal_px, scaled_focal_for_image

    camera = SimpleNamespace(
        id=1,
        model="SIMPLE_RADIAL",
        width=4946,
        height=3286,
        params=np.array([4647.934416051961, 2473.0, 1643.0, 0.0]),
    )
    assert camera_focal_px(camera) == camera.params[0]

    lookup = {
        "_dsc8679.jpg": ColmapFocal(
            image_name="_DSC8679.JPG",
            camera_model="SIMPLE_RADIAL",
            camera_width=4946,
            camera_height=3286,
            focal_px=camera_focal_px(camera),
        )
    }

    focal_512 = scaled_focal_for_image(lookup, Path("images_2") / "_DSC8679.JPG", 512)
    assert np.isclose(focal_512, 481.144848568258)


def test_colmap_intrinsics_can_be_disabled():
    from model.colmap_intrinsics import resolve_colmap_focals

    model_dir, lookup, matched = resolve_colmap_focals(["image.jpg"], model_dir="none")

    assert model_dir is None
    assert lookup is None
    assert matched == 0
