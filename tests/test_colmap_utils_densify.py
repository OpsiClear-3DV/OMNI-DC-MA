from types import SimpleNamespace

import numpy as np


def _args(**overrides):
    values = {
        "cell_size": 8,
        "min_points_per_cell": 1,
        "points_per_cell_per_iteration": 1,
        "max_points_per_image": 0,
        "min_depth": 0.0,
        "max_depth": 0.0,
        "max_inv_depth_diff": 0.0,
        "max_inv_depth_rel_diff": 0.25,
        "consistency_align_scale": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _camera(width=16, height=16):
    from colmap_utils.io import Camera

    return Camera(
        id=1,
        model="PINHOLE",
        width=width,
        height=height,
        params=np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float64),
    )


def _image(point_ids=None, xys=None):
    from colmap_utils.io import Image

    point_ids = [] if point_ids is None else point_ids
    xys = np.empty((0, 2), dtype=np.float64) if xys is None else np.asarray(xys, dtype=np.float64)
    return Image(
        id=1,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        tvec=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        camera_id=1,
        name="frame.jpg",
        xys=xys,
        point3D_ids=np.asarray(point_ids, dtype=np.int64),
    )


def test_densify_adds_one_point_per_empty_cell():
    from colmap_utils.densify import densify_image_from_depth
    from colmap_utils.editing import validate_point_references

    images = {1: _image()}
    points3d = {}
    depth = np.full((16, 16), 2.0, dtype=np.float32)

    edited_images, edited_points, next_id, stats = densify_image_from_depth(
        images,
        points3d,
        image_id=1,
        camera=_camera(),
        depth_map=depth,
        rgb_path=None,
        args=_args(),
        next_point_id=1,
    )

    assert stats.added_points == 4
    assert next_id == 5
    assert len(edited_points) == 4
    assert edited_images[1].point3D_ids.tolist() == [1, 2, 3, 4]
    assert validate_point_references(edited_images, edited_points) == []


def test_densify_counts_existing_depth_consistent_observations():
    from colmap_utils.densify import densify_image_from_depth
    from colmap_utils.io import Point3D

    images = {1: _image(point_ids=[10], xys=[[3.0, 3.0]])}
    points3d = {
        10: Point3D(
            id=10,
            xyz=np.asarray([6.0, 6.0, 2.0], dtype=np.float64),
            rgb=np.asarray([255, 255, 255], dtype=np.uint8),
            error=0.1,
            image_ids=np.asarray([1], dtype=np.int32),
            point2D_idxs=np.asarray([0], dtype=np.int32),
        )
    }
    depth = np.full((16, 16), 2.0, dtype=np.float32)

    edited_images, edited_points, _next_id, stats = densify_image_from_depth(
        images,
        points3d,
        image_id=1,
        camera=_camera(),
        depth_map=depth,
        rgb_path=None,
        args=_args(),
        next_point_id=11,
    )

    assert stats.added_points == 3
    assert len(edited_points) == 4
    assert edited_images[1].point3D_ids.tolist()[0] == 10

