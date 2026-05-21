from __future__ import annotations

import numpy as np


def _image(image_id: int, point_ids: list[int]):
    from colmap_utils.io import Image

    return Image(
        id=image_id,
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.array([0.0, 0.0, 0.0]),
        camera_id=1,
        name=f"{image_id}.jpg",
        xys=np.zeros((len(point_ids), 2), dtype=np.float64),
        point3D_ids=np.asarray(point_ids, dtype=np.int64),
    )


def _point(point_id: int, observations: list[tuple[int, int]]):
    from colmap_utils.io import Point3D

    return Point3D(
        id=point_id,
        xyz=np.array([float(point_id), 0.0, 1.0]),
        rgb=np.array([255, 255, 255], dtype=np.uint8),
        error=0.1,
        image_ids=np.asarray([image_id for image_id, _idx in observations], dtype=np.int32),
        point2D_idxs=np.asarray([idx for _image_id, idx in observations], dtype=np.int32),
    )


def test_remove_points_clears_image_references():
    from colmap_utils.editing import remove_points, validate_point_references

    images = {
        1: _image(1, [10, 20, -1]),
        2: _image(2, [20, 30]),
    }
    points3d = {
        10: _point(10, [(1, 0)]),
        20: _point(20, [(1, 1), (2, 0)]),
        30: _point(30, [(2, 1)]),
    }

    edited_images, edited_points, stats = remove_points(images, points3d, [20])

    assert sorted(edited_points) == [10, 30]
    assert edited_images[1].point3D_ids.tolist() == [10, -1, -1]
    assert edited_images[2].point3D_ids.tolist() == [-1, 30]
    assert stats.removed_points == 1
    assert stats.removed_observations == 2
    assert validate_point_references(edited_images, edited_points) == []


def test_remove_observations_updates_point_tracks():
    from colmap_utils.editing import remove_observations

    images = {1: _image(1, [10, 20, -1])}
    points3d = {
        10: _point(10, [(1, 0)]),
        20: _point(20, [(1, 1)]),
    }

    edited_images, edited_points, stats = remove_observations(images, points3d, {1: [1, 2, 99]})

    assert edited_images[1].point3D_ids.tolist() == [10, -1, -1]
    assert sorted(edited_points) == [10]
    assert stats.removed_observations == 1
    assert stats.removed_points == 1
    assert stats.touched_images == 1


def test_add_tracked_point_updates_point_and_images():
    from colmap_utils.editing import add_tracked_point, validate_point_references

    images = {
        1: _image(1, [-1, -1]),
        2: _image(2, [-1]),
    }
    points3d = {}

    edited_images, edited_points, stats = add_tracked_point(
        images,
        points3d,
        point_id=42,
        xyz=[1.0, 2.0, 3.0],
        observations=[(1, 1), (2, 0)],
    )

    assert edited_images[1].point3D_ids.tolist() == [-1, 42]
    assert edited_images[2].point3D_ids.tolist() == [42]
    assert edited_points[42].image_ids.tolist() == [1, 2]
    assert edited_points[42].point2D_idxs.tolist() == [1, 0]
    assert stats.added_points == 1
    assert stats.added_observations == 2
    assert validate_point_references(edited_images, edited_points) == []
