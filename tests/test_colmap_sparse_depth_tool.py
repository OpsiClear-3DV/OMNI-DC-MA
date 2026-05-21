import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "generate_colmap_sparse_depth.py"
    spec = importlib.util.spec_from_file_location("generate_colmap_sparse_depth", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "consistency_depth_dir": "ref",
        "max_inv_depth_diff": 0.0,
        "max_inv_depth_rel_diff": 0.25,
        "consistency_align_scale": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_inverse_depth_consistency_rejects_outliers_and_invalid_reference():
    tool = _load_tool()
    sfm_depth = np.array([2.0, 2.0, 4.0], dtype=np.float64)
    reference_depth = np.array([2.0, 1.0, 0.0], dtype=np.float64)

    keep, invalid_reference = tool._inverse_depth_consistency_mask(sfm_depth, reference_depth, _args())

    assert keep.tolist() == [True, False, False]
    assert invalid_reference == 1


def test_inverse_depth_consistency_can_align_reference_scale():
    tool = _load_tool()
    sfm_depth = np.array([2.0, 4.0], dtype=np.float64)
    reference_depth = np.array([4.0, 8.0], dtype=np.float64)

    keep_without_align, _ = tool._inverse_depth_consistency_mask(
        sfm_depth,
        reference_depth,
        _args(max_inv_depth_rel_diff=0.1),
    )
    keep_with_align, _ = tool._inverse_depth_consistency_mask(
        sfm_depth,
        reference_depth,
        _args(max_inv_depth_rel_diff=0.1, consistency_align_scale=True),
    )

    assert keep_without_align.tolist() == [False, False]
    assert keep_with_align.tolist() == [True, True]


def test_depth_map_sampling_supports_reference_maps_at_different_resolution():
    tool = _load_tool()
    depth_map = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    xys = np.array([[0.0, 0.0], [3.0, 3.0], [3.0, 0.0]], dtype=np.float64)

    sampled = tool._sample_depth_map(depth_map, xys, source_width=4, source_height=4)

    assert sampled.tolist() == [1.0, 4.0, 2.0]
