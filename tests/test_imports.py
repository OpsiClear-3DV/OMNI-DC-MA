"""Import-graph smoke test. No GPU, weights, or data required.

Catches the class of breakage an aggressive `ruff --fix` could introduce
(a removed import that had an import-time side effect, a broken relative
import, etc.). Runs in CI on every push. Path/CWD bootstrap lives in
conftest.py.
"""

import importlib

import numpy as np


def test_config_parses_defaults():
    import config

    assert hasattr(config.args, "num_resolution")
    assert hasattr(config.args, "demo_rgb")  # our added CLI flag
    assert config.args.depth_activation_format in ("exp", "linear")


def test_config_parses_intuitive_sky_cli_aliases():
    import config

    parsed = config.parser.parse_args([
        "--gpus", "0",
        "--far_depth_factor", "1.25",
        "--save_colmap_mask",
        "--no_apply_sky_mask",
    ])

    assert parsed.anchor_cap_factor == 1.25
    assert parsed.save_colmap_mask
    assert parsed.apply_sky_mask_to_depth is False


def test_ma_depthmap_prior_importable():
    # Importing must not require weights (those load lazily in __init__).
    from ma_depthmap import MADepthMapPrior
    from ma_depthmap.prior import MADepthMapPrior as P2

    assert MADepthMapPrior is P2
    assert MADepthMapPrior.__doc__ is not None  # documented public API
    assert callable(MADepthMapPrior.forward)


def test_ognidc_module_graph():
    # Exercises the sibling-import chain (backbone, convgru, optim_layer,
    # ma_depthmap) without instantiating CUDA modules.
    mod = importlib.import_module("model.ognidc")
    assert hasattr(mod, "OGNIDC")
    assert hasattr(mod, "upsample_depth")


def test_apply_anchor_cap():
    from model.infer import apply_anchor_cap

    depth = np.array([[1.0, 5.0, 100.0], [2.0, 50.0, 999.0]], dtype=np.float32)
    sparse = np.array([[0.0, 10.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32)  # max anchor = 20

    capped, thr, n = apply_anchor_cap(depth, sparse, factor=2.0)
    assert thr == 40.0  # 2 x 20
    assert n == 3       # 100, 50, 999 all exceed 40 m
    # > 40 m zeroed; <= 40 m kept; input untouched (pure)
    assert capped.tolist() == [[1.0, 5.0, 0.0], [2.0, 0.0, 0.0]]
    assert depth[1, 2] == 999.0  # original not mutated

    # factor <= 0 disables; no valid anchors disables (n_capped == 0)
    off, thr_off, n_off = apply_anchor_cap(depth, sparse, factor=0)
    assert thr_off == float("inf") and n_off == 0 and np.array_equal(off, depth)
    none, thr_none, n_none = apply_anchor_cap(depth, np.zeros_like(sparse), factor=2.0)
    assert thr_none == float("inf") and n_none == 0 and np.array_equal(none, depth)


def test_apply_sky_mask():
    from model.infer import apply_sky_mask

    depth = np.array([[1.0, 5.0, 0.0], [2.0, 50.0, 999.0]], dtype=np.float32)
    sky = np.array([[0.0, 0.7, 1.0], [0.5, 0.49, 1.0]], dtype=np.float32)

    masked, n = apply_sky_mask(depth, sky)
    assert n == 2  # 5 and 999; already-zero sky pixel is not double-counted
    assert masked.tolist() == [[1.0, 0.0, 0.0], [2.0, 50.0, 0.0]]
    assert depth[0, 1] == 5.0  # original not mutated

    unchanged, n_none = apply_sky_mask(depth, None)
    assert n_none == 0 and np.array_equal(unchanged, depth)
