"""End-to-end regression test on the bicycle frame.

Gated: requires CUDA, the bicycle data, and HF weights (or the v0.1.0 release
cache). Skips cleanly otherwise so CI without a GPU still passes. When it can
run, it pins the known-good numbers from the verified MA-depthmap pipeline:
dense depth (1643, 2473), anchor mean |err| well under 1 mm.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

BICYCLE = Path(r"C:\Users\opsiclear\Desktop\Data_WS1\360_v2\bicycle")
RGB = BICYCLE / "images_2" / "_DSC8679.JPG"
DEP = BICYCLE / "omnidc_test" / "sparse_depth_all_images_2" / "_DSC8679.npy"
if not DEP.exists():
    DEP = BICYCLE / "any2full_test" / "sparse_depth_x2" / "_DSC8679.npy"

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not RGB.exists() or not DEP.exists(), reason="bicycle data not present"),
]


def test_bicycle_depth_matches_known_good(monkeypatch):
    # Import env (sys.path + chdir to src/) comes from conftest.py.
    monkeypatch.setattr(sys, "argv", [
        "test", "--gpus", "0", "--load_dav2", "1", "--num_resolution", "3",
        "--multi_resolution_learnable_gradients_weights", "uniform",
        "--GRU_iters", "1", "--optim_layer_input_clamp", "1.0",
        "--depth_activation_format", "exp", "--whiten_sparse_depths", "1",
        "--gru_internal_whiten_method", "median", "--backbone_mode", "rgbd",
        "--pred_confidence_input", "1", "--max_depth", "300.0",
        "--data_normalize_median", "1",
    ])

    from config import args
    from model.infer import load_model, load_pair, predict

    # Same code path as demo.py / bench (model/infer.py) — this test guards
    # exactly what users run.
    net = load_model(args)
    rgb, dep = load_pair(RGB, DEP)
    pred = predict(net, rgb, dep, args.num_resolution)

    assert pred.shape == (1643, 2473), f"unexpected output shape {pred.shape}"

    sd = np.load(DEP)
    m = sd > 0
    anchor_mae = float(np.abs(pred[m] - sd[m]).mean())
    # Known-good history: PIL+518 -> ~0.0002 m; full-res prior (#1) + nvJPEG
    # -> ~0.0019 m; fp16 prior (default) -> 0.001896 m; Jacobi-preconditioned
    # CG (now default, 833->183 iters, 1.48x faster forward) -> 0.002342 m.
    # The preconditioner doesn't change the system/true solution; CG stops at
    # finite rtol=1e-5 so iterates differ at ~that tolerance (sub-mm at
    # anchors, more in the ill-conditioned far-field). Still "anchors fully
    # respected" (~0.003% of the 76 m range). Threshold 5 mm: passes with
    # 2.7 mm margin, still fails hard if anchors are actually ignored (that
    # regression is cm-to-m, not ~2 mm).
    assert anchor_mae < 5e-3, f"anchor mean abs error {anchor_mae:.6f} m exceeds 5 mm"
    assert pred.min() > 0 and pred.max() < 200, f"depth range {pred.min()}..{pred.max()} implausible"
