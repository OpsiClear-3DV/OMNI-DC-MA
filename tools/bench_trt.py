"""Honest eager-vs-TRT benchmark on the bicycle frame.

Builds the model twice (eager, then --trt), runs the first TRT forward so the
explicit engines can perform their one-time correctness self-checks, times the
steady-state forward, and reports the prediction delta.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for sub in ("src", "src/model", "src/model/deformconv"):
    sys.path.insert(0, str(REPO / sub))
_ARGV = [
    "bt", "--gpus", "0", "--load_dav2", "1", "--num_resolution", "3",
    "--multi_resolution_learnable_gradients_weights", "uniform", "--GRU_iters", "1",
    "--optim_layer_input_clamp", "1.0", "--depth_activation_format", "exp",
    "--whiten_sparse_depths", "1", "--gru_internal_whiten_method", "median",
    "--backbone_mode", "rgbd", "--pred_confidence_input", "1", "--max_depth", "300.0",
    "--data_normalize_median", "1",
]
sys.argv = list(_ARGV)
from config import args  # noqa: E402
from model.infer import load_model, load_pair, predict  # noqa: E402

BIC = Path(r"C:\Users\opsiclear\Desktop\Data_WS1\360_v2\bicycle")
RGB = BIC / "images_2" / "_DSC8679.JPG"
DEP = BIC / "omnidc_test" / "sparse_depth_all_images_2" / "_DSC8679.npy"
if not DEP.exists():
    DEP = BIC / "any2full_test" / "sparse_depth_x2" / "_DSC8679.npy"


def timed(net, rgb, dep, warmup, n):
    for _ in range(warmup):
        predict(net, rgb, dep, args.num_resolution)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(n):
        out = predict(net, rgb, dep, args.num_resolution)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n, out


rgb, dep = load_pair(RGB, DEP)
sparse = np.load(DEP).astype(np.float32)
m = sparse > 0

print("=== EAGER ===")
args.trt = False
net_e = load_model(args)
te, oe = timed(net_e, rgb, dep, warmup=2, n=5)
mae_e = float(np.abs(oe[m] - sparse[m]).mean())
print(f"eager forward: {te:.3f} s/frame   anchor MAE {mae_e:.6f} m")
del net_e
torch.cuda.empty_cache()

print("\n=== TRT (explicit prior + backbone decoder engines) ===")
args.trt = True
net_t = load_model(args)
# First forward includes the TensorRT correctness self-checks.
t_build0 = time.perf_counter()
predict(net_t, rgb, dep, args.num_resolution)
print(f"first forward (incl. TRT self-checks): {time.perf_counter() - t_build0:.1f} s")
tt, ot = timed(net_t, rgb, dep, warmup=3, n=5)
mae_t = float(np.abs(ot[m] - sparse[m]).mean())
print(f"trt forward:   {tt:.3f} s/frame   anchor MAE {mae_t:.6f} m")

d = np.abs(oe - ot)
print("\n=== verdict ===")
print(f"speedup:        {te / tt:.2f}x  ({te:.3f} -> {tt:.3f} s/frame)")
print(f"anchor MAE:     eager {mae_e:.6f} -> trt {mae_t:.6f} m  (delta {mae_t - mae_e:+.6f})")
print(f"pred vs eager:  mean|d|={d.mean():.4f} m  med={np.median(d):.4f} m  max={d.max():.2f} m")
