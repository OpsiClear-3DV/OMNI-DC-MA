"""Stage breakdown of the steady-state OGNIDC forward.

Wraps the major pipeline stages with CUDA-synchronized timers (monkeypatch,
no model edits) and reports where the ~2.5 s/frame actually goes:
prior -> backbone -> [GRU update -> optim-layer CG solve -> convex upsample
-> DySPN]. "other" = forward total minus the sum (pre/post-processing,
percentile-normalize, cat, etc.).

Pass --trt to profile the explicit TensorRT engine path.
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
for sub in ("src", "src/model", "src/model/deformconv"):
    sys.path.insert(0, str(REPO / sub))
PROFILE_TRT = "--trt" in sys.argv[1:] or os.environ.get("PROFILE_TRT", os.environ.get("BENCH_TRT", "0")) in {
    "1", "true", "TRUE", "yes", "YES",
}
CONFIG_ARGV = [
    "prof", "--gpus", "0", "--load_dav2", "1", "--num_resolution", "3",
    "--multi_resolution_learnable_gradients_weights", "uniform", "--GRU_iters", "1",
    "--optim_layer_input_clamp", "1.0", "--depth_activation_format", "exp",
    "--whiten_sparse_depths", "1", "--gru_internal_whiten_method", "median",
    "--backbone_mode", "rgbd", "--pred_confidence_input", "1", "--max_depth", "300.0",
    "--data_normalize_median", "1",
]
if PROFILE_TRT:
    CONFIG_ARGV.append("--trt")
if cg_check_interval := os.environ.get("PROFILE_CG_CHECK_INTERVAL", os.environ.get("BENCH_CG_CHECK_INTERVAL", "")):
    CONFIG_ARGV.extend(["--cg_check_interval", cg_check_interval])
if cg_fixed_iters := os.environ.get("PROFILE_CG_FIXED_ITERS", os.environ.get("BENCH_CG_FIXED_ITERS", "")):
    CONFIG_ARGV.extend(["--cg_fixed_iters", cg_fixed_iters])
if anchor_cap_factor := os.environ.get("PROFILE_ANCHOR_CAP_FACTOR", os.environ.get("BENCH_ANCHOR_CAP_FACTOR", "")):
    CONFIG_ARGV.extend(["--anchor_cap_factor", anchor_cap_factor])
if prior_batch_size := os.environ.get("PROFILE_PRIOR_BATCH_SIZE", os.environ.get("BENCH_PRIOR_BATCH_SIZE", "")):
    CONFIG_ARGV.extend(["--prior_batch_size", prior_batch_size])
if os.environ.get("PROFILE_CAPTURABLE", os.environ.get("BENCH_CAPTURABLE", "0")) in {"1", "true", "TRUE", "yes", "YES"}:
    CONFIG_ARGV.append("--capturable_inference")
sys.argv = CONFIG_ARGV

import model.ognidc as og  # noqa: E402
from config import args  # noqa: E402
from model.infer import load_model, load_pair  # noqa: E402

BIC = Path(r"C:\Users\opsiclear\Desktop\Data_WS1\360_v2\bicycle")
RGB = BIC / "images_2" / "_DSC8679.JPG"
DEP = BIC / "omnidc_test" / "sparse_depth_all_images_2" / "_DSC8679.npy"
if not DEP.exists():
    DEP = BIC / "any2full_test" / "sparse_depth_x2" / "_DSC8679.npy"

T = defaultdict(float)
TOP_LEVEL_LABELS = {
    "prior (MA-depthmap)",
    "backbone (PVT/CBAM)",
    "GRU update_block",
    "optim-layer CG solve",
    "convex upsample",
    "DySPN (6 iters)",
}


def _sync_timed(label, fn):
    def wrapped(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        T[label] += time.perf_counter() - t0
        return r
    return wrapped


net = load_model(args)
if getattr(net.depth_module, "net", None) is None:
    net.depth_module._ensure_net(torch.device("cuda"))

# Patch the stages (instance.forward for submodules; module globals for the
# free function / autograd.Function).
net.depth_module.forward = _sync_timed("prior (MA-depthmap)", net.depth_module.forward)
net.backbone.forward = _sync_timed("backbone (PVT/CBAM)", net.backbone.forward)
net.update_block.forward = _sync_timed("GRU update_block", net.update_block.forward)
net.prop_layer.forward = _sync_timed("DySPN (6 iters)", net.prop_layer.forward)
og.upsample_depth = _sync_timed("convex upsample", og.upsample_depth)

prior_net = net.depth_module.net
prior_net.infer = _sync_timed("  prior infer wrapper", prior_net.infer)
prior_net.encoder.forward = _sync_timed("  prior encoder total", prior_net.encoder.forward)
prior_net.encoder.patch_encoder.forward = _sync_timed(
    "  prior patch_encoder", prior_net.encoder.patch_encoder.forward
)
prior_net.decoder.forward = _sync_timed("  prior decoder", prior_net.decoder.forward)
prior_net.head.forward = _sync_timed("  prior head", prior_net.head.forward)

_orig_optim = og.DepthGradOptimLayer.apply


class _TimedOptim:
    apply = staticmethod(_sync_timed("optim-layer CG solve", _orig_optim))


og.DepthGradOptimLayer = _TimedOptim

rgb, dep = load_pair(RGB, DEP)

# Optional low-res and batch path: mirrors tools/bench_inference.py.
ms = int(os.environ.get("PROFILE_MAX_SIZE", os.environ.get("BENCH_MAX_SIZE", "0")))
if ms and max(rgb.shape[-2:]) > ms:
    s = ms / max(rgb.shape[-2:])
    nh, nw = round(rgb.shape[-2] * s), round(rgb.shape[-1] * s)
    rgb = F.interpolate(rgb, size=(nh, nw), mode="bilinear", align_corners=False)
    d = torch.where(dep > 0, dep, torch.full_like(dep, 1e9))
    d = -F.max_pool2d(-d, kernel_size=int(round(1 / s)), stride=None, ceil_mode=True)
    d = F.interpolate(d, size=(nh, nw), mode="nearest")
    d[d > 1e8] = 0.0
    dep = d
    print(f"low-res input: longest side -> {ms} ({nh}x{nw})")

batch_size = int(os.environ.get("PROFILE_BATCH_SIZE", os.environ.get("BENCH_BATCH_SIZE", "1")))
if batch_size < 1:
    raise ValueError(f"PROFILE_BATCH_SIZE must be >= 1, got {batch_size}")
if batch_size > 1:
    rgb = rgb.repeat(batch_size, 1, 1, 1).contiguous()
    dep = dep.repeat(batch_size, 1, 1, 1).contiguous()
    print(f"batch input: repeated sample -> batch {batch_size}")
    if os.environ.get("PROFILE_MIX_EACH_PIXEL", os.environ.get("BENCH_MIX_EACH_PIXEL", "0")) in {
        "1", "true", "TRUE", "yes", "YES",
    }:
        delta = float(os.environ.get("PROFILE_MIX_EACH_DELTA", os.environ.get("BENCH_MIX_EACH_DELTA", "0")))
        if delta:
            for idx in range(1, batch_size):
                rgb[idx, 0, 0, 0] = rgb[idx, 0, 0, 0] + delta * idx
            print(f"batch input: every non-first sample differs by RGB delta {delta:g}")
        else:
            base = rgb[0, 0, 0, 0].clone()
            for idx in range(1, batch_size):
                val = base
                for _ in range(idx):
                    val = torch.nextafter(val, val + 1.0)
                rgb[idx, 0, 0, 0] = val
            print("batch input: every non-first sample differs by a distinct RGB ulp")
    elif os.environ.get("PROFILE_MIX_LAST_PIXEL", "0") in {"1", "true", "TRUE", "yes", "YES"}:
        rgb[-1, 0, 0, 0] = torch.nextafter(rgb[-1, 0, 0, 0], rgb[-1, 0, 0, 0] + 1.0)
        print("batch input: last sample differs by one RGB ulp")

_, _, H, W = rgb.shape
diviser = int(4 * 2 ** (args.num_resolution - 1))
H_pad = (-H) % diviser
W_pad = (-W) % diviser
if H_pad or W_pad:
    rgb = F.pad(rgb, (0, W_pad, 0, H_pad))
    dep = F.pad(dep, (0, W_pad, 0, H_pad))
H_full, W_full = rgb.shape[-2:]

K = torch.eye(3, device=rgb.device).reshape(1, 3, 3)
if batch_size > 1:
    K = K.repeat(batch_size, 1, 1).contiguous()
sample = {"rgb": rgb, "dep": dep, "K": K, "pattern": 0}

# Warmup (autotune) — not counted.
for _ in range(2):
    with torch.inference_mode():
        _ = net(sample)

N = 5
T.clear()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    with torch.inference_mode():
        _ = net(sample)
torch.cuda.synchronize()
total = (time.perf_counter() - t0) / N

print(
    f"\nsteady-state forward: {total:.3f} s/batch, "
    f"{total / batch_size:.3f} s/frame (mean of {N}, padded {H_full}x{W_full})\n"
)
named = sorted(T.items(), key=lambda kv: -kv[1])
acc = 0.0
for label, secs in named:
    per = secs / N
    if label in TOP_LEVEL_LABELS:
        acc += per
    print(f"  {label:24s} {per:7.3f} s   {100 * per / total:5.1f}%")
print(f"  {'other (norm/cat/io/pad)':24s} {total - acc:7.3f} s   {100 * (total - acc) / total:5.1f}%")
print(f"  {'TOTAL':24s} {total:7.3f} s   100.0%")
