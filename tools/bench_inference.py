"""Time OMNI-DC inference end-to-end on the bicycle frame.

Reports model load time, first-pass warmup time, and steady-state per-frame
forward time over 5 timed runs, plus peak VRAM. Mirrors the args used by
testing_scripts/bicycle_demo.sh so the numbers reflect what you'd see in
practice on the same hardware.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make the OMNI-DC src/ subtree importable when this script is run from anywhere.
REPO = Path(__file__).resolve().parents[1]
for sub in ("src", "src/model", "src/model/deformconv"):
    sys.path.insert(0, str(REPO / sub))

# Mimic the exact CLI we use in bicycle_demo.sh so OGNIDC builds identically.
sys.argv = [
    "bench",
    "--gpus", "0",
    "--load_dav2", "1",
    "--num_resolution", "3",
    "--multi_resolution_learnable_gradients_weights", "uniform",
    "--GRU_iters", "1",
    "--optim_layer_input_clamp", "1.0",
    "--depth_activation_format", "exp",
    "--whiten_sparse_depths", "1",
    "--gru_internal_whiten_method", "median",
    "--backbone_mode", "rgbd",
    "--pred_confidence_input", "1",
    "--max_depth", "300.0",
    "--data_normalize_median", "1",
]
if os.environ.get("BENCH_TRT", "0") in {"1", "true", "TRUE", "yes", "YES"}:
    sys.argv.append("--trt")
if integration_alpha := os.environ.get("BENCH_INTEGRATION_ALPHA"):
    sys.argv.extend(["--integration_alpha", integration_alpha])
if optim_layer_input_clamp := os.environ.get("BENCH_OPTIM_LAYER_INPUT_CLAMP"):
    sys.argv.extend(["--optim_layer_input_clamp", optim_layer_input_clamp])
if pred_confidence_input := os.environ.get("BENCH_PRED_CONFIDENCE_INPUT"):
    sys.argv.extend(["--pred_confidence_input", pred_confidence_input])
if multi_res_input_weights := os.environ.get("BENCH_MULTI_RES_INPUT_WEIGHTS"):
    sys.argv.extend(["--multi_resolution_learnable_input_weights", multi_res_input_weights])
if prop_time := os.environ.get("BENCH_PROP_TIME"):
    sys.argv.extend(["--prop_time", prop_time])
if affinity_gamma := os.environ.get("BENCH_AFFINITY_GAMMA"):
    sys.argv.extend(["--affinity_gamma", affinity_gamma])
if conf_min := os.environ.get("BENCH_CONF_MIN"):
    sys.argv.extend(["--conf_min", conf_min])
if cg_check_interval := os.environ.get("BENCH_CG_CHECK_INTERVAL"):
    sys.argv.extend(["--cg_check_interval", cg_check_interval])
if cg_fixed_iters := os.environ.get("BENCH_CG_FIXED_ITERS"):
    sys.argv.extend(["--cg_fixed_iters", cg_fixed_iters])
if anchor_cap_factor := os.environ.get("BENCH_ANCHOR_CAP_FACTOR"):
    sys.argv.extend(["--anchor_cap_factor", anchor_cap_factor])
if prior_batch_size := os.environ.get("BENCH_PRIOR_BATCH_SIZE"):
    sys.argv.extend(["--prior_batch_size", prior_batch_size])
if os.environ.get("BENCH_PRIOR_REUSE_FIRST", "0") in {"1", "true", "TRUE", "yes", "YES"}:
    sys.argv.append("--prior_reuse_first_in_batch")
if os.environ.get("BENCH_CAPTURABLE", "0") in {"1", "true", "TRUE", "yes", "YES"}:
    sys.argv.append("--capturable_inference")
if os.environ.get("BENCH_CUDA_GRAPH", "0") in {"1", "true", "TRUE", "yes", "YES"}:
    sys.argv.append("--demo_cuda_graph")

from config import args  # noqa: E402
from model.final_reps import (  # noqa: E402
    interpolate_rep_predictions,
    select_final_rep_indices,
    select_final_rep_mode,
)
from model.infer import load_model, load_pair  # noqa: E402
from model.tensor_stats import quantile_02_98_flat  # noqa: E402

if os.environ.get("BENCH_SKIP_BACKBONE_TRT", "0") in {"1", "true", "TRUE", "yes", "YES"}:
    args.skip_backbone_trt = True

BICYCLE = Path(r"C:\Users\opsiclear\Desktop\Data_WS1\360_v2\bicycle")
RGB_PATH = BICYCLE / "images_2" / "_DSC8679.JPG"
DEP_PATH = BICYCLE / "omnidc_test" / "sparse_depth_all_images_2" / "_DSC8679.npy"
if not DEP_PATH.exists():
    DEP_PATH = BICYCLE / "any2full_test" / "sparse_depth_x2" / "_DSC8679.npy"


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) in {"1", "true", "TRUE", "yes", "YES"}


def _resize_pair(rgb: torch.Tensor, dep: torch.Tensor, max_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not max_size or max(rgb.shape[-2:]) <= max_size:
        return rgb, dep
    import torch.nn.functional as _F

    scale = max_size / max(rgb.shape[-2:])
    new_h, new_w = round(rgb.shape[-2] * scale), round(rgb.shape[-1] * scale)
    rgb = _F.interpolate(rgb, size=(new_h, new_w), mode="bilinear", align_corners=False)
    valid_depth = torch.where(dep > 0, dep, torch.full_like(dep, 1e9))
    valid_depth = -_F.max_pool2d(
        -valid_depth,
        kernel_size=int(round(1 / scale)),
        stride=None,
        ceil_mode=True,
    )
    valid_depth = _F.interpolate(valid_depth, size=(new_h, new_w), mode="nearest")
    valid_depth[valid_depth > 1e8] = 0.0
    return rgb, valid_depth


def _parse_rep_indices(batch_size: int) -> tuple[int, ...]:
    spec = os.environ.get("BENCH_FINAL_REP_INDICES", "0,3,6,10,15")
    reps = tuple(dict.fromkeys(int(item.strip()) for item in spec.split(",") if item.strip()))
    if not reps:
        raise ValueError("BENCH_FINAL_REP_INDICES must contain at least one index")
    if reps[0] < 0 or reps[-1] >= batch_size:
        raise ValueError(f"representative indices {reps} are outside batch size {batch_size}")
    return reps


def _build_final_rep_replay(
    net,
    rgb: torch.Tensor,
    dep: torch.Tensor,
    k: torch.Tensor,
    h: int,
    w: int,
    reps: tuple[int, ...],
    mode: str,
    grad_context,
):
    rep_count = len(reps)
    static_rgb = torch.empty_like(rgb[:rep_count])
    static_dep = torch.empty_like(dep[:rep_count])
    static_k = k[:rep_count].contiguous()
    sample = {"rgb": static_rgb, "dep": static_dep, "K": static_k, "pattern": 0}

    # The representative batch is intentionally distinct samples. Disable the
    # prior's exposure-reuse detector by default so it does not collapse the
    # reps before the final-output interpolation step. BENCH_FINAL_REP_PRIOR_REUSE
    # is an experiment switch for testing larger final-rep sets under the same
    # latency budget.
    depth_module = getattr(net, "depth_module", None)
    saved_auto_reuse = None
    allow_rep_prior_reuse = _truthy_env("BENCH_FINAL_REP_PRIOR_REUSE")
    if (
        not allow_rep_prior_reuse
        and depth_module is not None
        and hasattr(depth_module, "auto_reuse_identical_batch")
    ):
        saved_auto_reuse = depth_module.auto_reuse_identical_batch
        depth_module.auto_reuse_identical_batch = False

    try:
        static_rgb.copy_(rgb[list(reps)])
        static_dep.copy_(dep[list(reps)])
        with grad_context():
            _ = net(sample)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with grad_context(), torch.cuda.graph(graph):
            static_rep_pred = net(sample)["pred"][..., :h, :w]
            static_pred = interpolate_rep_predictions(static_rep_pred, reps, rgb.shape[0], mode)
        graph.replay()
        torch.cuda.synchronize()
    finally:
        if saved_auto_reuse is not None:
            depth_module.auto_reuse_identical_batch = saved_auto_reuse

    def replay_once() -> dict[str, torch.Tensor]:
        static_rgb.copy_(rgb[list(reps)])
        static_dep.copy_(dep[list(reps)])
        graph.replay()
        return {"pred": static_pred}

    return replay_once


def main() -> None:
    use_inference_mode = _truthy_env("BENCH_INFERENCE_MODE", "1")
    use_cuda_graph = _truthy_env("BENCH_CUDA_GRAPH")
    use_final_rep_interp = _truthy_env("BENCH_FINAL_REP_INTERP")
    grad_context = torch.inference_mode if use_inference_mode else torch.no_grad

    t0 = time.time()
    net = load_model(args)  # shared with demo.py — see model/infer.py
    t_load = time.time() - t0
    print(f"model load (HF + ckpt + to(cuda)):  {t_load:6.2f}s")

    batch_size = int(os.environ.get("BENCH_BATCH_SIZE", "1"))
    if batch_size < 1:
        raise ValueError(f"BENCH_BATCH_SIZE must be >= 1, got {batch_size}")
    ms = int(os.environ.get("BENCH_MAX_SIZE", "0"))

    # Optional low-res path: BENCH_MAX_SIZE=512 mirrors demo.py's --demo_max_size
    # (RGB bilinear; sparse = max-pool of valid depths so real anchors survive).
    if _truthy_env("BENCH_REAL16"):
        rgb_dir = Path(os.environ.get("BENCH_REAL_RGB_DIR", BICYCLE / "images_2"))
        dep_dir = Path(os.environ.get("BENCH_REAL_DEPTH_DIR", REPO / ".autotune" / "demo-real16-depth-links"))
        rgb_files = sorted(rgb_dir.glob("*.JPG"))[:batch_size]
        if len(rgb_files) != batch_size:
            raise RuntimeError(f"BENCH_REAL16 expected {batch_size} RGB files in {rgb_dir}, found {len(rgb_files)}")
        rgbs = []
        deps = []
        for rgb_path in rgb_files:
            dep_path = dep_dir / f"{rgb_path.stem}.npy"
            rgb_i, dep_i = load_pair(rgb_path, dep_path)
            rgb_i, dep_i = _resize_pair(rgb_i, dep_i, ms)
            rgbs.append(rgb_i)
            deps.append(dep_i)
        rgb = torch.cat(rgbs, dim=0).contiguous()
        dep = torch.cat(deps, dim=0).contiguous()
        print(
            f"real RGB batch: {batch_size} {rgb_files[0].name} ... {rgb_files[-1].name} "
            f"resized to {tuple(rgb.shape[-2:])}"
        )
    else:
        rgb, dep = load_pair(RGB_PATH, DEP_PATH)
        rgb, dep = _resize_pair(rgb, dep, ms)
        if ms:
            print(f"low-res input: longest side -> {ms} ({rgb.shape[-2]}x{rgb.shape[-1]})")

    if batch_size > 1 and not _truthy_env("BENCH_REAL16"):
        rgb = rgb.repeat(batch_size, 1, 1, 1).contiguous()
        dep = dep.repeat(batch_size, 1, 1, 1).contiguous()
        print(f"batch input: repeated sample -> batch {batch_size}")
        if os.environ.get("BENCH_MIX_ALTERNATE_PIXEL", "0") in {"1", "true", "TRUE", "yes", "YES"}:
            for idx in range(1, batch_size, 2):
                rgb[idx, 0, 0, 0] = torch.nextafter(rgb[idx, 0, 0, 0], rgb[idx, 0, 0, 0] + 1.0)
            print("batch input: alternating samples differ by one RGB ulp")
        elif os.environ.get("BENCH_MIX_FIRST_PIXEL", "0") in {"1", "true", "TRUE", "yes", "YES"}:
            rgb[0, 0, 0, 0] = torch.nextafter(rgb[0, 0, 0, 0], rgb[0, 0, 0, 0] + 1.0)
            print("batch input: first sample differs by one RGB ulp")
        elif os.environ.get("BENCH_MIX_EACH_PIXEL", "0") in {"1", "true", "TRUE", "yes", "YES"}:
            delta = float(os.environ.get("BENCH_MIX_EACH_DELTA", "0"))
            if delta:
                if os.environ.get("BENCH_MIX_GLOBAL_DELTA", "0") in {"1", "true", "TRUE", "yes", "YES"}:
                    for idx in range(1, batch_size):
                        rgb[idx] = rgb[idx] + delta * idx
                    print(f"batch input: every non-first sample differs by global RGB delta {delta:g}")
                else:
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
        elif os.environ.get("BENCH_MIX_LAST_PIXEL", "0") in {"1", "true", "TRUE", "yes", "YES"}:
            rgb[-1, 0, 0, 0] = torch.nextafter(rgb[-1, 0, 0, 0], rgb[-1, 0, 0, 0] + 1.0)
            print("batch input: last sample differs by one RGB ulp")

    K = torch.eye(3).reshape(1, 3, 3).cuda()
    if batch_size > 1:
        K = K.repeat(batch_size, 1, 1).contiguous()

    # The benchmark deliberately separates pad (setup) from the timed forward,
    # so the pad is inlined here rather than going through infer.predict()
    # (which also does the post-forward .cpu()/.numpy() sync we don't want
    # in the steady-state number).
    _, _, H, W = rgb.shape
    diviser = int(4 * 2 ** (args.num_resolution - 1))
    H_pad = (-H) % diviser
    W_pad = (-W) % diviser
    if H_pad or W_pad:
        rgb = torch.nn.functional.pad(rgb, (0, W_pad, 0, H_pad))
        dep = torch.nn.functional.pad(dep, (0, W_pad, 0, H_pad))
    H_full, W_full = rgb.shape[-2:]

    sample = {"rgb": rgb, "dep": dep, "K": K, "pattern": 0}
    precompute_mono = os.environ.get("BENCH_PRECOMPUTE_MONO", "0") in {
        "1", "true", "TRUE", "yes", "YES",
    }
    if precompute_mono:
        B = rgb.shape[0]
        max_metric_depth = None
        cap_factor = getattr(args, "anchor_cap_factor", 0.0)
        if cap_factor and cap_factor > 0:
            valid = dep > 0
            max_anchor = dep.masked_fill(~valid, 0.0).reshape(B, -1).amax(dim=1)
            max_metric_depth = torch.where(
                valid.reshape(B, -1).any(dim=1),
                cap_factor * max_anchor,
                torch.full_like(max_anchor, float("inf")),
            )
        depth_module = net.depth_module
        saved_trt_state = None
        force_eager_mono = os.environ.get("BENCH_PRECOMPUTE_MONO_EAGER", "0") in {
            "1", "true", "TRUE", "yes", "YES",
        }
        if force_eager_mono and getattr(depth_module, "_trt_requested", False):
            if getattr(depth_module, "_patch_trt_installed", False):
                raise RuntimeError("BENCH_PRECOMPUTE_MONO_EAGER must run before prior TRT is installed")
            saved_trt_state = (
                depth_module._trt_requested,
                depth_module._patch_trt_available,
                depth_module._full_prior_512_path_exists,
                depth_module._full_prior_512_selfcheck_cached,
                depth_module.full_prior_512,
            )
            depth_module._trt_requested = False
            depth_module._patch_trt_available = False
            depth_module._full_prior_512_path_exists = False
            depth_module._full_prior_512_selfcheck_cached = False
            depth_module.full_prior_512 = None
        try:
            with grad_context():
                prior_disp = depth_module.forward(rgb, max_metric_depth=max_metric_depth)
                depth_pred_raw = torch.relu(prior_disp.unsqueeze(1))
                depth_flat = depth_pred_raw.reshape(B, -1)
                q_min, q_max = quantile_02_98_flat(depth_flat)
                q_min = q_min.reshape(B, 1, 1, 1)
                q_max = q_max.reshape(B, 1, 1, 1)
                sample["mono_dep"] = ((depth_pred_raw - q_min) / (q_max - q_min).clamp_min(1e-6)).contiguous()
        finally:
            if saved_trt_state is not None:
                (
                    depth_module._trt_requested,
                    depth_module._patch_trt_available,
                    depth_module._full_prior_512_path_exists,
                    depth_module._full_prior_512_selfcheck_cached,
                    depth_module.full_prior_512,
                ) = saved_trt_state
        torch.cuda.synchronize()
        print(f"precomputed mono_dep: {tuple(sample['mono_dep'].shape)}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    graph = None
    output = None
    final_rep_replay = None
    if use_final_rep_interp:
        if not use_cuda_graph:
            raise ValueError("BENCH_FINAL_REP_INTERP requires BENCH_CUDA_GRAPH=1")
        reps = (
            _parse_rep_indices(batch_size)
            if os.environ.get("BENCH_FINAL_REP_INDICES")
            else select_final_rep_indices(rgb)
        )
        mode = os.environ.get("BENCH_FINAL_REP_MODE") or select_final_rep_mode(rgb, reps)
        final_rep_replay = _build_final_rep_replay(net, rgb, dep, K, H, W, reps, mode, grad_context)
        output = final_rep_replay()
        print(f"final-output representative graph: reps={reps} mode={mode}")
    elif use_cuda_graph:
        with grad_context():
            output = net(sample)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with grad_context(), torch.cuda.graph(graph):
            output = net(sample)
    else:
        with grad_context():
            output = net(sample)
    torch.cuda.synchronize()
    print(f"warmup forward (1st pass):          {time.time() - t0:6.2f}s/batch")

    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        if final_rep_replay is not None:
            output = final_rep_replay()
        elif graph is not None:
            graph.replay()
        else:
            with grad_context():
                output = net(sample)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    mean = sum(times) / len(times)
    std = (sum((x - mean) ** 2 for x in times) / len(times)) ** 0.5
    print(
        f"steady-state forward (n=5):         "
        f"mean={mean:.3f}s/batch +/- {std:.3f}s   min={min(times):.3f}s   "
        f"max={max(times):.3f}s   per-frame={mean / batch_size:.3f}s   "
        f"(image padded to {H_full}x{W_full})"
    )

    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"peak VRAM:                          {peak_gb:.2f} GB")

    pred_np = None
    if quality_ref := os.environ.get("BENCH_QUALITY_REF"):
        pred_np = output["pred"][..., :H, :W].detach().float().cpu().numpy()[:, 0]
        ref = np.load(quality_ref)
        diff = np.abs(pred_np - ref[None, :, :])
        print(
            "quality_vs_ref:                    "
            f"mean_abs={diff.mean():.6f}m   median={np.median(diff):.6f}m   "
            f"p95={np.percentile(diff, 95):.6f}m   max={diff.max():.6f}m"
        )
    if save_pred := os.environ.get("BENCH_SAVE_PRED"):
        if pred_np is None:
            pred_np = output["pred"][..., :H, :W].detach().float().cpu().numpy()[:, 0]
        np.save(save_pred, pred_np)
        print(f"saved_pred:                         {save_pred}")


if __name__ == "__main__":
    main()
