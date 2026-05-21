import os
import random
import time
from collections import defaultdict
from pathlib import Path

from config import args as args_config

os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port

import numpy as np
import torch

torch.autograd.set_detect_anomaly(bool(getattr(args_config, "debug_anomaly", False)))

from model.colmap_intrinsics import resolve_colmap_focals, scaled_focal_for_image
from model.final_reps import (
    interpolate_rep_predictions,
    is_validated_final_rep_batch_shape,
    select_final_rep_indices,
    select_final_rep_mode,
)
from model.infer import apply_anchor_cap, apply_sky_mask, load_model, load_pair, predict_tensor
from model.tensor_stats import quantile_02_98_flat

torch.backends.cudnn.deterministic = bool(getattr(args_config, "demo_deterministic", False))
torch.backends.cudnn.benchmark = bool(getattr(args_config, "demo_cudnn_benchmark", False))

_PROFILE_TIMES = defaultdict(float)
_PROFILE_COUNTS = defaultdict(int)


def _profile_begin(args):
    if not getattr(args, "demo_profile", False):
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _profile_end(args, label, t0, count=1):
    if t0 is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _PROFILE_TIMES[label] += time.perf_counter() - t0
    _PROFILE_COUNTS[label] += count


def _profile_print():
    if not _PROFILE_TIMES:
        return
    print("\n=== demo profile ===")
    total = sum(_PROFILE_TIMES.values())
    for label, secs in sorted(_PROFILE_TIMES.items(), key=lambda kv: -kv[1]):
        count = _PROFILE_COUNTS[label]
        per = secs / max(count, 1)
        print(f"{label:28s} {secs:8.3f}s  n={count:<3d}  avg={per:8.4f}s")
    print(f"{'profiled spans (nested)':28s} {total:8.3f}s")


def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), f"file not found: {args.pretrain}"

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            # new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume
            new_args.start_epoch = checkpoint['epoch'] + 1

    return new_args


def _apply_demo_runtime_defaults(args):
    if getattr(args, "demo_mono_cache_dir", None):
        args.prior_lazy_load = True
        args.skip_backbone_trt = True
        if getattr(args, "demo_input_cache_dir", None) is None:
            args.demo_input_cache_dir = str(Path(args.demo_mono_cache_dir) / "_inputs_v1")
    return args


def _file_sig(path):
    stat = Path(path).stat()
    return f"{stat.st_size:x}-{stat.st_mtime_ns:x}"


def _resolve_pairs(args):
    """Return one explicit pair, matched directory pairs, or the bundled demo pair."""
    import glob

    if args.demo_rgb and args.demo_depth:
        return [(args.demo_rgb, args.demo_depth)]
    if args.demo_rgb_dir and args.demo_depth_dir:
        rgb_files = []
        seen_rgb = set()
        for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
            for rgb_p in glob.glob(os.path.join(args.demo_rgb_dir, ext)):
                key = os.path.normcase(os.path.abspath(rgb_p))
                if key not in seen_rgb:
                    seen_rgb.add(key)
                    rgb_files.append(rgb_p)
        depth_map = {Path(p).stem: p for p in glob.glob(os.path.join(args.demo_depth_dir, "*.npy"))}
        pairs = [
            (rgb_p, depth_map[Path(rgb_p).stem])
            for rgb_p in sorted(rgb_files)
            if Path(rgb_p).stem in depth_map
        ]
        if not pairs:
            raise RuntimeError(
                f"No matching RGB+depth pairs found in {args.demo_rgb_dir} / "
                f"{args.demo_depth_dir}"
            )
        return pairs

    repo = Path(__file__).resolve().parents[1]
    return [(str(repo / "figures" / "demo_rgb.png"),
             str(repo / "figures" / "demo_sparse_depth.npy"))]


def _resolve_colmap_intrinsics(args, pairs):
    image_paths = [rgb_path for rgb_path, _depth_path in pairs]
    model_dir, lookup, matched = resolve_colmap_focals(
        image_paths,
        getattr(args, "demo_colmap_model_dir", "auto"),
    )
    if model_dir is None:
        print("COLMAP intrinsics: none found; MA uses f_px=0.6*width fallback")
        return None
    print(f"COLMAP intrinsics: {model_dir} ({matched}/{len(pairs)} RGB images matched)")
    if matched == 0:
        print("    no image names matched; MA uses f_px=0.6*width fallback")
    return lookup


def _input_cache_path(args, rgb_path, depth_path):
    cache_dir = getattr(args, "demo_input_cache_dir", None)
    if not cache_dir:
        return None
    ms = int(getattr(args, "demo_max_size", 0) or 0)
    name = (
        f"input_v1__{Path(rgb_path).stem}__{Path(depth_path).stem}__"
        f"max{ms}__r{_file_sig(rgb_path)}__d{_file_sig(depth_path)}.npz"
    )
    return Path(cache_dir) / name


def _prepare_pair(args, rgb_path, depth_path):
    cache_path = _input_cache_path(args, rgb_path, depth_path)
    if cache_path is not None and cache_path.exists():
        t0 = _profile_begin(args)
        with np.load(cache_path) as data:
            rgb = torch.from_numpy(data["rgb"]).cuda()
            dep = torch.from_numpy(data["dep"]).cuda()
            sparse = data["sparse"].astype(np.float32, copy=False)
        _profile_end(args, "input cache hit", t0)
        print(f"    input cache hit: {cache_path.name}")
        return rgb, dep, sparse

    t0 = _profile_begin(args)
    rgb, dep, sparse = load_pair(rgb_path, depth_path, return_sparse=True)
    _profile_end(args, "load_pair", t0)

    ms = getattr(args, "demo_max_size", 0)
    if ms and max(rgb.shape[-2:]) > ms:
        import torch.nn.functional as _F

        t0 = _profile_begin(args)
        s = ms / max(rgb.shape[-2:])
        nh, nw = round(rgb.shape[-2] * s), round(rgb.shape[-1] * s)
        rgb = _F.interpolate(rgb, size=(nh, nw), mode="bilinear", align_corners=False)

        d = torch.where(dep > 0, dep, torch.full_like(dep, 1e9))
        d = -_F.max_pool2d(-d, kernel_size=int(round(1 / s)), stride=None, ceil_mode=True)
        d = _F.interpolate(d, size=(nh, nw), mode="nearest")
        d[d > 1e8] = 0.0
        dep = d
        sparse = dep.squeeze().cpu().numpy()
        _profile_end(args, "resize inputs", t0)

    if cache_path is not None:
        t0 = _profile_begin(args)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            rgb=rgb.detach().float().cpu().numpy(),
            dep=dep.detach().float().cpu().numpy(),
            sparse=sparse.astype(np.float32, copy=False),
        )
        _profile_end(args, "input cache write", t0)
        print(f"    input cache miss -> wrote {cache_path.name}")
    return rgb, dep, sparse


def _infer_colmap_mask_root(args, rgb_path):
    explicit = getattr(args, "demo_colmap_mask_dir", None)
    if explicit:
        return Path(explicit)
    rgb_dir = getattr(args, "demo_rgb_dir", None)
    if rgb_dir:
        return Path(rgb_dir).parent / "masks"
    parent = Path(rgb_path).parent
    if parent.name.lower().startswith("images"):
        return parent.parent / "masks"
    return parent / "masks"


def _colmap_mask_path(args, rgb_path):
    mask_root = _infer_colmap_mask_root(args, rgb_path)
    rel = Path(rgb_path).name
    rgb_dir = getattr(args, "demo_rgb_dir", None)
    if rgb_dir:
        try:
            rel_path = Path(rgb_path).resolve().relative_to(Path(rgb_dir).resolve())
            rel = str(rel_path)
        except ValueError:
            rel = Path(rgb_path).name
    rel_path = Path(rel)
    return mask_root / rel_path.parent / f"{rel_path.name}.png"


def _write_colmap_mask(args, rgb_path, sky, depth_shape):
    """Write COLMAP mask_path-compatible PNG: white = keep, black = ignore."""
    from PIL import Image

    sky_mask = np.zeros(depth_shape, dtype=bool) if sky is None else np.asarray(sky) > 0.5
    colmap_mask = np.where(sky_mask, 0, 255).astype(np.uint8)
    img = Image.fromarray(colmap_mask, mode="L")
    with Image.open(rgb_path) as rgb_img:
        rgb_size = rgb_img.size
    if img.size != rgb_size:
        img = img.resize(rgb_size, Image.Resampling.NEAREST)
    path = _colmap_mask_path(args, rgb_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _write_outputs(args, outputs, cmap, save_sky, save_colmap_mask, out_dir, rgb_path, sparse, depth_raw, sky):
    if not np.isfinite(depth_raw).all():
        raise RuntimeError(
            f"non-finite depth for {rgb_path} (CG likely diverged - a "
            f"degenerate sparse pattern, often a bad --demo_max_size "
            f"ratio; 512 and full-res are validated). Aborting rather "
            f"than saving NaN."
        )

    t0 = _profile_begin(args)
    depth_pred, thr, n_capped = apply_anchor_cap(depth_raw, sparse, args.anchor_cap_factor)
    _profile_end(args, "anchor cap output", t0)
    print(f"    anchor cap @ {thr:.1f} m -> zeroed {n_capped} px "
          f"({100 * n_capped / depth_raw.size:.2f}%)")
    t0 = _profile_begin(args)
    depth_pred, n_sky_masked = apply_sky_mask(depth_pred, sky)
    _profile_end(args, "sky mask output", t0)
    if sky is not None:
        print(f"    prior sky/far mask -> zeroed {n_sky_masked} px "
              f"({100 * n_sky_masked / depth_raw.size:.2f}%)")

    stem = Path(rgb_path).stem
    capped_differs = n_capped > 0 or n_sky_masked > 0
    if "depth" in outputs:
        t0 = _profile_begin(args)
        np.save(out_dir / f"{stem}.npy", depth_pred)
        _profile_end(args, "save depth npy", t0)
    if "raw" in outputs and (capped_differs or "depth" not in outputs):
        t0 = _profile_begin(args)
        np.save(out_dir / f"{stem}_raw.npy", depth_raw)
        _profile_end(args, "save raw npy", t0)
    elif "raw" in outputs:
        (out_dir / f"{stem}_raw.npy").unlink(missing_ok=True)
        print(f"    (raw == capped, no px zeroed -> {stem}.npy is the raw; skipped dup)")

    if "vis" in outputs:
        t0 = _profile_begin(args)
        import matplotlib.pyplot as plt
        from PIL import Image

        valid = depth_pred > 0
        if valid.any():
            lo = np.percentile(depth_pred[valid], 5)
            hi = np.percentile(depth_pred[valid], 95)
        else:
            lo, hi = float(depth_pred.min()), float(depth_pred.max())
        norm = plt.Normalize(vmin=lo, vmax=hi)
        depth_colormap_uint8 = (cmap(norm(depth_pred)) * 255).astype(np.uint8)
        Image.fromarray(depth_colormap_uint8[..., :3]).save(out_dir / f"{stem}.png")
        _profile_end(args, "write vis png", t0)

    if save_sky:
        t0 = _profile_begin(args)
        from PIL import Image

        sky_mask = np.zeros_like(depth_pred, dtype=bool) if sky is None else np.asarray(sky) > 0.5
        Image.fromarray((sky_mask.astype(np.uint8) * 255), mode="L").save(
            out_dir / f"{Path(rgb_path).name}.png"
        )
        _profile_end(args, "write skymask png", t0)

    if save_colmap_mask:
        t0 = _profile_begin(args)
        mask_path = _write_colmap_mask(args, rgb_path, sky, depth_pred.shape)
        _profile_end(args, "write colmap mask", t0)
        print(f"    wrote COLMAP mask: {mask_path}")


def _prior_max_metric_depth(args, dep_b):
    cap_factor = getattr(args, "anchor_cap_factor", 0.0)
    if not cap_factor or cap_factor <= 0:
        return None
    valid = dep_b > 0
    max_anchor = dep_b.masked_fill(~valid, 0.0).reshape(dep_b.shape[0], -1).amax(dim=1)
    return torch.where(
        valid.reshape(dep_b.shape[0], -1).any(dim=1),
        cap_factor * max_anchor,
        torch.full_like(max_anchor, float("inf")),
    )


def _prior_reuse_graph_plan(net, args, rgb_b, dep_b, mono_b, f_px_b):
    """Return the prior branch shape that a CUDA Graph capture will bake in."""
    if mono_b is not None:
        return ("mono",)
    depth_module = getattr(net, "depth_module", None)
    if depth_module is None or rgb_b.shape[0] <= 1:
        return ("none",)

    rgb_in = rgb_b.half() if getattr(depth_module, "fp16", False) else rgb_b
    if getattr(depth_module, "reuse_first_in_batch", False):
        return ("forced-single", rgb_in.shape[0] // 2)
    if not getattr(depth_module, "auto_reuse_identical_batch", True):
        return ("none",)

    max_metric_depth = _prior_max_metric_depth(args, dep_b)
    caps_equal = True
    if max_metric_depth is not None:
        caps = max_metric_depth.reshape(-1)
        caps_equal = torch.equal(caps, caps[:1].expand_as(caps))
    f_equal = True
    if f_px_b is not None:
        f_vals = f_px_b.reshape(-1)
        f_equal = torch.equal(f_vals, f_vals[:1].expand_as(f_vals))

    exact_rgb = torch.equal(rgb_in, rgb_in[:1].expand_as(rgb_in))
    if exact_rgb and caps_equal and f_equal:
        return ("single", rgb_in.shape[0] // 2)

    endpoint_kind = depth_module._rgb_same_kind(rgb_in[0], rgb_in[-1])
    batch_kind = None if endpoint_kind is None else depth_module._rgb_same_kind(
        rgb_in,
        rgb_in[:1].expand_as(rgb_in),
    )
    if batch_kind is not None and caps_equal and f_equal:
        if "exposure" in {endpoint_kind, batch_kind}:
            reps = depth_module._even_representatives(
                rgb_in.shape[0],
                getattr(depth_module, "reuse_exposure_representatives", 1),
            )
            return ("reps", reps)
        return ("single", rgb_in.shape[0] // 2)

    groups = []
    reps = []
    for idx in range(rgb_in.shape[0]):
        for group_idx, rep_idx in enumerate(reps):
            same_f = True
            if f_px_b is not None:
                same_f = torch.equal(f_px_b[idx:idx + 1], f_px_b[rep_idx:rep_idx + 1])
            if same_f and depth_module._rgb_same(rgb_in[idx:idx + 1], rgb_in[rep_idx:rep_idx + 1]):
                groups[group_idx].append(idx)
                break
        else:
            reps.append(idx)
            groups.append([idx])
    duplicate_groups = tuple(tuple(group) for group in groups)
    if any(len(group) > 1 for group in duplicate_groups):
        return ("groups", duplicate_groups)
    return ("none",)


def _build_final_rep_graph_entry(net, args, rgb_b, dep_b, f_px_b, want_sky, reps, mode):
    rep_count = len(reps)
    static_rgb = torch.empty_like(rgb_b[:rep_count])
    static_dep = torch.empty_like(dep_b[:rep_count])
    static_f_px = torch.empty_like(f_px_b[:rep_count]) if f_px_b is not None else None

    depth_module = getattr(net, "depth_module", None)
    saved_auto_reuse = None
    if depth_module is not None and hasattr(depth_module, "auto_reuse_identical_batch"):
        saved_auto_reuse = depth_module.auto_reuse_identical_batch
        depth_module.auto_reuse_identical_batch = False

    try:
        static_rgb.copy_(rgb_b[list(reps)])
        static_dep.copy_(dep_b[list(reps)])
        if static_f_px is not None:
            static_f_px.copy_(f_px_b[list(reps)])
        with torch.inference_mode():
            if want_sky:
                predict_tensor(
                    net, static_rgb, static_dep, args.num_resolution,
                    return_sky_mask=True, f_px=static_f_px,
                )
            else:
                predict_tensor(net, static_rgb, static_dep, args.num_resolution, f_px=static_f_px)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            if want_sky:
                static_rep_depth, static_rep_sky = predict_tensor(
                    net, static_rgb, static_dep, args.num_resolution,
                    return_sky_mask=True, f_px=static_f_px,
                )
                static_depth = interpolate_rep_predictions(
                    static_rep_depth, reps, rgb_b.shape[0],
                    mode=mode,
                )
                static_sky = interpolate_rep_predictions(
                    static_rep_sky.float(), reps, rgb_b.shape[0],
                ) > 0.5
            else:
                static_rep_depth = predict_tensor(
                    net, static_rgb, static_dep, args.num_resolution, f_px=static_f_px
                )
                static_depth = interpolate_rep_predictions(
                    static_rep_depth, reps, rgb_b.shape[0],
                    mode=mode,
                )
                static_sky = None
        graph.replay()
        torch.cuda.synchronize()
    finally:
        if saved_auto_reuse is not None:
            depth_module.auto_reuse_identical_batch = saved_auto_reuse

    return {
        "graph": graph,
        "rgb": static_rgb,
        "dep": static_dep,
        "f_px": static_f_px,
        "depth": static_depth,
        "sky": static_sky,
        "reps": reps,
    }


def _run_final_rep_graph_entry(entry, rgb_b, dep_b, f_px_b):
    reps = entry["reps"]
    entry["rgb"].copy_(rgb_b[list(reps)])
    entry["dep"].copy_(dep_b[list(reps)])
    if entry["f_px"] is not None:
        entry["f_px"].copy_(f_px_b[list(reps)])
    entry["graph"].replay()

    if entry["sky"] is None:
        return entry["depth"], None
    return entry["depth"], entry["sky"]


def _mono_cache_path(args, rgb_path, depth_path, h, w, f_px):
    cache_dir = getattr(args, "demo_mono_cache_dir", None)
    if not cache_dir:
        return None
    cap = f"{float(getattr(args, 'anchor_cap_factor', 0.0)):.6g}".replace("-", "m").replace(".", "p")
    focal = "default" if f_px is None else f"{float(f_px.reshape(-1)[0]):.6g}".replace("-", "m").replace(".", "p")
    fill = getattr(args, "demo_mono_cache_fill", "eager")
    name = (
        f"mono_v4_{fill}__{Path(rgb_path).stem}__{Path(depth_path).stem}__"
        f"{h}x{w}__f{focal}__cap{cap}__r{_file_sig(rgb_path)}__d{_file_sig(depth_path)}.npy"
    )
    return Path(cache_dir) / name


def _compute_mono_dep(net, args, rgb, dep, f_px):
    cap_factor = getattr(args, "anchor_cap_factor", 0.0)
    max_metric_depth = None
    if cap_factor and cap_factor > 0:
        t0 = _profile_begin(args)
        valid = dep > 0
        max_anchor = dep.masked_fill(~valid, 0.0).reshape(dep.shape[0], -1).amax(dim=1)
        max_metric_depth = torch.where(
            valid.reshape(dep.shape[0], -1).any(dim=1),
            cap_factor * max_anchor,
            torch.full_like(max_anchor, float("inf")),
        )
        _profile_end(args, "mono cap prep", t0)

    depth_module = net.depth_module
    fill = getattr(args, "demo_mono_cache_fill", "eager")
    saved_trt_state = None
    if fill == "eager" and getattr(depth_module, "_trt_requested", False):
        if getattr(depth_module, "_patch_trt_installed", False):
            raise RuntimeError(
                "--demo_mono_cache_fill eager must run before prior TRT is installed. "
                "Use a fresh process or --demo_mono_cache_fill runtime."
            )
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
        with torch.inference_mode():
            t0 = _profile_begin(args)
            prior_disp = depth_module.forward(rgb, max_metric_depth=max_metric_depth, f_px=f_px)
            _profile_end(args, "mono prior forward", t0)
            t0 = _profile_begin(args)
            depth_pred_raw = torch.relu(prior_disp.unsqueeze(1))
            depth_flat = depth_pred_raw.reshape(rgb.shape[0], -1)
            q_min, q_max = quantile_02_98_flat(depth_flat)
            q_min = q_min.reshape(rgb.shape[0], 1, 1, 1)
            q_max = q_max.reshape(rgb.shape[0], 1, 1, 1)
            mono = ((depth_pred_raw - q_min) / (q_max - q_min).clamp_min(1e-6)).contiguous()
            _profile_end(args, "mono quantile norm", t0)
            return mono
    finally:
        if saved_trt_state is not None:
            (
                depth_module._trt_requested,
                depth_module._patch_trt_available,
                depth_module._full_prior_512_path_exists,
                depth_module._full_prior_512_selfcheck_cached,
                depth_module.full_prior_512,
            ) = saved_trt_state


def _load_or_compute_mono_dep(net, args, rgb_path, depth_path, rgb, dep, f_px):
    cache_path = _mono_cache_path(args, rgb_path, depth_path, rgb.shape[-2], rgb.shape[-1], f_px)
    if cache_path is None:
        return None

    if cache_path.exists():
        t0 = _profile_begin(args)
        mono_np = np.load(cache_path).astype(np.float32, copy=False)
        mono = torch.from_numpy(mono_np).to(device=rgb.device)
        if tuple(mono.shape) != (1, 1, rgb.shape[-2], rgb.shape[-1]):
            raise RuntimeError(f"bad mono cache shape in {cache_path}: {tuple(mono.shape)}")
        _profile_end(args, "mono cache hit", t0)
        print(f"    mono cache hit: {cache_path.name}")
        return mono

    if getattr(args, "demo_mono_cache_require_hit", False):
        raise FileNotFoundError(
            f"mono cache miss with --demo_mono_cache_require_hit: {cache_path}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mono = _compute_mono_dep(net, args, rgb, dep, f_px)
    t0 = _profile_begin(args)
    np.save(cache_path, mono.detach().float().cpu().numpy())
    _profile_end(args, "mono cache write", t0)
    print(f"    mono cache miss -> wrote {cache_path.name}")
    return mono


def _predict_batch_tensor(net, args, rgb_b, dep_b, mono_b, f_px_b, want_sky, graph_cache):
    use_graph = bool(getattr(args, "demo_cuda_graph", False))
    if not use_graph:
        if want_sky:
            return predict_tensor(
                net, rgb_b, dep_b, args.num_resolution,
                return_sky_mask=True, mono_dep=mono_b, f_px=f_px_b,
            )
        return predict_tensor(net, rgb_b, dep_b, args.num_resolution, mono_dep=mono_b, f_px=f_px_b), None

    if not getattr(args, "capturable_inference", False) or getattr(args, "cg_fixed_iters", 0) <= 0:
        raise ValueError("--demo_cuda_graph requires --capturable_inference and --cg_fixed_iters > 0")

    # The MA prior can auto-collapse or approximate a batch with fewer prior
    # calls. That changes the captured graph, so keep each prior branch plan in
    # a separate graph-cache entry. The plan mirrors the depth-module
    # predicates that decide the actual branch choice.
    prior_graph_plan = _prior_reuse_graph_plan(net, args, rgb_b, dep_b, mono_b, f_px_b)
    final_rep_reps = ()
    final_rep_mode = None
    # The calibrated final-output replay is only validated for B16 512-preview
    # exposure batches; other shapes use the stricter full-output graph.
    if (
        mono_b is None
        and prior_graph_plan[0] == "reps"
        and is_validated_final_rep_batch_shape(rgb_b.shape[0], rgb_b.shape[-2:])
    ):
        final_rep_reps = select_final_rep_indices(rgb_b)
        if len(final_rep_reps) < 2:
            final_rep_reps = ()
        else:
            final_rep_mode = select_final_rep_mode(rgb_b, final_rep_reps)
    has_mono = mono_b is not None
    mono_shape = None if mono_b is None else tuple(mono_b.shape)
    focal_shape = None if f_px_b is None else tuple(f_px_b.shape)
    key = (
        tuple(rgb_b.shape), tuple(dep_b.shape), mono_shape,
        focal_shape,
        bool(want_sky), prior_graph_plan,
        ("final-reps", final_rep_reps, final_rep_mode),
    )
    entry = graph_cache.get(key)
    if final_rep_reps:
        if entry is None:
            t0 = _profile_begin(args)
            entry = _build_final_rep_graph_entry(
                net, args, rgb_b, dep_b, f_px_b, want_sky, final_rep_reps, final_rep_mode
            )
            graph_cache[key] = entry
            depth, sky = _run_final_rep_graph_entry(entry, rgb_b, dep_b, f_px_b)
            _profile_end(args, "cuda graph build final reps", t0)
            print(f"    cuda graph final-output reps: {final_rep_reps} mode={final_rep_mode}")
            return depth, sky

        t0 = _profile_begin(args)
        depth, sky = _run_final_rep_graph_entry(entry, rgb_b, dep_b, f_px_b)
        _profile_end(args, "cuda graph replay final reps", t0)
        return depth, sky

    if entry is None:
        t0 = _profile_begin(args)
        static_rgb = torch.empty_like(rgb_b)
        static_dep = torch.empty_like(dep_b)
        static_mono = torch.empty_like(mono_b) if has_mono else None
        static_f_px = torch.empty_like(f_px_b) if f_px_b is not None else None
        static_rgb.copy_(rgb_b)
        static_dep.copy_(dep_b)
        if static_mono is not None:
            static_mono.copy_(mono_b)
        if static_f_px is not None:
            static_f_px.copy_(f_px_b)

        # One eager warmup pays TRT self-checks and allocator setup outside capture.
        with torch.inference_mode():
            if want_sky:
                predict_tensor(
                    net, static_rgb, static_dep, args.num_resolution,
                    return_sky_mask=True, mono_dep=static_mono, f_px=static_f_px,
                )
            else:
                predict_tensor(
                    net, static_rgb, static_dep, args.num_resolution,
                    mono_dep=static_mono, f_px=static_f_px,
                )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            if want_sky:
                static_depth, static_sky = predict_tensor(
                    net, static_rgb, static_dep, args.num_resolution,
                    return_sky_mask=True, mono_dep=static_mono, f_px=static_f_px,
                )
            else:
                static_depth = predict_tensor(
                    net, static_rgb, static_dep, args.num_resolution,
                    mono_dep=static_mono, f_px=static_f_px,
                )
                static_sky = None
        entry = {
            "graph": graph,
            "rgb": static_rgb,
            "dep": static_dep,
            "mono": static_mono,
            "f_px": static_f_px,
            "depth": static_depth,
            "sky": static_sky,
        }
        graph_cache[key] = entry
        entry["graph"].replay()
        _profile_end(args, "cuda graph build", t0)
    else:
        t0 = _profile_begin(args)
        entry["rgb"].copy_(rgb_b)
        entry["dep"].copy_(dep_b)
        if entry["mono"] is not None:
            entry["mono"].copy_(mono_b)
        if entry["f_px"] is not None:
            entry["f_px"].copy_(f_px_b)
        entry["graph"].replay()
        _profile_end(args, "cuda graph replay", t0)

    return entry["depth"], entry["sky"]


def _run_batch(
    net,
    args,
    pairs,
    outputs,
    cmap,
    request_sky,
    save_sky,
    save_colmap_mask,
    out_dir,
    start_idx,
    total,
    graph_cache,
    focal_lookup,
):
    import torch.nn.functional as _F

    prepared = []
    for offset, (rgb_path, depth_path) in enumerate(pairs, 1):
        print(f"[{start_idx + offset - 1}/{total}] {rgb_path} | {depth_path}")
        rgb, dep, sparse = _prepare_pair(args, rgb_path, depth_path)
        prepared.append((rgb_path, depth_path, rgb, dep, sparse, rgb.shape[-2], rgb.shape[-1]))

    t0 = _profile_begin(args)
    max_h = max(item[5] for item in prepared)
    max_w = max(item[6] for item in prepared)
    diviser = int(4 * 2 ** (args.num_resolution - 1))
    final_h = max_h + (-max_h) % diviser
    final_w = max_w + (-max_w) % diviser
    rgbs = []
    deps = []
    monos = []
    focal_values = []
    focal_matches = 0
    for rgb_path, depth_path, rgb, dep, _sparse, h, w in prepared:
        pad = (0, final_w - w, 0, final_h - h)
        rgb_p = _F.pad(rgb, pad)
        dep_p = _F.pad(dep, pad)
        f_px = scaled_focal_for_image(focal_lookup, rgb_path, final_w)
        if f_px is None:
            focal_values.append(0.6 * final_w)
            f_px_t = None
        else:
            focal_matches += 1
            focal_values.append(float(f_px))
            f_px_t = torch.tensor([float(f_px)], device=rgb_p.device, dtype=torch.float32)
        mono = _load_or_compute_mono_dep(net, args, rgb_path, depth_path, rgb_p, dep_p, f_px_t)
        rgbs.append(rgb_p)
        deps.append(dep_p)
        if mono is not None:
            monos.append(mono)

    rgb_b = torch.cat(rgbs, dim=0).contiguous()
    dep_b = torch.cat(deps, dim=0).contiguous()
    mono_b = torch.cat(monos, dim=0).contiguous() if monos else None
    f_px_b = None
    if focal_matches:
        f_px_b = torch.tensor(focal_values, device=rgb_b.device, dtype=torch.float32)
    _profile_end(args, "batch pad/cat", t0)

    t0 = _profile_begin(args)
    depth_t, sky_t = _predict_batch_tensor(net, args, rgb_b, dep_b, mono_b, f_px_b, request_sky, graph_cache)
    _profile_end(args, "predict batch", t0)

    for i, (rgb_path, _depth_path, _rgb, _dep, sparse, h, w) in enumerate(prepared):
        t0 = _profile_begin(args)
        depth_raw = depth_t[i, 0, :h, :w].cpu().numpy()
        sky = None if sky_t is None else sky_t[i, 0, :h, :w].cpu().numpy()
        _profile_end(args, "copy output to cpu", t0)
        _write_outputs(args, outputs, cmap, save_sky, save_colmap_mask, out_dir, rgb_path, sparse, depth_raw, sky)


def test(args):
    t0 = _profile_begin(args)
    net = load_model(args)
    _profile_end(args, "load model", t0)

    pairs = _resolve_pairs(args)
    focal_lookup = _resolve_colmap_intrinsics(args, pairs)
    out_dir = Path(args.demo_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    known_outputs = {"depth", "raw", "vis", "skymask", "colmap_mask"}
    outputs = {o.strip() for o in args.demo_outputs.split(",") if o.strip()}
    unknown = outputs - known_outputs
    if unknown:
        raise ValueError(
            f"--demo_outputs: unknown {sorted(unknown)}; valid: {sorted(known_outputs)}"
        )
    if not outputs:
        raise ValueError("--demo_outputs selected nothing to write")

    save_sky = "skymask" in outputs
    save_colmap_mask = "colmap_mask" in outputs
    request_sky = save_sky or save_colmap_mask or bool(getattr(args, "anchor_cap_factor", 0.0) > 0.0)
    if "vis" in outputs:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("jet")
    else:
        cmap = None
    batch_size = max(1, int(getattr(args, "demo_batch_size", 1)))
    num_batches = (len(pairs) + batch_size - 1) // batch_size
    if getattr(args, "demo_cuda_graph", False) and num_batches < 4:
        args.demo_cuda_graph = False
        print(
            "demo CUDA graph skipped: "
            f"{num_batches} batch(es) do not amortize capture"
        )
    graph_cache = {}

    for start in range(0, len(pairs), batch_size):
        _run_batch(net, args, pairs[start:start + batch_size], outputs, cmap, request_sky, save_sky, save_colmap_mask,
                   out_dir, start + 1, len(pairs), graph_cache, focal_lookup)
    if getattr(args, "demo_profile", False):
        _profile_print()


def main(args):
    init_seed()
    test(args)


if __name__ == "__main__":
    args_main = _apply_demo_runtime_defaults(check_args(args_config))

    print("\n\n=== Arguments ===")
    for cnt, key in enumerate(sorted(vars(args_main))):
        print(key, ":", getattr(args_main, key), end="  |  ")
        if (cnt + 1) % 5 == 0:
            print("")
    print("\n")

    main(args_main)
