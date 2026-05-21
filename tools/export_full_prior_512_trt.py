"""Export the fixed 352x512 MA-depthmap prior to ONNX -> TensorRT.

This is a 512-preview-specific companion to tools/export_prior_trt.py. It
captures the whole MA prior at the padded shape produced by BENCH_MAX_SIZE=512
for the bicycle frame: RGB (1,3,352,512) -> metric depth (1,352,512).

The general TRT path remains the dynamic DINOv3 patch-encoder engine. This
fixed engine is only used by MADepthMapPrior when the input is exactly 352x512.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
for sub in ("src", "src/model", "src/model/deformconv"):
    sys.path.insert(0, str(REPO / sub))

OUT = REPO / "checkpoints" / "trt"
OUT.mkdir(parents=True, exist_ok=True)
ONNX = OUT / "prior_full_352x512.onnx"
ENGINE = OUT / "prior_full_352x512_fp16.engine"


class FullPrior512ExportWrapper(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # Fixed 352x512 input exported at the legacy f_px=0.6*W. Runtime
        # rescales the metric output when COLMAP intrinsics provide a
        # different focal length.
        out = self.net.infer(rgb, f_px=307.2)["depth"]
        if out.dim() == 2:
            out = out.unsqueeze(0)
        elif out.dim() == 4 and out.shape[1] == 1:
            out = out.squeeze(1)
        return out


def export_onnx(path: Path) -> None:
    from ma_depthmap import MADepthMapPrior

    prior = MADepthMapPrior(fp16=True, trt=False).eval().cuda()
    wrapper = FullPrior512ExportWrapper(prior.net).eval().cuda()
    dummy = torch.randn(1, 3, 352, 512, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"[full-prior-onnx] output shape={tuple(out.shape)} dtype={out.dtype}")

    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(path),
            input_names=["rgb"],
            output_names=["metric_depth"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"[full-prior-onnx] exported {path.name} in {time.perf_counter() - t0:.1f}s "
          f"({path.stat().st_size / 1e6:.0f} MB)")


def _tactic_source_mask(trt, spec: str) -> int:
    aliases = {
        "cublas": trt.TacticSource.CUBLAS,
        "cublas_lt": trt.TacticSource.CUBLAS_LT,
        "cublaslt": trt.TacticSource.CUBLAS_LT,
        "cudnn": trt.TacticSource.CUDNN,
        "edge_mask": trt.TacticSource.EDGE_MASK_CONVOLUTIONS,
        "edge": trt.TacticSource.EDGE_MASK_CONVOLUTIONS,
        "jit": trt.TacticSource.JIT_CONVOLUTIONS,
    }
    mask = 0
    for item in spec.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key == "all":
            for source in aliases.values():
                mask |= 1 << int(source)
            continue
        if key not in aliases:
            raise ValueError(f"unknown tactic source {item!r}; choices: {', '.join(sorted(aliases))}, all")
        mask |= 1 << int(aliases[key])
    if not mask:
        raise ValueError("at least one tactic source is required")
    return mask


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_gb: int,
    builder_opt_level: int,
    avg_timing_iterations: int,
    tactic_sources: str,
    precision_constraints: str,
    allow_tf32: bool,
    strict_nans: bool,
) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print("  parse err:", parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

    softmax_fp32 = 0
    patch_embed_fp32 = 0
    patch_embed_re = re.compile(r"/encoder/patch_encoder/patch_embed/proj/Conv$")
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        force_fp32 = False
        if layer.type == trt.LayerType.SOFTMAX:
            softmax_fp32 += 1
            force_fp32 = True
        elif patch_embed_re.search(layer.name):
            patch_embed_fp32 += 1
            force_fp32 = True
        if force_fp32:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)

    for i in range(network.num_outputs):
        network.get_output(i).dtype = trt.float32

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.builder_optimization_level = builder_opt_level
    config.avg_timing_iterations = avg_timing_iterations
    config.set_flag(trt.BuilderFlag.FP16)
    if allow_tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)
    if precision_constraints == "obey":
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    elif precision_constraints == "prefer":
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    if strict_nans:
        config.set_flag(trt.BuilderFlag.STRICT_NANS)
    config.set_tactic_sources(_tactic_source_mask(trt, tactic_sources))

    profile = builder.create_optimization_profile()
    shape = (1, 3, 352, 512)
    profile.set_shape("rgb", min=shape, opt=shape, max=shape)
    config.add_optimization_profile(profile)

    print(f"[full-prior-engine] building FP16 fixed 352x512 ({network.num_layers} layers, "
          f"{softmax_fp32} softmax FP32, {patch_embed_fp32} patch-embed FP32, "
          f"opt={builder_opt_level}, "
          f"avg_timing={avg_timing_iterations}, tactics={tactic_sources}, "
          f"constraints={precision_constraints}, tf32={allow_tf32}, "
          f"strict_nans={strict_nans}) -> {engine_path.name}...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build returned None")
    engine_path.write_bytes(serialized)
    print(f"[full-prior-engine] built {engine_path.name} in {time.perf_counter() - t0:.1f}s "
          f"({engine_path.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["onnx", "engine", "all"], default="all")
    parser.add_argument("--onnx", type=Path, default=ONNX)
    parser.add_argument("--engine", type=Path, default=ENGINE)
    parser.add_argument("--workspace-gb", type=int, default=24)
    parser.add_argument("--builder-opt-level", type=int, default=5)
    parser.add_argument("--avg-timing-iterations", type=int, default=1)
    parser.add_argument("--tactic-sources", default="cublas_lt",
                        help="Comma-separated TensorRT tactic sources. Default is the retained cuBLASLt-only path.")
    parser.add_argument("--precision-constraints", choices=["none", "prefer", "obey"], default="obey")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--strict-nans", action="store_true", default=True,
                        help="Keep TensorRT NaN semantics strict. Enabled by default for the retained engine.")
    args = parser.parse_args()

    if args.stage in ("onnx", "all"):
        export_onnx(args.onnx)
    if args.stage in ("engine", "all"):
        build_engine(
            args.onnx,
            args.engine,
            args.workspace_gb,
            args.builder_opt_level,
            args.avg_timing_iterations,
            args.tactic_sources,
            args.precision_constraints,
            args.allow_tf32,
            args.strict_nans,
        )


if __name__ == "__main__":
    main()
