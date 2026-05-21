"""Export the MA-depthmap DINOv3-H patch_encoder to ONNX -> TensorRT.

Recipe adapted from OpsiClear-3DV/romaxx engine_builder (DINOv3 path):
  1. fp16 torch.onnx.export, opset 17, dynamo=False, do_constant_folding=True,
     dynamic batch axis.
  2. onnxsim.simplify(skip_shape_inference=True)  (DINOv3 RoPE has a dynamic Sub)
  3. TRT EXPLICIT_BATCH FP16 engine, opt level 5, dynamic-batch profile. This
     is the fastest validated prior path on this setup.

The decoder needs 5 ViT tensors (final + the 4 hooked intermediate blocks
[7,13,19,25] -> backbone_highres_hook0..3). Forward hooks don't export to
ONNX, so the wrapper returns them as explicit outputs.

Stages are gated: ONNX export is the do-or-die step (RoPE ViT-H, 5 outputs,
>2 GB external data). Run with --stage onnx|sim|engine|all.
"""

from __future__ import annotations

import argparse
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
ONNX = OUT / "prior_dinov3h.onnx"
ONNX_SIM = OUT / "prior_dinov3h_simplified.onnx"
ENGINE = OUT / "prior_dinov3h_fp16.engine"

HOOK_BLOCKS = [7, 13, 19, 25]  # VIT_CONFIG dinov3_vith16plus encoder_feature_layer_ids


class DinoExportWrapper(nn.Module):
    """patch_encoder -> (final_encoding, blk7, blk13, blk19, blk25).

    Mirrors exactly what MetricAnythingEncoder's forward hooks capture: the
    output tensor of each hooked transformer block, plus the encoder's normal
    forward output. Hooks (side effects) are replaced by explicit returns so
    the graph is ONNX-exportable.
    """

    def __init__(self, patch_encoder: nn.Module):
        super().__init__()
        self.pe = patch_encoder
        self._taps: dict[int, torch.Tensor] = {}
        for bid in HOOK_BLOCKS:
            self.pe.blocks[bid].register_forward_hook(self._mk(bid))

    def _mk(self, bid):
        def hook(_m, _i, out):
            self._taps[bid] = out[0] if isinstance(out, (list, tuple)) else out
        return hook

    def forward(self, x):
        self._taps.clear()
        final = self.pe(x)
        if isinstance(final, (list, tuple)):
            final = final[0]
        return (final, *[self._taps[b] for b in HOOK_BLOCKS])


def _load_pe():
    from ma_depthmap import MADepthMapPrior
    p = MADepthMapPrior(fp16=True).eval().cuda()
    return p.net.encoder.patch_encoder


def export_onnx():
    pe = _load_pe()
    w = DinoExportWrapper(pe).eval().cuda()
    dummy = torch.randn(1, 3, 384, 384, device="cuda", dtype=torch.float16)
    names_out = ["final", "blk7", "blk13", "blk19", "blk25"]
    dyn = {"img": {0: "b"}, **{n: {0: "b"} for n in names_out}}
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            w, dummy, str(ONNX),
            input_names=["img"], output_names=names_out,
            dynamic_axes=dyn, opset_version=17,
            do_constant_folding=True, dynamo=False,
        )
    print(f"[onnx] exported {ONNX.name} in {time.perf_counter()-t0:.1f}s")
    import onnx
    m = onnx.load(str(ONNX))
    print(f"[onnx] {len(m.graph.node)} nodes, {len(m.graph.initializer)} initializers")


def simplify_onnx():
    import onnx
    from onnxsim import simplify
    m = onnx.load(str(ONNX))
    nb = len(m.graph.node)
    t0 = time.perf_counter()
    ms, ok = simplify(m, check_n=0, perform_optimization=True, skip_shape_inference=True)
    assert ok, "onnxsim ok=False"
    data_path = ONNX_SIM.with_name(ONNX_SIM.name + ".data")
    ONNX_SIM.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)
    onnx.save(ms, str(ONNX_SIM), save_as_external_data=True,
              all_tensors_to_one_file=True, location=ONNX_SIM.name + ".data",
              size_threshold=1024)
    print(f"[sim] {nb} -> {len(ms.graph.node)} nodes in {time.perf_counter()-t0:.1f}s")


def build_engine(
    min_batch: int = 1,
    opt_batch: int = 35,
    max_batch: int = 64,
    workspace_gb: int = 16,
    engine_path: Path = ENGINE,
):
    import tensorrt as trt
    if min_batch < 1 or opt_batch < min_batch or max_batch < opt_batch:
        raise ValueError(f"invalid batch profile: min={min_batch}, opt={opt_batch}, max={max_batch}")
    lg = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(lg)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    p = trt.OnnxParser(net, lg)
    if not p.parse_from_file(str(ONNX_SIM)):
        for i in range(p.num_errors):
            print("  parse err:", p.get_error(i))
        raise RuntimeError("ONNX parse failed")
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    cfg.builder_optimization_level = 5
    cfg.set_flag(trt.BuilderFlag.FP16)
    prof = b.create_optimization_profile()
    prof.set_shape("img", min=(min_batch, 3, 384, 384), opt=(opt_batch, 3, 384, 384),
                   max=(max_batch, 3, 384, 384))
    cfg.add_optimization_profile(prof)
    print(f"[engine] building fp16 ({net.num_layers} layers, "
          f"profile min={min_batch}, opt={opt_batch}, max={max_batch}) -> {engine_path.name}...")
    t0 = time.perf_counter()
    eng = b.build_serialized_network(net, cfg)
    if eng is None:
        raise RuntimeError("engine build returned None")
    engine_path.write_bytes(eng)
    print(f"[engine] {engine_path.name} built in {time.perf_counter()-t0:.1f}s "
          f"({engine_path.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["onnx", "sim", "engine", "all"], default="onnx")
    ap.add_argument("--opt-batch", type=int, default=35,
                    help="TensorRT profile opt batch in ViT patch images. One MA image uses 35 patches.")
    ap.add_argument("--min-batch", type=int, default=1,
                    help="TensorRT profile min batch in ViT patch images. Use 35 for the fixed default path.")
    ap.add_argument("--max-batch", type=int, default=64,
                    help="TensorRT profile max batch in ViT patch images. Default supports one MA image (35 patches).")
    ap.add_argument("--workspace-gb", type=int, default=16)
    ap.add_argument("--engine", type=Path, default=None,
                    help="Output engine path. Defaults to prior_dinov3h_fp16.engine.")
    a = ap.parse_args()
    engine_path = a.engine or ENGINE
    if a.stage in ("onnx", "all"):
        export_onnx()
    if a.stage in ("sim", "all"):
        simplify_onnx()
    if a.stage in ("engine", "all"):
        build_engine(a.min_batch, a.opt_batch, a.max_batch, a.workspace_gb, engine_path)
