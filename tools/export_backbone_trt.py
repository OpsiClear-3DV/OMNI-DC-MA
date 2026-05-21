"""Export OMNI-DC backbone subgraphs to ONNX -> TensorRT.

This is the explicit-engine path for the PVT/CBAM backbone, matching the
repo's TRT direction: no Torch-TensorRT / torch.compile. Exports are
fixed-shape because PVT positional embedding shape logic is traced as a
constant by the legacy ONNX exporter and the currently wired runtime engine is
shape-specific.

Components:
  - dec2: recommended production candidate. For the padded bicycle frame,
    build with --component dec2 --height 824 --width 1240.
  - dec3: fixed-shape decoder block feeding dec2. For the padded bicycle
    frame, build with --component dec3 --height 412 --width 620.
  - dec4/dec5/dec6: smaller fixed-shape decoder blocks in the same path.
  - backbone: experimental full backbone engine.

Important full-backbone limitation:
  - FP16/autocast backbone execution produced NaNs on real inference inputs.
  - The full 1648x2480 FP32 engine builds, but a direct execution test consumed
    essentially all 32 GB GPU memory and did not finish within 20 minutes.

So the full backbone path is useful for experiments and smaller fixed shapes.
Do not wire the full-resolution backbone engine into production inference until
engine execution is separately proven on the target GPU.
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
for sub in ("src", "src/model", "src/model/deformconv"):
    sys.path.insert(0, str(REPO / sub))

OUT = REPO / "checkpoints" / "trt"
OUT.mkdir(parents=True, exist_ok=True)


class BackboneExportWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        _, guide, spn_confidence, context, confidence_input, _ = self.backbone(rgb, depth, 0)
        return guide, spn_confidence, context, confidence_input


class DecoderBlockExportWrapper(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x)


DECODER_INPUT_CHANNELS = {
    "dec6": 512,
    "dec5": 576,
    "dec4": 256,
    "dec2": 192,
    "dec3": 128,
}


def _args_for_model() -> None:
    sys.argv = [
        "export_backbone_trt",
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


def patch_cbam_reductions(module: nn.Module) -> int:
    """Make CBAM export TensorRT-friendly without changing math.

    AdaptiveAvgPool2d(1) and AdaptiveMaxPool2d(1) are exactly mean/amax over
    H,W. The default ONNX export lowers AdaptiveMaxPool2d(1) to a giant
    full-image MaxPool kernel, which TensorRT rejects at full image sizes.
    """
    from resnet_cbam import ChannelAttention

    count = 0
    for child in module.modules():
        if isinstance(child, ChannelAttention):

            def forward(self, x):
                avg_out = self.fc(torch.mean(x, dim=(2, 3), keepdim=True))
                max_out = self.fc(torch.amax(x, dim=(2, 3), keepdim=True))
                return self.sigmoid(avg_out + max_out)

            child.forward = types.MethodType(forward, child)
            count += 1
    return count


def load_export_wrapper(component: str) -> nn.Module:
    _args_for_model()
    from config import args
    from model.infer import load_model

    net = load_model(args)
    if component == "backbone":
        wrapper = BackboneExportWrapper(net.backbone).eval().cuda()
    elif component in DECODER_INPUT_CHANNELS:
        wrapper = DecoderBlockExportWrapper(getattr(net.backbone, component)).eval().cuda()
    else:
        raise ValueError(component)
    patched = patch_cbam_reductions(wrapper)
    print(f"[backbone-trt] patched {patched} CBAM attention blocks for export.")
    return wrapper


def export_onnx(component: str, height: int, width: int, batch: int, onnx_path: Path) -> None:
    wrapper = load_export_wrapper(component)
    if component == "backbone":
        inputs = (
            torch.randn(batch, 3, height, width, device="cuda"),
            torch.randn(batch, 2, height, width, device="cuda"),
        )
        input_names = ["rgb", "depth"]
        output_names = ["guide", "spn_confidence", "context", "confidence_input"]
    elif component in DECODER_INPUT_CHANNELS:
        inputs = (torch.randn(batch, DECODER_INPUT_CHANNELS[component], height, width, device="cuda"),)
        input_names = ["x"]
        output_names = ["y"]
    else:
        raise ValueError(component)

    with torch.no_grad():
        outs = wrapper(*inputs)
    torch.cuda.synchronize()
    outs = outs if isinstance(outs, (list, tuple)) else (outs,)
    print("[backbone-trt] output shapes:", [tuple(t.shape) for t in outs])

    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"[backbone-trt] exported {onnx_path.name} in {time.perf_counter() - t0:.1f}s "
          f"({onnx_path.stat().st_size / 1e6:.0f} MB)")


def build_engine(
    component: str,
    height: int,
    width: int,
    batch: int,
    precision: str,
    onnx_path: Path,
    engine_path: Path,
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

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 24 << 30)
    config.builder_optimization_level = 5
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    if component == "backbone":
        input_shapes = [
            ("rgb", (batch, 3, height, width)),
            ("depth", (batch, 2, height, width)),
        ]
    elif component in DECODER_INPUT_CHANNELS:
        input_shapes = [("x", (batch, DECODER_INPUT_CHANNELS[component], height, width))]
    else:
        raise ValueError(component)
    for name, shape in input_shapes:
        profile.set_shape(name, min=shape, opt=shape, max=shape)
    config.add_optimization_profile(profile)

    print(f"[backbone-trt] building {component} {precision.upper()} fixed-shape engine "
          f"(batch {batch}, {height}x{width}, {network.num_layers} layers)...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build returned None")
    engine_path.write_bytes(serialized)
    print(f"[backbone-trt] built {engine_path.name} in {time.perf_counter() - t0:.1f}s "
          f"({engine_path.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["backbone", "dec2", "dec3", "dec4", "dec5", "dec6"],
                        default="backbone",
                        help="Subgraph to export. For decoder blocks, height/width are block input dimensions.")
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--stage", choices=["onnx", "engine", "all"], default="all")
    args = parser.parse_args()

    batch_tag = "" if args.batch == 1 else f"_b{args.batch}"
    stem = f"{args.component}{batch_tag}_{args.height}x{args.width}_reduce_{args.precision}"
    onnx_path = OUT / f"{stem}.onnx"
    engine_path = OUT / f"{stem}.engine"
    if args.stage in ("onnx", "all"):
        export_onnx(args.component, args.height, args.width, args.batch, onnx_path)
    if args.stage in ("engine", "all"):
        build_engine(args.component, args.height, args.width, args.batch, args.precision,
                     onnx_path, engine_path)


if __name__ == "__main__":
    main()
