"""TensorRT runtime helpers for narrow OMNI-DC backbone subgraphs.

The full fixed-shape backbone engine is not wired into inference: it builds,
but full-resolution execution was memory-hostile on the 32 GB target GPU.
The practical romaxx-style path is to keep explicit ONNX -> TRT engines and
install only proven subgraphs. Today that is the fixed-shape decoder chain
(``backbone.dec6`` through ``backbone.dec2``) for the padded full-resolution
bicycle frame plus the validated 512-preview representative batch shapes:

  dec6 input  x: (1, 512, 51, 77) FP32
  dec6 output y: (1, 256, 102, 154) FP32
  dec5 input  x: (1, 576, 103, 155) FP32
  dec5 output y: (1, 128, 206, 310) FP32
  dec4 input  x: (1, 256, 206, 310) FP32
  dec4 output y: (1, 64, 412, 620) FP32
  dec3 input  x: (1, 128, 412, 620) FP32
  dec3 output y: (1, 64, 824, 1240) FP32
  dec2 input  x: (1, 192, 824, 1240) FP32
  dec2 output y: (1, 64, 1648, 2480) FP32

The 512-preview path additionally ships B5 engines for final-output
representative replay and B16 engines for full-batch preview graphs. The
final dec2 block stays eager on the 512-preview path: with the relaxed
<1s/batch target it gives lower eager-reference error than the B5/B16 dec2 TRT
engines while staying inside the latency budget. The full-resolution dec2 TRT
engine remains available for the original large-frame path.

Shape mismatch, deserialization failure, runtime failure, or self-check
failure permanently falls back to eager PyTorch for that block.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn

_TRT_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "trt"
BACKBONE_TRT_ENGINES = {
    "dec6": [
        _TRT_DIR / "dec6_51x77_reduce_fp32.engine",
        _TRT_DIR / "dec6_b5_11x16_reduce_fp32.engine",
        _TRT_DIR / "dec6_b16_11x16_reduce_fp32.engine",
    ],
    "dec5": [
        _TRT_DIR / "dec5_103x155_reduce_fp32.engine",
        _TRT_DIR / "dec5_b5_22x32_reduce_fp32.engine",
        _TRT_DIR / "dec5_b16_22x32_reduce_fp32.engine",
    ],
    "dec4": [
        _TRT_DIR / "dec4_206x310_reduce_fp32.engine",
        _TRT_DIR / "dec4_b5_44x64_reduce_fp32.engine",
        _TRT_DIR / "dec4_b16_44x64_reduce_fp32.engine",
    ],
    "dec3": [
        _TRT_DIR / "dec3_412x620_reduce_fp32.engine",
        _TRT_DIR / "dec3_b5_88x128_reduce_fp32.engine",
        _TRT_DIR / "dec3_b16_88x128_reduce_fp32.engine",
    ],
    "dec2": [
        _TRT_DIR / "dec2_824x1240_reduce_fp32.engine",
    ],
}


def _selfcheck_cache_enabled() -> bool:
    return os.environ.get("OMNIDC_TRT_SELF_CHECK_CACHE", "1") not in {
        "0", "false", "FALSE", "no", "NO",
    }


def _force_selfcheck() -> bool:
    return os.environ.get("OMNIDC_TRT_FORCE_SELF_CHECK", "0") in {
        "1", "true", "TRUE", "yes", "YES",
    }


def _engine_sig(path: Path) -> dict[str, int | str]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _selfcheck_cache_path(path: Path) -> Path:
    return Path(str(path) + ".selfcheck.json")


def _selfcheck_cache_hit(path: Path, tag: str) -> bool:
    if not _selfcheck_cache_enabled() or _force_selfcheck():
        return False
    try:
        data = json.loads(_selfcheck_cache_path(path).read_text())
        return data.get("tag") == tag and data.get("engine") == _engine_sig(path)
    except Exception:
        return False


def _mark_selfcheck_ok(path: Path, tag: str, rel_err: float) -> None:
    if not _selfcheck_cache_enabled():
        return
    data = {
        "tag": tag,
        "engine": _engine_sig(path),
        "rel_err": float(rel_err),
    }
    try:
        _selfcheck_cache_path(path).write_text(json.dumps(data, indent=2, sort_keys=True))
    except OSError:
        pass


def _slot_tag(name: str, input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> str:
    return f"backbone:{name}:{input_shape}->{output_shape}"


class TrtDecoderBlock(nn.Module):
    """Drop-in wrapper for a fixed-shape decoder block with eager fallback."""

    def __init__(self, name: str, eager_block: nn.Module, engine_paths: list[Path]):
        super().__init__()
        self._name = name
        self._eager = eager_block
        self._in = "x"
        self._out = "y"
        self._dtype_map = {}
        self._slots = []
        self.force_eager = False

        try:
            import tensorrt as trt

            self._dtype_map = {
                trt.DataType.BF16: torch.bfloat16,
                trt.DataType.HALF: torch.float16,
                trt.DataType.FLOAT: torch.float32,
            }
            rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            for engine_path in engine_paths:
                engine = rt.deserialize_cuda_engine(engine_path.read_bytes())
                input_shape = tuple(engine.get_tensor_shape(self._in))
                output_shape = tuple(engine.get_tensor_shape(self._out))
                tag = _slot_tag(name, input_shape, output_shape)
                checked = _selfcheck_cache_hit(engine_path, tag)
                self._slots.append({
                    "path": engine_path,
                    "tag": tag,
                    "engine": engine,
                    "ctx": engine.create_execution_context(),
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                    "checked": checked,
                    "failed": False,
                })
                if checked:
                    print(f"[trt] backbone {name} self-check cache hit ({engine_path.name}).")
                print(f"[trt] backbone {name} engine loaded ({engine_path.name}).")
        except Exception as e:
            print(f"[trt] backbone {name} engine unavailable ({type(e).__name__}: {str(e)[:100]}); "
                  f"eager {name}.")

    @property
    def available(self) -> bool:
        return any(not slot["failed"] for slot in self._slots)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.force_eager:
            return self._eager(x)

        slot = self._slot_for(x)
        if slot is None:
            return self._eager(x)

        try:
            out = self._run(slot, x)
        except Exception as e:
            print(f"[trt] backbone {self._name} engine run failed ({type(e).__name__}: {str(e)[:100]}); "
                  "reverting that shape to eager.")
            slot["failed"] = True
            return self._eager(x)

        if not slot["checked"]:
            slot["checked"] = True
            ref = self._eager(x)
            denom = ref.float().abs().mean().clamp(min=1e-3)
            rel = float((ref.float() - out.float()).abs().mean() / denom)
            if not math.isfinite(rel) or rel > 0.02:
                print(f"[trt] backbone {self._name} FAILED correctness self-check (rel err {rel:.3f} > 0.02). "
                      "Permanent revert to eager for that shape.")
                slot["failed"] = True
                return ref
            _mark_selfcheck_ok(slot["path"], slot["tag"], rel)
            print(f"[trt] backbone {self._name} self-check OK (rel err {rel:.4f}).")

        return out

    def _slot_for(self, x: torch.Tensor):
        if not x.is_cuda:
            return None
        shape = tuple(x.shape)
        for slot in self._slots:
            if not slot["failed"] and shape == slot["input_shape"]:
                return slot
        return None

    def _run(self, slot, x: torch.Tensor) -> torch.Tensor:
        ctx, eng = slot["ctx"], slot["engine"]
        input_dtype = self._dtype_map.get(eng.get_tensor_dtype(self._in), torch.float32)
        x_run = x.contiguous().to(input_dtype)

        ctx.set_input_shape(self._in, tuple(x_run.shape))
        ctx.set_tensor_address(self._in, x_run.data_ptr())

        shp = tuple(ctx.get_tensor_shape(self._out))
        if any(dim < 0 for dim in shp):
            raise RuntimeError(f"unresolved TRT output shape for {self._out}: {shp}")
        if shp != slot["output_shape"]:
            raise RuntimeError(f"unexpected TRT output shape for {self._out}: {shp} != {slot['output_shape']}")
        output_dtype = self._dtype_map.get(eng.get_tensor_dtype(self._out), torch.float32)
        out = torch.empty(shp, device=x.device, dtype=output_dtype)
        ctx.set_tensor_address(self._out, out.data_ptr())

        ok = ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")
        return out


def install_backbone_trt(backbone: nn.Module) -> bool:
    """Install proven backbone TRT subgraphs. Returns True when an engine is active."""
    active = False
    for name, engine_paths in BACKBONE_TRT_ENGINES.items():
        existing = [path for path in engine_paths if path.exists()]
        if not existing:
            print(f"[trt] backbone {name} engines missing ({engine_paths[0].parent}); eager {name}.")
            continue
        parent, attr, block = _resolve_child(backbone, name)
        if block is None:
            print(f"[trt] backbone has no {name} module; eager {name}.")
            continue

        if isinstance(block, TrtDecoderBlock):
            active = active or block.available
            continue

        wrapped = TrtDecoderBlock(name, block, existing)
        if wrapped.available:
            setattr(parent, attr, wrapped)
            active = True
    return active


def _resolve_child(root: nn.Module, dotted_name: str) -> tuple[nn.Module | None, str, nn.Module | None]:
    parts = dotted_name.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        child = getattr(parent, part, None)
        if not isinstance(child, nn.Module):
            return None, parts[-1], None
        parent = child
    attr = parts[-1]
    child = getattr(parent, attr, None)
    return parent, attr, child if isinstance(child, nn.Module) else None
