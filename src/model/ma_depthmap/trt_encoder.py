"""TRT-backed drop-in for the MA-depthmap DINOv3-H patch_encoder.

The engine (built by tools/export_prior_trt.py per the romaxx ONNX->TRT
recipe) takes img (B,3,384,384) and returns 5 tensors:
  final, blk7, blk13, blk19, blk25
i.e. the patch_encoder's forward output plus the 4 intermediate-block
tensors the original MetricAnythingEncoder captured via forward hooks.

`TrtPatchEncoder` replaces `encoder.patch_encoder`. On forward it runs the
engine and writes the encoder's `backbone_highres_hook0..3` attributes
(the hooks no longer fire — engine replaces the block forwards), then
returns `final`. Falls back to the original eager patch_encoder on any
TRT failure so the prior never hard-crashes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn

ENGINE_PATH = Path(os.environ.get(
    "OMNIDC_PRIOR_TRT_ENGINE",
    Path(__file__).resolve().parents[3] / "checkpoints" / "trt" / "prior_dinov3h_fp16.engine",
))
FULL_PRIOR_512_PATH = Path(os.environ.get(
    "OMNIDC_FULL_PRIOR_TRT_ENGINE",
    Path(__file__).resolve().parents[3] / "checkpoints" / "trt" / "prior_full_352x512_fp16.engine",
))
_HOOK_ATTRS = ("backbone_highres_hook0", "backbone_highres_hook1",
               "backbone_highres_hook2", "backbone_highres_hook3")


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


class TrtPatchEncoder(nn.Module):
    def __init__(self, eager_pe: nn.Module, encoder: nn.Module,
                 out_dtype: torch.dtype = torch.float32):
        super().__init__()
        self._eager = eager_pe          # fallback (legit submodule)
        # Back-ref to the PARENT encoder. Must bypass nn.Module.__setattr__:
        # registering the parent as a child submodule makes encoder.patch_encoder
        # -> self._encoder -> encoder a cycle that PyTorch's recursive module
        # traversal (.eval/.half/.parameters) blows the stack on.
        object.__setattr__(self, "_encoder", encoder)
        self._out_dtype = out_dtype     # match downstream encoder/decoder dtype
        self.embed_dim = eager_pe.embed_dim
        self.patch_embed = eager_pe.patch_embed
        self._ctx = None
        self._failed = False
        self._checked = False  # one-time engine-vs-eager correctness gate
        self.max_batch: int | None = None
        self._dtype_map = {}
        self._engine_path = ENGINE_PATH
        try:
            import tensorrt as trt
            self._dtype_map = {
                trt.DataType.BF16: torch.bfloat16,
                trt.DataType.HALF: torch.float16,
                trt.DataType.FLOAT: torch.float32,
            }
            logger = trt.Logger(trt.Logger.ERROR)
            rt = trt.Runtime(logger)
            self._engine = rt.deserialize_cuda_engine(ENGINE_PATH.read_bytes())
            self._ctx = self._engine.create_execution_context()
            self._in = "img"
            self._outs = ["final", "blk7", "blk13", "blk19", "blk25"]
            try:
                self.max_batch = int(self._engine.get_tensor_profile_shape(self._in, 0)[2][0])
            except Exception:
                self.max_batch = None
            tag = f"patch_encoder:{self._out_dtype}"
            if _selfcheck_cache_hit(self._engine_path, tag):
                self._checked = True
                print(f"[trt] prior engine self-check cache hit ({ENGINE_PATH.name}).")
            print(f"[trt] prior DINOv3-H engine loaded ({ENGINE_PATH.name}).")
        except Exception as e:
            print(f"[trt] prior engine unavailable ({type(e).__name__}: "
                  f"{str(e)[:100]}); eager patch_encoder.")
            self._failed = True

    @torch.no_grad()
    def _eager_5(self, x):
        """Eager patch_encoder run, returning (final, blk7,13,19,25) the same
        way the engine does — used for the one-time correctness self-check."""
        taps: dict[int, torch.Tensor] = {}
        hs = [self._eager.blocks[b].register_forward_hook(
                  (lambda i: (lambda _m, _i, o: taps.__setitem__(
                      i, o[0] if isinstance(o, (list, tuple)) else o)))(b))
              for b in (7, 13, 19, 25)]
        try:
            f = self._eager(x)
        finally:
            for h in hs:
                h.remove()
        f = f[0] if isinstance(f, (list, tuple)) else f
        return [f, taps[7], taps[13], taps[19], taps[25]]

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if self._failed:
            return self._set_and_return(self._eager_5(x))
        try:
            out = self._run(x)
        except Exception as e:
            print(f"[trt] prior engine run failed ({type(e).__name__}: "
                  f"{str(e)[:100]}); reverting to eager for the rest of the run.")
            self._failed = True
            return self._set_and_return(self._eager_5(x))

        # One-time correctness self-check. The serialized engine dictates its
        # own I/O dtypes, so verify the actual artifact against eager once
        # before trusting it as the conditioning prior.
        if not self._checked:
            self._checked = True
            check_n = min(x.shape[0], 35)
            ref = self._eager_5(x[:check_n])
            out_check = [o[:check_n] for o in out]
            worst = 0.0
            for r, o in zip(ref, out_check, strict=True):
                r = r.float()
                o = o.float()
                denom = r.abs().mean().clamp(min=1e-3)
                worst = max(worst, float((r - o).abs().mean() / denom))
            if worst > 0.05:  # >5% relative — unusable as a conditioning prior
                print(f"[trt] prior engine FAILED correctness self-check "
                      f"(rel err {worst:.2f} > 0.05). Permanent revert to eager.")
                self._failed = True
                return self._set_and_return(self._eager_5(x))
            _mark_selfcheck_ok(self._engine_path, f"patch_encoder:{self._out_dtype}", worst)
            print(f"[trt] prior engine self-check OK (rel err {worst:.3f}).")

        return self._set_and_return(out)

    def _set_and_return(self, five):
        final, h0, h1, h2, h3 = five
        for attr, val in zip(_HOOK_ATTRS, (h0, h1, h2, h3), strict=True):
            setattr(self._encoder, attr, val)
        return final

    def _run(self, x: torch.Tensor):
        ctx, eng = self._ctx, self._engine
        input_dtype = self._dtype_map.get(eng.get_tensor_dtype(self._in), torch.float32)
        x = x.contiguous().to(input_dtype)
        ctx.set_input_shape(self._in, tuple(x.shape))
        ctx.set_tensor_address(self._in, x.data_ptr())
        outs = []
        for name in self._outs:
            shp = tuple(ctx.get_tensor_shape(name))
            if any(dim < 0 for dim in shp):
                raise RuntimeError(f"unresolved TRT output shape for {name}: {shp}")
            dtype = self._dtype_map.get(eng.get_tensor_dtype(name), torch.float32)
            t = torch.empty(shp, device=x.device, dtype=dtype)
            ctx.set_tensor_address(name, t.data_ptr())
            outs.append(t)
        stream = torch.cuda.current_stream().cuda_stream
        ok = ctx.execute_async_v3(stream)
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")
        # Cast to the downstream encoder/decoder dtype (fp16 or fp32).
        return [o.to(self._out_dtype) for o in outs]


class TrtFullPrior512(nn.Module):
    """Fixed-shape full MA prior engine for the 512 preview path.

    The engine takes one padded 352x512 ImageNet-normalized RGB tensor and
    returns metric depth at 352x512. It is intentionally shape-specific: the
    general path still uses the dynamic patch-encoder TRT engine above.
    """

    input_hw = (352, 512)

    def __init__(self, eager_net: nn.Module | None):
        super().__init__()
        # Back-ref only: the parent MADepthMapPrior already owns this module.
        object.__setattr__(self, "_eager", eager_net)
        self._failed = False
        self._checked = False
        self._dtype_map = {}
        self._engine_path = FULL_PRIOR_512_PATH
        try:
            import tensorrt as trt
            self._dtype_map = {
                trt.DataType.BF16: torch.bfloat16,
                trt.DataType.HALF: torch.float16,
                trt.DataType.FLOAT: torch.float32,
            }
            logger = trt.Logger(trt.Logger.ERROR)
            rt = trt.Runtime(logger)
            self._engine = rt.deserialize_cuda_engine(FULL_PRIOR_512_PATH.read_bytes())
            self._ctx = self._engine.create_execution_context()
            self._in = "rgb"
            self._out = "metric_depth"
            if _selfcheck_cache_hit(self._engine_path, "full_prior_352x512"):
                self._checked = True
                print(f"[trt] full 352x512 prior self-check cache hit ({FULL_PRIOR_512_PATH.name}).")
            print(f"[trt] full 352x512 prior engine loaded ({FULL_PRIOR_512_PATH.name}).")
        except Exception as e:
            print(f"[trt] full 352x512 prior engine unavailable ({type(e).__name__}: "
                  f"{str(e)[:100]}); eager prior tail.")
            self._failed = True

    @property
    def available(self) -> bool:
        return not self._failed

    def supports(self, x: torch.Tensor) -> bool:
        return (
            x.is_cuda
            and x.ndim == 4
            and x.shape[0] == 1
            and tuple(x.shape[-2:]) == self.input_hw
        )

    @torch.no_grad()
    def infer_depth(self, x: torch.Tensor, f_px: float | torch.Tensor) -> torch.Tensor:
        if self._failed or not self.supports(x):
            return self._eager_depth(x, f_px)

        try:
            depth = self._run(x).float()
        except Exception as e:
            print(f"[trt] full 352x512 prior engine run failed ({type(e).__name__}: "
                  f"{str(e)[:100]}); reverting to eager full prior.")
            self._failed = True
            return self._eager_depth(x, f_px)

        if not self._checked:
            self._checked = True
            ref = self._eager_depth(x, f_px).float()
            denom = ref.abs().mean().clamp(min=1e-3)
            rel = float((ref - depth).abs().mean() / denom)
            if not torch.isfinite(depth).all() or rel > 0.05:
                print(f"[trt] full 352x512 prior FAILED self-check "
                      f"(rel err {rel:.3f} > 0.05). Permanent revert to eager.")
                self._failed = True
                return ref
            _mark_selfcheck_ok(self._engine_path, "full_prior_352x512", rel)
            print(f"[trt] full 352x512 prior self-check OK (rel err {rel:.4f}).")

        return depth

    def _eager_depth(self, x: torch.Tensor, f_px: float | torch.Tensor) -> torch.Tensor:
        if self._eager is None:
            raise RuntimeError("eager MA prior is not loaded")
        out = self._eager.infer(x, f_px=f_px)["depth"]
        if out.dim() == 2:
            out = out.unsqueeze(0)
        elif out.dim() == 4 and out.shape[1] == 1:
            out = out.squeeze(1)
        return out

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        ctx, eng = self._ctx, self._engine
        input_dtype = self._dtype_map.get(eng.get_tensor_dtype(self._in), torch.float32)
        x = x.contiguous().to(input_dtype)
        ctx.set_input_shape(self._in, tuple(x.shape))
        ctx.set_tensor_address(self._in, x.data_ptr())

        shp = tuple(ctx.get_tensor_shape(self._out))
        if any(dim < 0 for dim in shp):
            raise RuntimeError(f"unresolved TRT output shape for {self._out}: {shp}")
        output_dtype = self._dtype_map.get(eng.get_tensor_dtype(self._out), torch.float32)
        out = torch.empty(shp, device=x.device, dtype=output_dtype)
        ctx.set_tensor_address(self._out, out.data_ptr())

        ok = ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")
        if out.dim() == 2:
            out = out.unsqueeze(0)
        elif out.dim() == 4 and out.shape[1] == 1:
            out = out.squeeze(1)
        return out
