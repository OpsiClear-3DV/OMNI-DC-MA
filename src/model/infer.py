"""Single inference entry point for OMNI-DC-MA.

The OGNIDC orchestration — load, ImageNet-normalize, pad to the resolution
multiple, forward, crop back — was duplicated across demo.py,
tools/bench_inference.py, and tests/. It now lives here, once. Keeping one
copy means the verified bicycle baseline (anchor MAE ~0.0002 m, byte-stable)
can't silently drift between call sites.

The op sequence (pad order, crop, squeeze) mirrors the original demo.py. RGB
is decoded on the GPU via torchvision/nvJPEG when possible; that decoder is
not bit-identical to PIL/libjpeg, so outputs are *numerically equivalent* but
not byte-identical to the pre-GPU-decode baseline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.io import ImageReadMode, decode_jpeg, read_file

from model.ognidc import OGNIDC

# Module-level so every call site shares one definition (was copy-pasted).
_RGB_TF = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
_DEP_TF = T.ToTensor()

# ImageNet normalization on-GPU (for the nvJPEG decode path).
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Dummy K — OGNIDC ignores it; build once instead of per predict() call (#4).
_DUMMY_K = torch.eye(3).reshape(1, 3, 3)
_DUMMY_K_BY_DEVICE: dict[torch.device, torch.Tensor] = {}


def _load_rgb_gpu(rgb_path) -> torch.Tensor | None:
    """Decode an RGB image straight to a normalized (1,3,H,W) CUDA tensor via
    torchvision's nvJPEG path. Returns None (caller falls back to PIL) for
    non-JPEG inputs or any decode failure — robustness at the input boundary.

    nvJPEG != libjpeg bit-for-bit, so this changes pixel values slightly vs
    the old PIL path (and thus the prediction); that's inherent to GPU decode.
    """
    try:
        data = read_file(str(rgb_path))                       # raw bytes, CPU uint8
        img = decode_jpeg(data, mode=ImageReadMode.RGB, device="cuda")  # (3,H,W) uint8 cuda
        img = img.float().div_(255.0)
        return ((img - _MEAN.to(img.device)) / _STD.to(img.device)).unsqueeze(0)
    except Exception:
        return None  # PNG / unsupported / nvJPEG unavailable -> PIL fallback


def _dummy_k(device: torch.device) -> torch.Tensor:
    k = _DUMMY_K_BY_DEVICE.get(device)
    if k is None:
        k = _DUMMY_K.to(device=device)
        _DUMMY_K_BY_DEVICE[device] = k
    return k


def load_model(args, hf_repo: str = "zuoym15/OMNI-DC") -> nn.Module:
    """Load OGNIDC from HuggingFace and wrap for inference.

    Single-GPU bare module on CUDA (no DataParallel — #C). With
    ``args.trt`` set, inference uses explicit TensorRT engines where they are
    proven: MA-depthmap's DINOv3-H patch encoder, plus fixed-shape backbone
    decoder subgraphs when their engines are present. Missing/invalid engines
    stay on eager PyTorch.
    """
    if getattr(args, "model", "OGNIDC") != "OGNIDC":
        raise TypeError(args.model, ["OGNIDC"])
    net = OGNIDC.from_pretrained(hf_repo, args=args)
    net.eval()
    net = net.cuda()
    if getattr(args, "trt", False) and not getattr(args, "skip_backbone_trt", False):
        from model.backbone_trt import install_backbone_trt

        install_backbone_trt(net.backbone)
    return net


def load_pair(rgb_path, depth_path, return_sparse: bool = False):
    """Load an RGB image + sparse-depth .npy as the (1,3,H,W)/(1,1,H,W) CUDA
    tensors the model expects.

    RGB is GPU-decoded (nvJPEG) when possible, else PIL. The sparse .npy is
    read once; pass `return_sparse=True` to also get that ndarray back so
    callers don't re-`np.load` it for the anchor cap (#2). Default 2-tuple
    keeps existing callers/tests unchanged.
    """
    rgb = _load_rgb_gpu(rgb_path)
    if rgb is None:
        rgb = _RGB_TF(Image.open(rgb_path).convert("RGB"))[None].cuda()
    sparse = np.load(depth_path).astype(np.float32)
    dep = _DEP_TF(sparse)[None].cuda()
    if return_sparse:
        return rgb, dep, sparse
    return rgb, dep


def apply_anchor_cap(
    depth: np.ndarray, sparse: np.ndarray, factor: float = 2.0
) -> tuple[np.ndarray, float, int]:
    """Zero out predictions farther than ``factor`` x the deepest SfM anchor.

    OMNI-DC's metric scale is pinned by the sparse anchors; anything well
    beyond the farthest anchor is unconstrained extrapolation (open sky /
    far-field) — see the README's prior-firewall discussion. This caps that
    region using the same ``0 == invalid`` sentinel the sparse input uses, so
    downstream consumers (3DGS reg, point clouds) that already skip zeros need
    no changes.

    Pure / non-mutating. ``factor <= 0`` disables (returns a copy unchanged).
    Returns ``(masked_depth, threshold, n_capped)``; threshold is ``inf`` and
    n_capped 0 when disabled or when there are no valid anchors. n_capped is
    computed here from the same mask used to zero, so callers don't re-scan
    the array (#D).
    """
    out = depth.copy()
    if factor <= 0:
        return out, float("inf"), 0
    valid = sparse > 0
    if not valid.any():
        return out, float("inf"), 0
    threshold = factor * float(sparse[valid].max())
    over = out > threshold
    n_capped = int(over.sum())
    out[over] = 0.0
    return out, threshold, n_capped


def apply_sky_mask(depth: np.ndarray, sky: np.ndarray | None) -> tuple[np.ndarray, int]:
    """Zero output depth wherever the prior marked sky/far-field.

    ``sky`` may be bool or a soft/interpolated mask. Values > 0.5 are treated
    as invalid far-field. Returns a copy plus the number of nonzero depth
    pixels newly zeroed by the mask.
    """
    out = depth.copy()
    if sky is None:
        return out, 0
    sky_mask = np.asarray(sky) > 0.5
    if sky_mask.shape != out.shape:
        raise ValueError(f"sky mask shape {sky_mask.shape} does not match depth shape {out.shape}")
    newly_zeroed = int(np.count_nonzero(sky_mask & (out != 0.0)))
    out[sky_mask] = 0.0
    return out, newly_zeroed


@torch.inference_mode()
def predict_tensor(
    net: nn.Module,
    rgb: torch.Tensor,
    dep: torch.Tensor,
    num_resolution: int,
    return_sky_mask: bool = False,
    mono_dep: torch.Tensor | None = None,
):
    """Run OGNIDC and keep the cropped output on the input device.

    Returns B x 1 x H x W tensors. Use predict() for the legacy NumPy API.

    Args:
        net: result of `load_model`.
        rgb: (1, 3, H, W) ImageNet-normalized, on CUDA.
        dep: (1, 1, H, W) sparse depth in metres, on CUDA.
        num_resolution: args.num_resolution (controls the pad multiple).
        return_sky_mask: if True, also return the prior-flagged sky/far-field
            mask (bool (H, W), True = the prior capped this pixel as
            unconstrained far-field). Opt-in so existing callers are unchanged.
        mono_dep: optional precomputed mono-prior conditioning channel,
            already normalized the same way OGNIDC does internally. When
            provided, the expensive MA-depthmap prior is skipped.

    Returns:
        (H, W) float32 dense depth — or, if return_sky_mask, the tuple
        (depth, sky_mask) where sky_mask is bool (H, W) (all-False if no cap
        was applied / scene fully anchored).
    """
    _, _, H, W = rgb.shape
    diviser = int(4 * 2 ** (num_resolution - 1))
    H_pad = (-H) % diviser
    W_pad = (-W) % diviser
    if H_pad or W_pad:
        rgb = torch.nn.functional.pad(rgb, (0, W_pad, 0, H_pad))
        dep = torch.nn.functional.pad(dep, (0, W_pad, 0, H_pad))
        if mono_dep is not None:
            mono_dep = torch.nn.functional.pad(mono_dep, (0, W_pad, 0, H_pad))

    sample = {
        "rgb": rgb,
        "dep": dep,
        "K": _dummy_k(rgb.device),                  # dummy; unused by OGNIDC (#4)
        "pattern": 0,                               # dummy; unused by OGNIDC
    }
    if mono_dep is not None:
        sample["mono_dep"] = mono_dep
    output = net(sample)
    depth = output["pred"][..., :H, :W]
    if not return_sky_mask:
        return depth

    sm = output.get("sky_mask")
    if sm is None:
        sky = torch.zeros_like(depth, dtype=torch.bool)
    else:
        sky = sm[..., :H, :W] > 0.5
    return depth, sky


@torch.inference_mode()
def predict(
    net: nn.Module,
    rgb: torch.Tensor,
    dep: torch.Tensor,
    num_resolution: int,
    return_sky_mask: bool = False,
):
    """Run OGNIDC on one RGB + sparse-depth pair and return NumPy arrays."""
    if return_sky_mask:
        depth_t, sky_t = predict_tensor(net, rgb, dep, num_resolution, return_sky_mask=True)
        return depth_t.squeeze().cpu().numpy(), sky_t.squeeze().cpu().numpy()
    depth_t = predict_tensor(net, rgb, dep, num_resolution)
    return depth_t.squeeze().cpu().numpy()
