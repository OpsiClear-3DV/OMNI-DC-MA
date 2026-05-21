"""Metric-Anything Prompt-Free Depth Map student (vendored).

This package is a verbatim copy of the `models/student_depthmap/` subtree from
https://github.com/metric-anything/metric-anything (Apache-2.0; see LICENSE in
this directory). It's been vendored in-tree so OMNI-DC has zero runtime fetch
dependencies and no CWD-sensitive `torch.hub.load("network", ...)` call.

Public entry point: `MADepthMapPrior` is a thin nn.Module wrapper that exposes
the contract OMNI-DC's forward pass expects from a frozen monocular-depth prior
(see `ognidc.py`):

    rgb: (B, 3, H, W)  ImageNet-normalized RGB
    out: (B, H, W)     disparity-polarity dense prediction (high = close)

Internally we run the model in metric-depth mode then return `1 / depth` so the
prior's polarity matches DAv2's training distribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The vendored `depth_model.py` does `from network.{decoder,encoder,vit_factory}
# import ...`. Make `network` resolvable when our package is imported from
# anywhere — we don't want to rely on sys.path mutation from the call site.
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.append(str(_PKG_DIR))

from .prior import MADepthMapPrior  # noqa: E402

__all__ = ["MADepthMapPrior"]
