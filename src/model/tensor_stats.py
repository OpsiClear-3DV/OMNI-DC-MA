from __future__ import annotations

import torch


def quantile_02_98_flat(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact linear 2% and 98% quantiles for a BxN tensor."""
    sorted_flat = flat.sort(dim=1).values
    n = flat.shape[1]
    if n < 1:
        raise ValueError("flat must have at least one value per row")

    def gather(q: float) -> torch.Tensor:
        pos = (n - 1) * q
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return sorted_flat[:, lo] * (1.0 - frac) + sorted_flat[:, hi] * frac

    return gather(0.02), gather(0.98)
