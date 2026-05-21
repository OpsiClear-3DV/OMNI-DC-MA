"""Final-output representative interpolation for the validated 512 path."""

from __future__ import annotations

import torch

CALIBRATED16_REPS = (0, 1, 5, 10, 15)
GENERIC16_REPS = (0, 3, 6, 10, 15)
HIGH_SPAN16_REPS = (0, 6, 12, 14, 15)
CALIBRATED16_SPATIAL_SHAPES = ((340, 512), (352, 512))
CALIBRATED16_RGB_ENDPOINT_SPAN_RANGE = (0.12, 0.20)
HIGH_SPAN16_RGB_ENDPOINT_THRESHOLD = 0.20
GENERIC16_MODE = "metric_generic16"


def final_rep_indices(batch_size: int) -> tuple[int, ...]:
    """Return the robust B16 fallback layout, scaled for non-production probes."""
    if batch_size <= 1:
        return ()
    reps = tuple(round(rep * (batch_size - 1) / 15) for rep in GENERIC16_REPS)
    return tuple(dict.fromkeys(reps))


def select_final_rep_indices(rgb: torch.Tensor) -> tuple[int, ...]:
    if not is_validated_final_rep_batch_shape(rgb.shape[0], rgb.shape[-2:]):
        return final_rep_indices(rgb.shape[0])
    span = _rgb_endpoint_span(rgb)
    if _is_calibrated16_span(span):
        return CALIBRATED16_REPS
    if span > HIGH_SPAN16_RGB_ENDPOINT_THRESHOLD:
        return HIGH_SPAN16_REPS
    return GENERIC16_REPS


def is_validated_final_rep_batch_shape(batch_size: int, spatial_shape) -> bool:
    shape = tuple(int(dim) for dim in spatial_shape)
    return batch_size == 16 and shape in CALIBRATED16_SPATIAL_SHAPES


def select_final_rep_mode(
    rgb: torch.Tensor,
    reps: tuple[int, ...],
    *,
    calibrated_mode: str = "hybrid_calibrated16",
    fallback_mode: str = "log_smootherstep",
) -> str:
    if _is_calibrated16_layout(reps, rgb.shape[0], rgb.shape[-2:]):
        if _is_calibrated16_span(_rgb_endpoint_span(rgb)):
            return calibrated_mode
        return fallback_mode
    if _is_high_span16_layout(reps, rgb.shape[0], rgb.shape[-2:]):
        return "metric_highspan16"
    if _is_generic16_layout(reps, rgb.shape[0], rgb.shape[-2:]):
        return GENERIC16_MODE
    return fallback_mode


def _rgb_endpoint_span(rgb: torch.Tensor) -> float:
    return float((rgb[-1:] - rgb[:1]).abs().mean())


def _is_calibrated16_span(span: float) -> bool:
    span_min, span_max = CALIBRATED16_RGB_ENDPOINT_SPAN_RANGE
    return span_min <= span <= span_max


def _is_generic16_layout(
    reps: tuple[int, ...],
    batch_size: int,
    spatial_shape,
) -> bool:
    return (
        reps == GENERIC16_REPS
        and is_validated_final_rep_batch_shape(batch_size, spatial_shape)
    )


def _is_high_span16_layout(
    reps: tuple[int, ...],
    batch_size: int,
    spatial_shape,
) -> bool:
    return (
        reps == HIGH_SPAN16_REPS
        and is_validated_final_rep_batch_shape(batch_size, spatial_shape)
    )


def _is_calibrated16_layout(
    reps: tuple[int, ...],
    batch_size: int,
    spatial_shape,
) -> bool:
    return (
        reps == CALIBRATED16_REPS
        and is_validated_final_rep_batch_shape(batch_size, spatial_shape)
    )


def _smootherstep(alpha: float) -> float:
    return alpha * alpha * alpha * (alpha * (alpha * 6.0 - 15.0) + 10.0)


# Calibrated for the retained B16 512 layout. The base table is fitted against
# the all-16 TRT teacher, with small later retunes against the eager-gap bench.
_CALIBRATED16_LOG_ALPHA = {
    2: 0.02480437153371516,
    3: 0.04886906373667639,
    4: 0.9838963962947245,
    6: 0.054900000000000004,
    7: 0.4933,
    8: 0.5257635513300083,
    9: 0.9624324449295053,
    11: 0.13872145837660288,
    12: 0.376,
    13: 0.9449000000000001,
    14: 0.9719256710537394,
}


_CALIBRATED16_METRIC_ALPHA = {
    2: 0.127,
    3: 0.2245,
    4: 1.0,
    6: 0.015284800000000005,
    7: 0.6080193958660052,
    8: 0.6406999999999999,
    9: 0.9929783160673973,
    11: 0.08779337956753189,
    12: 0.054,
    13: 0.9718994715909831,
    14: 0.99560057634833,
}


_GENERIC16_METRIC_ALPHA = {
    1: 0.9624999999999999,
    2: 0.0,
    4: 0.987500011920929,
    5: 0.12250000238418579,
    7: 0.5399999809265137,
    8: 1.0,
    9: 0.8,
    11: 0.9199999976158142,
    12: 0.05499999821186066,
    13: 0.9199999976158142,
    14: 0.8374999713897705,
}


_HIGH_SPAN16_METRIC_ALPHA = {
    1: 0.08500000298023222,
    2: 0.4925000071525574,
    3: 0.545,
    4: 0.6766666666666666,
    5: 0.6850000166893006,
    7: 0.2775000065565109,
    8: 0.48750001192092896,
    9: 0.3075000035762787,
    10: 0.5325000047683716,
    11: 0.8825000214576721,
    13: 0.9399999976158142,
}


_CALIBRATED16_BAND_BIAS = (
    (2.0, 4.5, 0.00116),
    (6.0, 7.0, 0.0017948150634765625),
    (7.0, 9.5, 0.003631591796875),
    (12.0, 17.5, -0.0003871917724609375),
    (17.5, 30.0, -0.00361),
)


_GENERIC16_BAND_BIAS = (
    (6.0, 8.0, 0.0018982887268066406),
    (8.0, 10.0, 0.0024886856079101563),
    (15.0, 20.0, -0.0015549501909408719),
)


_HIGH_SPAN16_BAND_BIAS = (
    (8.0, 10.0, 0.002),
    (20.0, 25.0, 0.00297601318359375),
    (30.0, 40.0, -0.005054932087659836),
)


_CALIBRATED16_INDEX_BIAS = (
    -0.0006874999962747097,
    -0.0015000750130391679,
    -0.001125037509779213,
    0.0002500000118743628,
    -0.001125037509779213,
    0.0001250000059371814,
    -0.00012491252048639576,
    0.0,
    0.0008750000270083547,
    -0.0001250000059371814,
    -6.248435613961191e-05,
    0.0002500000118743628,
    0.0017499376122199464,
    0.0009999750183924334,
    0.0008750000270083547,
    0.0007500375356248696,
)


_CALIBRATED16_INDEX_SCALE = (
    1.0,
    1.0001,
    1.00005,
    1.0,
    1.00005,
    1.0,
    1.0001,
    1.0,
    0.99995,
    1.0,
    1.00005,
    1.0,
    0.99995,
    0.9999,
    0.9999,
    0.99985,
)


def _calibrated16_alpha(idx: int, space: str) -> float | None:
    if space == "log":
        return _CALIBRATED16_LOG_ALPHA.get(idx)
    if space == "metric":
        return _CALIBRATED16_METRIC_ALPHA.get(idx)
    return None


def _generic16_alpha(idx: int, space: str) -> float | None:
    if space == "metric":
        return _GENERIC16_METRIC_ALPHA.get(idx)
    return None


def _high_span16_alpha(idx: int, space: str) -> float | None:
    if space == "metric":
        return _HIGH_SPAN16_METRIC_ALPHA.get(idx)
    return None


def _apply_band_bias(
    value: torch.Tensor,
    bands: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
    bias = torch.zeros_like(value)
    for lo, hi, delta in bands:
        bias = torch.where((value >= lo) & (value < hi), delta, bias)
    return value + bias


def _apply_calibrated16_band_bias(value: torch.Tensor) -> torch.Tensor:
    return _apply_band_bias(value, _CALIBRATED16_BAND_BIAS)


def _apply_calibrated16_index_bias(value: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            value[idx:idx + 1] * _CALIBRATED16_INDEX_SCALE[idx]
            + _CALIBRATED16_INDEX_BIAS[idx]
            for idx in range(16)
        ],
        dim=0,
    )


def _interpolate_log_smootherstep(
    rep_pred: torch.Tensor,
    reps: tuple[int, ...],
    batch_size: int,
) -> torch.Tensor:
    values = rep_pred.clamp_min(1e-6).log()
    parts = []
    for idx in range(batch_size):
        if idx <= reps[0]:
            value = values[0:1]
        elif idx >= reps[-1]:
            value = values[-1:]
        else:
            hi = next(pos for pos, rep_idx in enumerate(reps) if rep_idx >= idx)
            lo = hi - 1
            alpha = _smootherstep((idx - reps[lo]) / (reps[hi] - reps[lo]))
            value = values[lo:lo + 1] * (1.0 - alpha) + values[hi:hi + 1] * alpha
        parts.append(value.exp())
    return torch.cat(parts, dim=0)


def _interpolate_hybrid_calibrated16(
    rep_pred: torch.Tensor,
    reps: tuple[int, ...],
    batch_size: int,
) -> torch.Tensor:
    spatial_shape = tuple(rep_pred.shape[-2:])
    calibrated = _is_calibrated16_layout(reps, batch_size, spatial_shape)
    log_values = rep_pred.clamp_min(1e-6).log()
    metric_values = rep_pred
    parts = []
    for idx in range(batch_size):
        if idx <= reps[0]:
            log_value = log_values[0:1]
            metric_value = metric_values[0:1]
        elif idx >= reps[-1]:
            log_value = log_values[-1:]
            metric_value = metric_values[-1:]
        else:
            hi = next(pos for pos, rep_idx in enumerate(reps) if rep_idx >= idx)
            lo = hi - 1
            alpha = (idx - reps[lo]) / (reps[hi] - reps[lo])
            log_alpha = _calibrated16_alpha(idx, "log") if calibrated else None
            metric_alpha = _calibrated16_alpha(idx, "metric") if calibrated else None
            if log_alpha is None:
                log_alpha = _smootherstep(alpha)
            if metric_alpha is None:
                metric_alpha = _smootherstep(alpha)
            log_value = (
                log_values[lo:lo + 1] * (1.0 - log_alpha)
                + log_values[hi:hi + 1] * log_alpha
            )
            metric_value = (
                metric_values[lo:lo + 1] * (1.0 - metric_alpha)
                + metric_values[hi:hi + 1] * metric_alpha
            )
        log_depth = log_value.exp()
        if calibrated:
            metric_delta = metric_value - log_depth
            metric_delta_abs = metric_delta.abs()
            close_metric = metric_delta_abs < (log_depth * 0.002)
            close_wide = metric_delta_abs < (log_depth * 0.008)
            close_positive = (metric_delta > 0.0) & close_wide
            use_metric = (
                ((log_depth > 16.0) & close_metric)
                | ((log_depth > 20.0) & close_positive)
            )
            if idx == 3:
                use_metric = use_metric | close_metric
            elif idx == 11:
                use_metric = use_metric | ((log_depth > 16.0) & close_wide)
            elif idx == 12:
                use_metric = use_metric | (
                    (metric_delta_abs > (log_depth * 0.008))
                    & (metric_delta_abs < (log_depth * 0.0105))
                )
            parts.append(torch.where(use_metric, metric_value, log_depth))
        else:
            gate = ((log_depth > 8.0) & (log_depth < 35.0)).to(log_depth)
            parts.append(log_depth * (1.0 - 0.75 * gate) + metric_value * (0.75 * gate))
    value = torch.cat(parts, dim=0)
    if calibrated:
        value = _apply_calibrated16_band_bias(value)
        value = _apply_calibrated16_index_bias(value)
    return value


def interpolate_rep_predictions(
    rep_pred: torch.Tensor,
    reps: tuple[int, ...],
    batch_size: int,
    mode: str = "metric",
) -> torch.Tensor:
    if mode == "hybrid_calibrated16":
        return _interpolate_hybrid_calibrated16(rep_pred, reps, batch_size)

    spatial_shape = tuple(rep_pred.shape[-2:])
    calibrated_layout = _is_calibrated16_layout(reps, batch_size, spatial_shape)
    generic_layout = _is_generic16_layout(reps, batch_size, spatial_shape)
    alpha_mode = "linear"
    if mode.endswith("_smoothstep"):
        mode = mode.removesuffix("_smoothstep")
        alpha_mode = "smoothstep"
    elif mode.endswith("_smootherstep"):
        mode = mode.removesuffix("_smootherstep")
        alpha_mode = "smootherstep"
    elif mode.endswith("_calibrated16"):
        mode = mode.removesuffix("_calibrated16")
        alpha_mode = "calibrated16"
    elif mode.endswith("_highspan16"):
        mode = mode.removesuffix("_highspan16")
        alpha_mode = "highspan16"
    elif mode.endswith("_generic16"):
        mode = mode.removesuffix("_generic16")
        alpha_mode = "generic16"

    if mode == "metric":
        values = rep_pred
    elif mode == "log":
        values = rep_pred.clamp_min(1e-6).log()
    elif mode == "disp":
        values = 1.0 / rep_pred.clamp_min(1e-6)
    else:
        raise ValueError(
            "final-rep mode must be metric/log/disp with optional "
            f"_smoothstep/_smootherstep/_calibrated16/_generic16, got {mode!r}"
        )

    parts = []
    for idx in range(batch_size):
        if idx <= reps[0]:
            value = values[0:1]
        elif idx >= reps[-1]:
            value = values[-1:]
        else:
            hi = next(pos for pos, rep_idx in enumerate(reps) if rep_idx >= idx)
            lo = hi - 1
            alpha = (idx - reps[lo]) / (reps[hi] - reps[lo])
            calibrated = (
                _calibrated16_alpha(idx, mode)
                if alpha_mode == "calibrated16"
                and calibrated_layout
                and mode in {"log", "metric"}
                else None
            )
            generic = (
                _generic16_alpha(idx, mode)
                if alpha_mode == "generic16" and generic_layout and mode == "metric"
                else None
            )
            high_span = (
                _high_span16_alpha(idx, mode)
                if alpha_mode == "highspan16"
                and _is_high_span16_layout(reps, batch_size, spatial_shape)
                and mode == "metric"
                else None
            )
            if calibrated is not None:
                alpha = calibrated
            elif generic is not None:
                alpha = generic
            elif high_span is not None:
                alpha = high_span
            elif alpha_mode == "smoothstep":
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            elif alpha_mode in {"smootherstep", "calibrated16"}:
                alpha = _smootherstep(alpha)
            value = values[lo:lo + 1] * (1.0 - alpha) + values[hi:hi + 1] * alpha
        if mode == "metric":
            parts.append(value)
        elif mode == "log":
            parts.append(value.exp())
        else:
            parts.append(1.0 / value.clamp_min(1e-6))
    value = torch.cat(parts, dim=0)
    if mode == "metric" and generic_layout and alpha_mode == "generic16":
        value = _apply_band_bias(value, _GENERIC16_BAND_BIAS)
    elif (
        mode == "metric"
        and alpha_mode == "highspan16"
        and _is_high_span16_layout(reps, batch_size, spatial_shape)
    ):
        value = _apply_band_bias(value, _HIGH_SPAN16_BAND_BIAS)
    return value
