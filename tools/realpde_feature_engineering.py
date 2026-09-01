#!/usr/bin/env python3
"""Feature construction helpers for RealPDE Track 1.

The competition input convention is channels-last:

    (batch, time=20, height=32, width=64, channels=3)

where channels are approximately ``u, v, p``.  These helpers keep that layout
and append deterministic, submission-safe features derived only from the given
input frames.  No training statistics, metadata, or future frames are used.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


RAW_CHANNEL_NAMES = ("u", "v", "p")
DERIVED_UV_NAMES = (
    "speed",
    "kinetic_energy",
    "du_dt",
    "dv_dt",
    "vorticity",
    "divergence",
    "strain_magnitude",
    "x_coord",
    "y_coord",
    "t_coord",
)
FUTURE_CONTEXT_NAMES = (
    "hist_u_mean",
    "hist_v_mean",
    "hist_p_mean",
    "hist_u_std",
    "hist_v_std",
    "hist_p_std",
    "last_u_minus_hist_mean",
    "last_v_minus_hist_mean",
    "last_p_minus_hist_mean",
    "last_u_trend",
    "last_v_trend",
    "last_p_trend",
    "hist_speed_mean",
    "hist_speed_std",
    "last_speed_minus_hist_mean",
    "edge_x_distance",
    "edge_y_distance",
    "edge_min_distance",
)


def feature_names(include_pressure: bool = True) -> list[str]:
    """Return channel names emitted by :func:`augment_numpy`/``augment_torch``."""

    raw = list(RAW_CHANNEL_NAMES if include_pressure else RAW_CHANNEL_NAMES[:2])
    return raw + list(DERIVED_UV_NAMES)


def future_context_feature_names() -> list[str]:
    """Return channel names emitted by :func:`future_context_torch`."""

    return list(FUTURE_CONTEXT_NAMES)


def future_context_feature_count() -> int:
    """Return the number of history/boundary context features."""

    return len(FUTURE_CONTEXT_NAMES)


def _validate_channels_last_shape(shape: Sequence[int]) -> None:
    if len(shape) < 5:
        raise ValueError(
            "expected shape (..., time, height, width, channels), "
            f"got {tuple(shape)}"
        )
    if shape[-1] < 2:
        raise ValueError(f"expected at least u/v channels, got channel count {shape[-1]}")


def _central_diff_numpy(arr: np.ndarray, axis: int) -> np.ndarray:
    """Central finite difference with one-sided edges, no wraparound."""

    axis = axis % arr.ndim
    out = np.empty_like(arr)
    n = arr.shape[axis]
    if n <= 1:
        out[...] = 0.0
        return out

    middle = [slice(None)] * arr.ndim
    before = [slice(None)] * arr.ndim
    after = [slice(None)] * arr.ndim
    middle[axis] = slice(1, -1)
    before[axis] = slice(None, -2)
    after[axis] = slice(2, None)
    out[tuple(middle)] = 0.5 * (arr[tuple(after)] - arr[tuple(before)])

    first = [slice(None)] * arr.ndim
    second = [slice(None)] * arr.ndim
    first[axis] = 0
    second[axis] = 1
    out[tuple(first)] = arr[tuple(second)] - arr[tuple(first)]

    last = [slice(None)] * arr.ndim
    prev = [slice(None)] * arr.ndim
    last[axis] = -1
    prev[axis] = -2
    out[tuple(last)] = arr[tuple(last)] - arr[tuple(prev)]
    return out


def _backward_diff_numpy(arr: np.ndarray, axis: int) -> np.ndarray:
    """Backward difference with a zero first slice."""

    axis = axis % arr.ndim
    out = np.zeros_like(arr)
    n = arr.shape[axis]
    if n <= 1:
        return out
    current = [slice(None)] * arr.ndim
    previous = [slice(None)] * arr.ndim
    current[axis] = slice(1, None)
    previous[axis] = slice(None, -1)
    out[tuple(current)] = arr[tuple(current)] - arr[tuple(previous)]
    return out


def _coordinate_features_numpy(
    shape_without_channels: tuple[int, ...],
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast x/y/t coordinates to ``(..., T, H, W)``."""

    if len(shape_without_channels) < 4:
        raise ValueError(f"expected (..., T, H, W), got {shape_without_channels}")
    t, h, w = shape_without_channels[-3:]
    leading = (1,) * (len(shape_without_channels) - 3)
    x = np.linspace(-1.0, 1.0, w, dtype=dtype).reshape(leading + (1, 1, w))
    y = np.linspace(-1.0, 1.0, h, dtype=dtype).reshape(leading + (1, h, 1))
    time = np.linspace(-1.0, 1.0, t, dtype=dtype).reshape(leading + (t, 1, 1))
    return (
        np.broadcast_to(x, shape_without_channels),
        np.broadcast_to(y, shape_without_channels),
        np.broadcast_to(time, shape_without_channels),
    )


def augment_numpy(
    x: np.ndarray,
    *,
    include_pressure: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Append deterministic fluid features to a channels-last NumPy array.

    Parameters
    ----------
    x:
        Input array shaped ``(..., T, H, W, C)``.  Competition data uses
        ``(N, 20, 32, 64, 3)``.
    include_pressure:
        Whether to keep the raw pressure channel as a feature.  Most current
        Track 1 scoring appears to measure u/v only, but p can still help a
        model infer flow state when present.
    eps:
        Numerical stabilizer for ``sqrt``.
    """

    arr = np.asarray(x, dtype=np.float32)
    _validate_channels_last_shape(arr.shape)

    u = arr[..., 0]
    v = arr[..., 1]
    p = arr[..., 2] if arr.shape[-1] >= 3 else np.zeros_like(u)

    du_dt = _backward_diff_numpy(u, axis=-3)
    dv_dt = _backward_diff_numpy(v, axis=-3)
    du_dy = _central_diff_numpy(u, axis=-2)
    du_dx = _central_diff_numpy(u, axis=-1)
    dv_dy = _central_diff_numpy(v, axis=-2)
    dv_dx = _central_diff_numpy(v, axis=-1)

    speed = np.sqrt(u * u + v * v + eps).astype(np.float32, copy=False)
    kinetic = (0.5 * (u * u + v * v)).astype(np.float32, copy=False)
    vorticity = (dv_dx - du_dy).astype(np.float32, copy=False)
    divergence = (du_dx + dv_dy).astype(np.float32, copy=False)
    strain = np.sqrt((du_dx - dv_dy) ** 2 + (du_dy + dv_dx) ** 2 + eps).astype(
        np.float32,
        copy=False,
    )
    x_coord, y_coord, t_coord = _coordinate_features_numpy(u.shape, arr.dtype)

    channels: list[np.ndarray] = [u, v]
    if include_pressure:
        channels.append(p)
    channels.extend(
        [
            speed,
            kinetic,
            du_dt,
            dv_dt,
            vorticity,
            divergence,
            strain,
            x_coord,
            y_coord,
            t_coord,
        ]
    )
    return np.stack(channels, axis=-1).astype(np.float32, copy=False)


def augment_torch(x, *, include_pressure: bool = True, eps: float = 1e-6):
    """Torch equivalent of :func:`augment_numpy`.

    ``torch`` is imported lazily so the NumPy feature tests can run on machines
    without a local PyTorch install.
    """

    import torch

    if x.dtype != torch.float32:
        x = x.float()
    _validate_channels_last_shape(tuple(x.shape))

    u = x[..., 0]
    v = x[..., 1]
    p = x[..., 2] if x.shape[-1] >= 3 else torch.zeros_like(u)

    du_dt = _backward_diff_torch(u, axis=-3)
    dv_dt = _backward_diff_torch(v, axis=-3)
    du_dy = _central_diff_torch(u, axis=-2)
    du_dx = _central_diff_torch(u, axis=-1)
    dv_dy = _central_diff_torch(v, axis=-2)
    dv_dx = _central_diff_torch(v, axis=-1)

    speed = torch.sqrt(u * u + v * v + eps)
    kinetic = 0.5 * (u * u + v * v)
    vorticity = dv_dx - du_dy
    divergence = du_dx + dv_dy
    strain = torch.sqrt((du_dx - dv_dy) ** 2 + (du_dy + dv_dx) ** 2 + eps)
    x_coord, y_coord, t_coord = _coordinate_features_torch(u)

    channels = [u, v]
    if include_pressure:
        channels.append(p)
    channels.extend(
        [
            speed,
            kinetic,
            du_dt,
            dv_dt,
            vorticity,
            divergence,
            strain,
            x_coord,
            y_coord,
            t_coord,
        ]
    )
    return torch.stack(channels, dim=-1)


def future_context_torch(x, out_steps: int, *, eps: float = 1e-6):
    """Build future-aligned history and boundary context features.

    The residual corrector predicts the next 20 frames from a frozen CNO
    forecast.  These context features summarize the observed 20-frame history
    and repeat it along the future time axis, so the corrector can distinguish
    steady regions, recently accelerating regions, and near-boundary cells
    without seeing any target/future data.
    """

    import torch

    if x.dtype != torch.float32:
        x = x.float()
    _validate_channels_last_shape(tuple(x.shape))
    if out_steps <= 0:
        raise ValueError(f"out_steps must be positive, got {out_steps}")

    if x.shape[-1] >= 3:
        raw = x[..., :3].clone()
    else:
        raw = torch.cat([x[..., :2], torch.zeros_like(x[..., :1])], dim=-1)
    raw[..., 2] = 0.0

    batch, _, height, width, _ = raw.shape
    last = raw[:, -1:]
    if raw.shape[1] > 1:
        trend = raw[:, -1:] - raw[:, -2:-1]
    else:
        trend = torch.zeros_like(last)

    hist_mean = raw.mean(dim=1, keepdim=True)
    hist_std = raw.std(dim=1, keepdim=True, unbiased=False)
    last_minus_mean = last - hist_mean

    speed = torch.sqrt(raw[..., 0] * raw[..., 0] + raw[..., 1] * raw[..., 1] + eps)
    speed_mean = speed.mean(dim=1, keepdim=True).unsqueeze(-1)
    speed_std = speed.std(dim=1, keepdim=True, unbiased=False).unsqueeze(-1)
    last_speed_minus_mean = (speed[:, -1:] - speed_mean.squeeze(-1)).unsqueeze(-1)

    x_coord = torch.linspace(-1.0, 1.0, width, device=raw.device, dtype=raw.dtype).view(
        1,
        1,
        1,
        width,
        1,
    )
    y_coord = torch.linspace(-1.0, 1.0, height, device=raw.device, dtype=raw.dtype).view(
        1,
        1,
        height,
        1,
        1,
    )
    edge_x = (1.0 - torch.abs(x_coord)).clamp_min(0.0).expand(batch, out_steps, height, width, 1)
    edge_y = (1.0 - torch.abs(y_coord)).clamp_min(0.0).expand(batch, out_steps, height, width, 1)
    edge_min = torch.minimum(edge_x, edge_y)

    def repeat_future(value):
        return value.expand(-1, out_steps, -1, -1, -1)

    return torch.cat(
        [
            repeat_future(hist_mean),
            repeat_future(hist_std),
            repeat_future(last_minus_mean),
            repeat_future(trend),
            repeat_future(speed_mean),
            repeat_future(speed_std),
            repeat_future(last_speed_minus_mean),
            edge_x,
            edge_y,
            edge_min,
        ],
        dim=-1,
    )


def _central_diff_torch(arr, axis: int):
    import torch

    axis = axis % arr.ndim
    out = torch.empty_like(arr)
    n = arr.shape[axis]
    if n <= 1:
        out.zero_()
        return out

    middle = [slice(None)] * arr.ndim
    before = [slice(None)] * arr.ndim
    after = [slice(None)] * arr.ndim
    middle[axis] = slice(1, -1)
    before[axis] = slice(None, -2)
    after[axis] = slice(2, None)
    out[tuple(middle)] = 0.5 * (arr[tuple(after)] - arr[tuple(before)])

    first = [slice(None)] * arr.ndim
    second = [slice(None)] * arr.ndim
    first[axis] = 0
    second[axis] = 1
    out[tuple(first)] = arr[tuple(second)] - arr[tuple(first)]

    last = [slice(None)] * arr.ndim
    prev = [slice(None)] * arr.ndim
    last[axis] = -1
    prev[axis] = -2
    out[tuple(last)] = arr[tuple(last)] - arr[tuple(prev)]
    return out


def _backward_diff_torch(arr, axis: int):
    import torch

    axis = axis % arr.ndim
    out = torch.zeros_like(arr)
    n = arr.shape[axis]
    if n <= 1:
        return out
    current = [slice(None)] * arr.ndim
    previous = [slice(None)] * arr.ndim
    current[axis] = slice(1, None)
    previous[axis] = slice(None, -1)
    out[tuple(current)] = arr[tuple(current)] - arr[tuple(previous)]
    return out


def _coordinate_features_torch(u):
    import torch

    shape = tuple(u.shape)
    t, h, w = shape[-3:]
    leading = (1,) * (len(shape) - 3)
    x = torch.linspace(-1.0, 1.0, w, device=u.device, dtype=u.dtype).reshape(
        leading + (1, 1, w)
    )
    y = torch.linspace(-1.0, 1.0, h, device=u.device, dtype=u.dtype).reshape(
        leading + (1, h, 1)
    )
    time = torch.linspace(-1.0, 1.0, t, device=u.device, dtype=u.dtype).reshape(
        leading + (t, 1, 1)
    )
    return x.expand(shape), y.expand(shape), time.expand(shape)


def augment(x, *, include_pressure: bool = True, eps: float = 1e-6):
    """Dispatch to the NumPy or Torch implementation."""

    module_name = type(x).__module__.split(".", 1)[0]
    if module_name == "torch":
        return augment_torch(x, include_pressure=include_pressure, eps=eps)
    return augment_numpy(x, include_pressure=include_pressure, eps=eps)


if __name__ == "__main__":
    sample = np.zeros((2, 20, 32, 64, 3), dtype=np.float32)
    features = augment_numpy(sample)
    print({"shape": features.shape, "features": feature_names()})
