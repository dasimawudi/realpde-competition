#!/usr/bin/env python3
"""Direct-HDF5 feature-adapter training for RealPDE Track 1.

Some competition workspaces store the released trajectories as top-level HDF5
fields (``u``, ``v``, ``x``, ``y``, ``re``, ``aoa``) instead of the historical
RealPDEBench ``Foil`` dataset layout.  This script reads those files directly
and trains the same conservative residual adapter used by
``realpde_feature_adapter_train.py``.

The submission interface remains unchanged: input and output tensors are both
channels-last ``[B, 20, 32, 64, 3]`` arrays.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from realpde_feature_engineering import augment_torch, feature_names  # noqa: E402


BAD_TRAIN_FILES = {"7575_0.h5"}
SIGMA_GLOBAL = 0.0563870259
T_NUMERICAL_SEC = 0.72896


@dataclass(frozen=True)
class WindowRef:
    path: Path
    start: int


def list_h5(root: Path, excludes: set[str]) -> list[Path]:
    paths = sorted(path for path in root.glob("*.h5") if path.name not in excludes)
    if not paths:
        raise ValueError(f"no usable .h5 files found in {root}")
    return paths


def split_paths(paths: Sequence[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if len(paths) < 2:
        raise ValueError("need at least two trajectories for a trajectory-level split")
    rng = np.random.default_rng(seed)
    order = np.arange(len(paths))
    rng.shuffle(order)
    n_val = min(max(1, round(len(paths) * val_fraction)), len(paths) - 1)
    val_ids = set(order[:n_val].tolist())
    train = [path for idx, path in enumerate(paths) if idx not in val_ids]
    val = [path for idx, path in enumerate(paths) if idx in val_ids]
    return train, val


def h5_field(handle: h5py.File, key: str):
    if key in handle:
        return handle[key]
    nested = f"measured_data/{key}"
    if nested in handle:
        return handle[nested]
    raise KeyError(key)


class H5WindowDataset(Dataset):
    """Windowed ``u/v/p`` loader for released RealPDE HDF5 trajectories."""

    def __init__(
        self,
        paths: Sequence[Path],
        *,
        in_steps: int = 20,
        out_steps: int = 20,
        stride: int = 20,
        sub_sample: int = 2,
        max_windows_per_trajectory: int | None = None,
        include_pressure: bool = False,
    ) -> None:
        if min(in_steps, out_steps, stride, sub_sample) < 1:
            raise ValueError("in_steps, out_steps, stride, and sub_sample must be positive")
        self.in_steps = int(in_steps)
        self.out_steps = int(out_steps)
        self.sub_sample = int(sub_sample)
        self.include_pressure = bool(include_pressure)
        self.paths = list(paths)
        self.refs: list[WindowRef] = []
        total = self.in_steps + self.out_steps
        for path in self.paths:
            with h5py.File(path, "r") as handle:
                length = int(h5_field(handle, "u").shape[0])
            starts = list(range(0, length - total + 1, stride))
            if max_windows_per_trajectory is not None:
                starts = starts[:max_windows_per_trajectory]
            self.refs.extend(WindowRef(path, start) for start in starts)
        if not self.refs:
            raise ValueError("requested windows do not fit in the provided trajectories")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        ref = self.refs[index]
        total = self.in_steps + self.out_steps
        with h5py.File(ref.path, "r") as handle:
            sl = slice(ref.start, ref.start + total)
            ss = self.sub_sample
            u = np.asarray(h5_field(handle, "u")[sl, ::ss, ::ss], dtype=np.float32)
            v = np.asarray(h5_field(handle, "v")[sl, ::ss, ::ss], dtype=np.float32)
            if self.include_pressure:
                try:
                    p = np.asarray(h5_field(handle, "p")[sl, ::ss, ::ss], dtype=np.float32)
                except KeyError:
                    p = np.zeros_like(u)
            else:
                p = np.zeros_like(u)
        full = torch.from_numpy(np.stack([u, v, p], axis=-1))
        return full[: self.in_steps], full[self.in_steps :]


class PointwiseFeatureAdapter(nn.Module):
    """Residual per-point projection from engineered features back to u/v/p."""

    def __init__(
        self,
        *,
        hidden: int = 32,
        include_pressure: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.include_pressure = bool(include_pressure)
        self.dropout = float(dropout)
        self.names = feature_names(include_pressure=self.include_pressure)
        n_features = len(self.names)
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, self.hidden),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 3),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def config(self) -> dict[str, object]:
        return {
            "hidden": self.hidden,
            "include_pressure": self.include_pressure,
            "dropout": self.dropout,
            "feature_names": self.names,
        }

    def forward(self, x: Tensor) -> Tensor:
        features = augment_torch(x, include_pressure=self.include_pressure)
        return x[..., :3] + self.net(features)


class FeatureAdapterModel(nn.Module):
    def __init__(self, base_model: nn.Module, adapter: PointwiseFeatureAdapter) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter

    def forward(self, x: Tensor) -> Tensor:
        return self.base_model(self.adapter(x))


def load_cno_class(realpdebench_root: Path):
    sys.path.insert(0, str(realpdebench_root))
    from realpdebench.model.cno import CNO3d

    return CNO3d


def unwrap_state(checkpoint) -> dict[str, Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"unsupported checkpoint type: {type(checkpoint)!r}")


def load_cno_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = unwrap_state(checkpoint)
    fixed = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.", "base_model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        fixed[new_key] = value
    result = model.load_state_dict(fixed, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(
            "base CNO checkpoint mismatch: "
            f"missing_keys={missing[:20]}, unexpected_keys={unexpected[:20]}"
        )
    return checkpoint if isinstance(checkpoint, dict) else {}


def rel_l2_loss_torch(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    bsz = pred.shape[0]
    p = pred.reshape(bsz, -1)
    t = target.reshape(bsz, -1)
    return (torch.linalg.norm(p - t, dim=1) / torch.linalg.norm(t, dim=1).clamp_min(eps)).mean()


def kinetic_energy_torch(x: Tensor) -> Tensor:
    u = x[..., 0]
    v = x[..., 1]
    u_prime = ((u - u.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
    v_prime = ((v - v.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
    return 0.5 * (u_prime + v_prime)


def physics_loss(pred: Tensor, target: Tensor, weights: dict[str, float]) -> tuple[Tensor, dict[str, float]]:
    pred_uv = pred[..., :2]
    target_uv = target[..., :2]
    point = rel_l2_loss_torch(pred_uv, target_uv)
    mse = torch.mean((pred_uv - target_uv) ** 2)
    pred_ke = kinetic_energy_torch(pred_uv)
    target_ke = kinetic_energy_torch(target_uv)
    tke = rel_l2_loss_torch(pred_ke, target_ke)
    temporal = rel_l2_loss_torch(pred_uv[:, 1:] - pred_uv[:, :-1], target_uv[:, 1:] - target_uv[:, :-1])
    pred_dx = pred_uv[:, :, :, 1:] - pred_uv[:, :, :, :-1]
    target_dx = target_uv[:, :, :, 1:] - target_uv[:, :, :, :-1]
    pred_dy = pred_uv[:, :, 1:, :] - pred_uv[:, :, :-1, :]
    target_dy = target_uv[:, :, 1:, :] - target_uv[:, :, :-1, :]
    grad = 0.5 * rel_l2_loss_torch(pred_dx, target_dx) + 0.5 * rel_l2_loss_torch(pred_dy, target_dy)
    p_zero = torch.mean(pred[..., 2] ** 2) if pred.shape[-1] > 2 else pred.new_tensor(0.0)
    loss = (
        weights["point"] * point
        + weights["mse"] * mse
        + weights["tke"] * tke
        + weights["temporal"] * temporal
        + weights["grad"] * grad
        + weights["p_zero"] * p_zero
    )
    parts = {
        "loss": float(loss.detach().cpu()),
        "point_rel": float(point.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "tke_rel": float(tke.detach().cpu()),
        "temporal_rel": float(temporal.detach().cpu()),
        "grad_rel": float(grad.detach().cpu()),
        "p_zero": float(p_zero.detach().cpu()),
    }
    return loss, parts


def measured_channels(target: np.ndarray) -> int:
    active = [not np.allclose(target[..., idx], 0.0) for idx in range(target.shape[-1])]
    return max(1, int(sum(active)))


def rel_l2_per_sample(pred: np.ndarray, target: np.ndarray, channels: int) -> np.ndarray:
    p = pred[..., :channels].reshape(pred.shape[0], -1)
    t = target[..., :channels].reshape(target.shape[0], -1)
    return np.linalg.norm(p - t, axis=1) / np.linalg.norm(t, axis=1).clip(min=1e-8)


def kinetic_energy_np(x: np.ndarray) -> np.ndarray:
    u = x[..., 0]
    v = x[..., 1]
    u_prime = np.mean((u - np.mean(u, axis=1, keepdims=True)) ** 2, axis=1)
    v_prime = np.mean((v - np.mean(v, axis=1, keepdims=True)) ** 2, axis=1)
    return 0.5 * (u_prime + v_prime)


def tke_rel_l2_per_sample(pred: np.ndarray, target: np.ndarray, channels: int) -> np.ndarray:
    if channels < 2:
        return np.zeros((pred.shape[0],), dtype=np.float32)
    p = kinetic_energy_np(pred[..., :channels]).reshape(pred.shape[0], -1)
    t = kinetic_energy_np(target[..., :channels]).reshape(target.shape[0], -1)
    return np.linalg.norm(p - t, axis=1) / np.linalg.norm(t, axis=1).clip(min=1e-8)


def mvpe_rel_l2_per_sample(pred: np.ndarray, target: np.ndarray, sub_s_real: int = 2) -> np.ndarray:
    d = 16
    center_x = 10
    center_y = 32
    n_probe = 9
    n, _, height, width, _ = pred.shape
    probe_center_y = int(center_y / sub_s_real)
    interval_y = min(2, int(height / (n_probe + 1)))
    probe_y = [
        probe_center_y + interval_y * j
        for j in range(-(n_probe - 1) // 2, n_probe - (n_probe - 1) // 2)
    ]
    probe_y = [idx for idx in probe_y if 0 <= idx < height]
    errors = []
    for idx in range(4):
        if int((2 * d + center_x) / sub_s_real) < width:
            probe_x = int(((idx + 1) * d + center_x) / sub_s_real)
        else:
            probe_x = int((0.5 * (idx + 2) * d + center_x) / sub_s_real)
        if not 0 <= probe_x < width:
            continue
        pp = pred[:, :, probe_y, probe_x, :2].mean(axis=1).reshape(n, -1)
        tt = target[:, :, probe_y, probe_x, :2].mean(axis=1).reshape(n, -1)
        errors.append(np.linalg.norm(pp - tt, axis=1) / np.linalg.norm(tt, axis=1).clip(min=1e-8))
    if not errors:
        return np.zeros((n,), dtype=np.float32)
    return np.mean(np.stack(errors, axis=0), axis=0)


def score_error(err: float, scale: float = 0.5) -> float:
    if not np.isfinite(err):
        return 0.0
    return float(100.0 / (1.0 + abs(scale) * max(float(err), 0.0)))


def score_time(t_neural: float, r_min: float = 1.0) -> float:
    if not np.isfinite(t_neural) or t_neural <= 0.0:
        return 0.0
    ratio = float(t_neural) / T_NUMERICAL_SEC
    return float(100.0 / (1.0 + (ratio / r_min) ** 0.5))


def init_sps_candidates(abs_widths, rel_widths) -> list[dict[str, float]]:
    return [
        {
            "abs": float(abs_width),
            "rel": float(rel_width),
            "sum_dm": 0.0,
            "sum_tke": 0.0,
            "sum_mvpe": 0.0,
            "inside": 0,
            "count": 0,
        }
        for abs_width in abs_widths
        for rel_width in rel_widths
    ]


def update_sps_candidates(candidates, pred, target, channels, rel, tke, mvpe) -> None:
    p = pred[..., :channels]
    t = target[..., :channels]
    p_abs = np.abs(p)
    err_abs = np.abs(p - t)
    count = int(np.prod(p.shape))
    pm_dm = rel / (0.5 + rel)
    pm_tke = tke / (0.5 + tke)
    pm_mvpe = mvpe / (0.5 + mvpe)
    shape_pm = (p.shape[0],) + (1,) * (p.ndim - 1)
    w_dm = (1.0 - pm_dm).reshape(shape_pm)
    w_tke = (1.0 - pm_tke).reshape(shape_pm)
    w_mvpe = (1.0 - pm_mvpe).reshape(shape_pm)
    for cand in candidates:
        half = cand["abs"] + cand["rel"] * p_abs
        inside = err_abs <= half
        tight = np.exp(-(2.0 * half) / SIGMA_GLOBAL)
        elem_mask = inside * tight
        cand["sum_dm"] += float(np.sum(w_dm * elem_mask))
        cand["sum_tke"] += float(np.sum(w_tke * elem_mask))
        cand["sum_mvpe"] += float(np.sum(w_mvpe * elem_mask))
        cand["inside"] += int(np.sum(inside))
        cand["count"] += count


def finalize_scores(metric_sums: dict, candidates: list[dict]) -> dict[str, object]:
    n = max(metric_sums["n"], 1)
    rel_l2 = metric_sums["rel_l2_sum"] / n
    tke = metric_sums["tke_sum"] / n
    mvpe = metric_sums["mvpe_sum"] / n
    mean_t = metric_sums["time_sum"] / max(metric_sums["time_n"], 1)
    base = {
        "n": n,
        "rel_l2": rel_l2,
        "tke": tke,
        "mvpe": mvpe,
        "mean_t_neural_s": mean_t,
        "rel_l2_score": score_error(rel_l2),
        "tke_score": score_error(tke),
        "mvpe_score": score_error(mvpe),
        "time_score": score_time(mean_t),
    }
    rows = []
    for cand in candidates:
        count = max(cand["count"], 1)
        sps_raw = 0.5 * cand["sum_dm"] / count + 0.3 * cand["sum_tke"] / count + 0.2 * cand["sum_mvpe"] / count
        sps_score = 100.0 * sps_raw
        rows.append(
            {
                "abs": cand["abs"],
                "rel": cand["rel"],
                "coverage": cand["inside"] / count,
                "sps_raw": sps_raw,
                "sps_score_linear": sps_score,
                "sps_score_used": sps_score,
                "final_est": float(np.mean([base["rel_l2_score"], base["tke_score"], base["mvpe_score"], base["time_score"], sps_score])),
            }
        )
    rows.sort(key=lambda row: row["final_est"], reverse=True)
    base["best_bounds"] = rows[:20]
    return base


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, abs_widths, rel_widths, max_batches: int | None = None) -> dict:
    model.eval()
    metric_sums = {
        "n": 0,
        "rel_l2_sum": 0.0,
        "tke_sum": 0.0,
        "mvpe_sum": 0.0,
        "time_sum": 0.0,
        "time_n": 0,
    }
    candidates = init_sps_candidates(abs_widths, rel_widths)
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        pred = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        target_np = y.detach().cpu().numpy().astype(np.float32)
        pred_np[..., 2] = 0.0
        channels = measured_channels(target_np)
        rel = rel_l2_per_sample(pred_np, target_np, channels)
        tke = tke_rel_l2_per_sample(pred_np, target_np, channels)
        mvpe = mvpe_rel_l2_per_sample(pred_np, target_np)
        batch_n = pred_np.shape[0]
        metric_sums["n"] += batch_n
        metric_sums["rel_l2_sum"] += float(np.sum(rel))
        metric_sums["tke_sum"] += float(np.sum(tke))
        metric_sums["mvpe_sum"] += float(np.sum(mvpe))
        metric_sums["time_sum"] += float(elapsed)
        metric_sums["time_n"] += batch_n
        update_sps_candidates(candidates, pred_np, target_np, channels, rel, tke, mvpe)
    return finalize_scores(metric_sums, candidates)


def trainable_parameter_groups(
    model: FeatureAdapterModel,
    *,
    train_base: bool,
    adapter_lr: float,
    base_lr: float,
    weight_decay: float,
) -> list[dict]:
    for param in model.base_model.parameters():
        param.requires_grad = train_base
    for param in model.adapter.parameters():
        param.requires_grad = True
    groups = [
        {
            "params": [p for p in model.adapter.parameters() if p.requires_grad],
            "lr": adapter_lr,
            "weight_decay": weight_decay,
        }
    ]
    if train_base:
        groups.append(
            {
                "params": [p for p in model.base_model.parameters() if p.requires_grad],
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
        )
    return groups


def save_checkpoint(
    path: Path,
    model: FeatureAdapterModel,
    *,
    iteration: int,
    best_score: float,
    train_log: list[dict],
    eval_log: list[dict],
    run_config: dict,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "adapter_config": model.adapter.config(),
            "run_config": run_config,
            "train_losses": train_log,
            "val_losses": eval_log,
            "iteration": iteration,
            "best_iteration": iteration,
            "best_val_loss": -best_score,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--realpdebench-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--adapter-lr", type=float, default=3e-4)
    parser.add_argument("--base-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--adapter-delta",
        type=float,
        default=0.02,
        help="MSE penalty on adapter(input)-input to keep the frozen-CNO probe conservative.",
    )
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-base", action="store_true")
    parser.add_argument("--drop-pressure-feature", action="store_true")
    parser.add_argument("--include-pressure-data", action="store_true")
    parser.add_argument("--in-steps", type=int, default=20)
    parser.add_argument("--out-steps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--sub-sample", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-windows-per-trajectory", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--point", type=float, default=1.0)
    parser.add_argument("--mse", type=float, default=0.05)
    parser.add_argument("--tke", type=float, default=0.08)
    parser.add_argument("--temporal", type=float, default=0.04)
    parser.add_argument("--grad", type=float, default=0.02)
    parser.add_argument("--p-zero", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"out_dir already exists, refusing to overwrite: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    paths = list_h5(args.real_root, BAD_TRAIN_FILES)
    train_paths, val_paths = split_paths(paths, args.val_fraction, args.seed)
    train_set = H5WindowDataset(
        train_paths,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        stride=args.stride,
        sub_sample=args.sub_sample,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        include_pressure=args.include_pressure_data,
    )
    val_set = H5WindowDataset(
        val_paths,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        stride=args.stride,
        sub_sample=args.sub_sample,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        include_pressure=args.include_pressure_data,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    CNO3d = load_cno_class(args.realpdebench_root)
    base_model = CNO3d(
        in_dim=3,
        out_dim=3,
        out_dim_mult=1,
        in_size=64,
        N_layers=3,
        activation="LeakyReLU",
    ).to(device)
    checkpoint_meta = load_cno_checkpoint(base_model, args.checkpoint, device)
    adapter = PointwiseFeatureAdapter(
        hidden=args.hidden,
        include_pressure=not args.drop_pressure_feature,
        dropout=args.dropout,
    ).to(device)
    model = FeatureAdapterModel(base_model, adapter).to(device)

    groups = trainable_parameter_groups(
        model,
        train_base=args.train_base,
        adapter_lr=args.adapter_lr,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.updates)
    weights = {
        "point": args.point,
        "mse": args.mse,
        "tke": args.tke,
        "temporal": args.temporal,
        "grad": args.grad,
        "p_zero": args.p_zero,
    }
    abs_widths = np.array([0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030, 0.040], dtype=np.float32)
    rel_widths = np.array([0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20], dtype=np.float32)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    run_config = {
        "device": str(device),
        "real_root": str(args.real_root),
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": {
            "iteration": checkpoint_meta.get("iteration") if isinstance(checkpoint_meta, dict) else None,
            "best_iteration": checkpoint_meta.get("best_iteration") if isinstance(checkpoint_meta, dict) else None,
            "best_val_loss": str(checkpoint_meta.get("best_val_loss")) if isinstance(checkpoint_meta, dict) else None,
        },
        "realpdebench_root": str(args.realpdebench_root),
        "out_dir": str(args.out_dir),
        "adapter_config": adapter.config(),
        "train_base": args.train_base,
        "excluded_files": sorted(BAD_TRAIN_FILES),
        "train_trajectories": len(train_paths),
        "val_trajectories": len(val_paths),
        "train_windows": len(train_set),
        "val_windows": len(val_set),
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "updates": args.updates,
        "eval_interval": args.eval_interval,
        "max_eval_batches": args.max_eval_batches,
        "weights": weights,
        "adapter_delta": args.adapter_delta,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    print(json.dumps(run_config, indent=2, default=str), flush=True)

    train_log: list[dict] = []
    eval_log: list[dict] = []
    summary = evaluate(model, val_loader, device, abs_widths, rel_widths, args.max_eval_batches)
    summary["iteration"] = 0
    eval_log.append(summary)
    best_score = float(summary["best_bounds"][0]["final_est"])
    best_iter = 0
    best_summary = summary
    print("EVAL " + json.dumps(summary, sort_keys=True), flush=True)
    save_checkpoint(
        args.out_dir / "model_best.pth",
        model,
        iteration=0,
        best_score=best_score,
        train_log=train_log,
        eval_log=eval_log,
        run_config=run_config,
    )

    iterator = iter(train_loader)
    smooth: dict[str, float] = {}
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    model.train()
    for step in range(1, args.updates + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        adapted = model.adapter(x)
        pred = model.base_model(adapted)
        loss, parts = physics_loss(pred, y, weights)
        if args.adapter_delta > 0:
            delta_penalty = torch.mean((adapted - x[..., :3]) ** 2)
            loss = loss + args.adapter_delta * delta_penalty
            parts["adapter_delta_mse"] = float(delta_penalty.detach().cpu())
            parts["loss"] = float(loss.detach().cpu())
        loss.backward()
        if args.clip_grad and args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_grad)
        optimizer.step()
        scheduler.step()

        for key, value in parts.items():
            smooth[key] = 0.98 * smooth.get(key, value) + 0.02 * value
        if step % 20 == 0:
            row = {
                "iteration": step,
                "adapter_lr": optimizer.param_groups[0]["lr"],
                "base_lr": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
                **smooth,
            }
            train_log.append(row)
            print("TRAIN " + json.dumps(row, sort_keys=True), flush=True)
        if step % args.eval_interval == 0 or step == args.updates:
            summary = evaluate(model, val_loader, device, abs_widths, rel_widths, args.max_eval_batches)
            summary["iteration"] = step
            eval_log.append(summary)
            current_score = float(summary["best_bounds"][0]["final_est"])
            print("EVAL " + json.dumps(summary, sort_keys=True), flush=True)
            save_checkpoint(
                args.out_dir / "model_latest.pth",
                model,
                iteration=step,
                best_score=current_score,
                train_log=train_log,
                eval_log=eval_log,
                run_config=run_config,
            )
            if current_score > best_score:
                best_score = current_score
                best_iter = step
                best_summary = summary
                save_checkpoint(
                    args.out_dir / "model_best.pth",
                    model,
                    iteration=step,
                    best_score=best_score,
                    train_log=train_log,
                    eval_log=eval_log,
                    run_config=run_config,
                )
                print(f"BEST iteration={best_iter} final_est={best_score:.6f}", flush=True)
            (args.out_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "best_score": best_score,
                        "best_iter": best_iter,
                        "best_summary": best_summary,
                        "latest_summary": summary,
                        "run_config": run_config,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    save_checkpoint(
        args.out_dir / "model_final.pth",
        model,
        iteration=args.updates,
        best_score=best_score,
        train_log=train_log,
        eval_log=eval_log,
        run_config=run_config,
    )
    print(f"DONE out_dir={args.out_dir} best_iter={best_iter} best_score={best_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
