#!/usr/bin/env python3
"""Train a conservative feature residual corrector for RealPDE Track 1.

This route keeps the current best CNO as the frozen backbone and learns only a
small correction on top of its 20-step forecast:

    final_prediction = cno_prediction + alpha * residual_corrector(features)

The corrector sees deterministic, submission-safe features built from the input
frames and from the CNO forecast.  Its last layer is initialized to zero, so the
iteration-0 model is exactly the original CNO.  During validation we scan
``alpha``; if the best alpha is 0, the run is not worth submitting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from realpde_feature_engineering import augment_torch, feature_names  # noqa: E402
from realpde_h5_feature_adapter_train import (  # noqa: E402
    BAD_TRAIN_FILES,
    H5WindowDataset,
    finalize_scores,
    init_sps_candidates,
    list_h5,
    load_cno_checkpoint,
    load_cno_class,
    measured_channels,
    mvpe_rel_l2_per_sample,
    physics_loss,
    rel_l2_per_sample,
    split_paths,
    tke_rel_l2_per_sample,
    update_sps_candidates,
)


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.replace(";", ",").split(",") if item.strip()]


def ensure_three_channels(x: Tensor) -> Tensor:
    if x.shape[-1] >= 3:
        return x[..., :3]
    return torch.cat([x[..., :2], torch.zeros_like(x[..., :1])], dim=-1)


def zero_pressure(x: Tensor) -> Tensor:
    if x.shape[-1] < 3:
        return x
    y = x.clone()
    y[..., 2] = 0.0
    return y


def future_linear_extrapolation(x: Tensor, out_steps: int) -> Tensor:
    raw = ensure_three_channels(x)
    last = raw[:, -1:]
    if raw.shape[1] > 1:
        trend = raw[:, -1:] - raw[:, -2:-1]
    else:
        trend = torch.zeros_like(last)
    steps = torch.linspace(
        1.0 / float(out_steps),
        1.0,
        out_steps,
        device=x.device,
        dtype=x.dtype,
    ).view(1, out_steps, 1, 1, 1)
    return zero_pressure(last + steps * trend)


def future_feature_count(include_pressure: bool) -> int:
    # CNO forecast features + last-observed features + three 3-channel relation
    # blocks: linear extrapolation, CNO-last, CNO-linear.
    return 2 * len(feature_names(include_pressure=include_pressure)) + 9


def build_future_features(x: Tensor, base_pred: Tensor, *, include_pressure: bool) -> Tensor:
    base = zero_pressure(ensure_three_channels(base_pred))
    out_steps = int(base.shape[1])
    last_raw = ensure_three_channels(x[:, -1:]).expand(-1, out_steps, -1, -1, -1)
    last_raw = zero_pressure(last_raw)
    linear = future_linear_extrapolation(x, out_steps)

    base_features = augment_torch(base, include_pressure=include_pressure)
    past_features = augment_torch(ensure_three_channels(x), include_pressure=include_pressure)
    last_features = past_features[:, -1:].expand(-1, out_steps, -1, -1, -1)

    return torch.cat(
        [
            base_features,
            last_features,
            linear,
            base - last_raw,
            base - linear,
        ],
        dim=-1,
    )


def norm_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(channels), channels),
            nn.SiLU(),
            nn.Dropout3d(float(dropout)),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(channels), channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


@dataclass(frozen=True)
class CorrectorConfig:
    hidden: int = 32
    blocks: int = 2
    dropout: float = 0.0
    include_pressure: bool = True
    max_delta: float = 0.05


class ResidualCorrector3D(nn.Module):
    def __init__(self, config: CorrectorConfig) -> None:
        super().__init__()
        self.config = config
        in_channels = future_feature_count(include_pressure=config.include_pressure)
        hidden = int(config.hidden)
        self.input_norm = nn.LayerNorm(in_channels)
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(hidden), hidden),
            nn.SiLU(),
        ]
        for _ in range(int(config.blocks)):
            layers.append(ResidualBlock3D(hidden, dropout=config.dropout))
        layers.append(nn.Conv3d(hidden, 3, kernel_size=1))
        self.net = nn.Sequential(*layers)
        final = self.net[-1]
        if isinstance(final, nn.Conv3d):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, x: Tensor, base_pred: Tensor) -> Tensor:
        features = build_future_features(x, base_pred, include_pressure=self.config.include_pressure)
        features = self.input_norm(features)
        z = features.permute(0, 4, 1, 2, 3).contiguous()
        raw_delta = self.net(z).permute(0, 2, 3, 4, 1).contiguous()
        if self.config.max_delta > 0:
            delta = self.config.max_delta * torch.tanh(raw_delta / self.config.max_delta)
        else:
            delta = raw_delta
        delta = delta.clone()
        delta[..., 2] = 0.0
        return delta


class ResidualCorrectionModel(nn.Module):
    def __init__(self, base_model: nn.Module, corrector: ResidualCorrector3D) -> None:
        super().__init__()
        self.base_model = base_model
        self.corrector = corrector

    @torch.no_grad()
    def base_predict(self, x: Tensor) -> Tensor:
        base = self.base_model(ensure_three_channels(x))
        return zero_pressure(ensure_three_channels(base))

    def predict_delta(self, x: Tensor, base_pred: Tensor) -> Tensor:
        return self.corrector(ensure_three_channels(x), base_pred)

    def combine(self, base_pred: Tensor, delta: Tensor, alpha: float) -> Tensor:
        pred = base_pred + float(alpha) * delta
        return zero_pressure(pred)

    def forward(self, x: Tensor, alpha: float = 1.0) -> Tensor:
        base = self.base_predict(x)
        delta = self.predict_delta(x, base)
        return self.combine(base, delta, alpha)


def load_frozen_cno(checkpoint: Path, realpdebench_root: Path, device: torch.device) -> nn.Module:
    CNO3d = load_cno_class(realpdebench_root)
    model = CNO3d(
        in_dim=3,
        out_dim=3,
        out_dim_mult=1,
        in_size=64,
        N_layers=3,
        activation="LeakyReLU",
    ).to(device)
    load_cno_checkpoint(model, checkpoint, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def empty_tracker(abs_widths: Sequence[float], rel_widths: Sequence[float]) -> dict[str, object]:
    return {
        "metric_sums": {
            "n": 0,
            "rel_l2_sum": 0.0,
            "tke_sum": 0.0,
            "mvpe_sum": 0.0,
            "time_sum": 0.0,
            "time_n": 0,
        },
        "candidates": init_sps_candidates(abs_widths, rel_widths),
    }


def update_tracker(tracker: dict[str, object], pred_np: np.ndarray, target_np: np.ndarray, elapsed: float) -> None:
    pred_np = pred_np.astype(np.float32, copy=False)
    target_np = target_np.astype(np.float32, copy=False)
    pred_np[..., 2] = 0.0
    channels = measured_channels(target_np)
    rel = rel_l2_per_sample(pred_np, target_np, channels)
    tke = tke_rel_l2_per_sample(pred_np, target_np, channels)
    mvpe = mvpe_rel_l2_per_sample(pred_np, target_np)
    batch_n = int(pred_np.shape[0])
    metric_sums = tracker["metric_sums"]
    assert isinstance(metric_sums, dict)
    metric_sums["n"] += batch_n
    metric_sums["rel_l2_sum"] += float(np.sum(rel))
    metric_sums["tke_sum"] += float(np.sum(tke))
    metric_sums["mvpe_sum"] += float(np.sum(mvpe))
    metric_sums["time_sum"] += float(elapsed)
    metric_sums["time_n"] += batch_n
    update_sps_candidates(tracker["candidates"], pred_np, target_np, channels, rel, tke, mvpe)


@torch.no_grad()
def evaluate_alphas(
    model: ResidualCorrectionModel,
    loader: DataLoader,
    device: torch.device,
    *,
    alphas: Sequence[float],
    abs_widths: Sequence[float],
    rel_widths: Sequence[float],
    max_batches: int | None = None,
) -> list[dict[str, object]]:
    model.eval()
    trackers = {float(alpha): empty_tracker(abs_widths, rel_widths) for alpha in alphas}
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        base = model.base_predict(x)
        delta = model.predict_delta(x, base)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        base_np = base.detach().cpu().numpy().astype(np.float32, copy=False)
        delta_np = delta.detach().cpu().numpy().astype(np.float32, copy=False)
        target_np = y.detach().cpu().numpy().astype(np.float32, copy=False)
        for alpha, tracker in trackers.items():
            pred_np = base_np + float(alpha) * delta_np
            update_tracker(tracker, pred_np, target_np, elapsed)

    summaries: list[dict[str, object]] = []
    for alpha, tracker in trackers.items():
        summary = finalize_scores(tracker["metric_sums"], tracker["candidates"])
        summary["alpha"] = alpha
        summary["best_final_est"] = summary["best_bounds"][0]["final_est"]
        summary["best_bound_abs"] = summary["best_bounds"][0]["abs"]
        summary["best_bound_rel"] = summary["best_bounds"][0]["rel"]
        summaries.append(summary)
    summaries.sort(key=lambda row: float(row["best_final_est"]), reverse=True)
    return summaries


def save_checkpoint(
    path: Path,
    model: ResidualCorrectionModel,
    *,
    iteration: int,
    best_score: float,
    best_alpha: float,
    best_bound_abs: float,
    best_bound_rel: float,
    train_log: list[dict[str, object]],
    eval_log: list[dict[str, object]],
    run_config: dict[str, object],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "corrector_config": asdict(model.corrector.config),
            "run_config": run_config,
            "train_losses": train_log,
            "val_losses": eval_log,
            "iteration": int(iteration),
            "best_iteration": int(iteration),
            "best_score": float(best_score),
            "best_alpha": float(best_alpha),
            "best_bound_abs": float(best_bound_abs),
            "best_bound_rel": float(best_bound_rel),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--realpdebench-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-delta", type=float, default=0.05)
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
    parser.add_argument("--train-alpha", type=float, default=1.0)
    parser.add_argument("--eval-alphas", default="0,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.5,0.75,1.0")
    parser.add_argument("--bound-abs", default="0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03,0.04")
    parser.add_argument("--bound-rel", default="0,0.0025,0.005,0.0075,0.01,0.0125,0.015,0.02,0.025,0.03,0.04,0.05,0.08,0.1")
    parser.add_argument("--point", type=float, default=1.0)
    parser.add_argument("--mse", type=float, default=0.05)
    parser.add_argument("--tke", type=float, default=0.06)
    parser.add_argument("--temporal", type=float, default=0.03)
    parser.add_argument("--grad", type=float, default=0.015)
    parser.add_argument("--p-zero", type=float, default=0.01)
    parser.add_argument("--residual-mse", type=float, default=0.25)
    parser.add_argument("--delta-penalty", type=float, default=0.02)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"out_dir already exists, refusing to overwrite: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    paths = list_h5(args.real_root, BAD_TRAIN_FILES)
    train_paths, val_paths = split_paths(paths, args.val_fraction, args.seed)
    train_dataset = H5WindowDataset(
        train_paths,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        stride=args.stride,
        sub_sample=args.sub_sample,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        include_pressure=args.include_pressure_data,
    )
    val_dataset = H5WindowDataset(
        val_paths,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        stride=args.stride,
        sub_sample=args.sub_sample,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        include_pressure=args.include_pressure_data,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    base_model = load_frozen_cno(args.checkpoint, args.realpdebench_root, device)
    corrector_config = CorrectorConfig(
        hidden=args.hidden,
        blocks=args.blocks,
        dropout=args.dropout,
        include_pressure=not args.drop_pressure_feature,
        max_delta=args.max_delta,
    )
    model = ResidualCorrectionModel(base_model, ResidualCorrector3D(corrector_config)).to(device)
    optimizer = torch.optim.AdamW(model.corrector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.updates))

    weights = {
        "point": args.point,
        "mse": args.mse,
        "tke": args.tke,
        "temporal": args.temporal,
        "grad": args.grad,
        "p_zero": args.p_zero,
    }
    alphas = parse_float_list(args.eval_alphas)
    abs_widths = parse_float_list(args.bound_abs)
    rel_widths = parse_float_list(args.bound_rel)

    run_config = {
        "real_root": str(args.real_root),
        "checkpoint": str(args.checkpoint),
        "realpdebench_root": str(args.realpdebench_root),
        "device": str(device),
        "train_trajectories": len(train_paths),
        "val_trajectories": len(val_paths),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "updates": args.updates,
        "eval_interval": args.eval_interval,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "corrector_config": asdict(corrector_config),
        "train_alpha": args.train_alpha,
        "eval_alphas": alphas,
        "abs_widths": abs_widths,
        "rel_widths": rel_widths,
        "loss_weights": weights,
        "residual_mse": args.residual_mse,
        "delta_penalty": args.delta_penalty,
        "trainable_parameters": sum(p.numel() for p in model.corrector.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    print(json.dumps(run_config, indent=2, default=str), flush=True)

    train_log: list[dict[str, object]] = []
    eval_log: list[dict[str, object]] = []

    summaries = evaluate_alphas(
        model,
        val_loader,
        device,
        alphas=alphas,
        abs_widths=abs_widths,
        rel_widths=rel_widths,
        max_batches=args.max_eval_batches,
    )
    top = summaries[0]
    top["iteration"] = 0
    eval_log.append({"iteration": 0, "summaries": summaries[:5]})
    best_score = float(top["best_final_est"])
    best_iter = 0
    best_alpha = float(top["alpha"])
    best_bound_abs = float(top["best_bound_abs"])
    best_bound_rel = float(top["best_bound_rel"])
    print("EVAL_TOP " + json.dumps(top, sort_keys=True), flush=True)
    save_checkpoint(
        args.out_dir / "model_best.pth",
        model,
        iteration=0,
        best_score=best_score,
        best_alpha=best_alpha,
        best_bound_abs=best_bound_abs,
        best_bound_rel=best_bound_rel,
        train_log=train_log,
        eval_log=eval_log,
        run_config=run_config,
    )

    train_iter = iter(train_loader)
    accum: dict[str, list[float]] = {}
    for step in range(1, args.updates + 1):
        model.corrector.train()
        model.base_model.eval()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            base = model.base_predict(x)
        delta = model.predict_delta(x, base)
        pred = model.combine(base, delta, args.train_alpha)
        loss, parts = physics_loss(pred, y, weights)
        residual_target = y[..., :2] - base[..., :2]
        residual_mse = torch.mean((delta[..., :2] - residual_target) ** 2)
        delta_penalty = torch.mean(delta[..., :2] ** 2)
        loss = loss + args.residual_mse * residual_mse + args.delta_penalty * delta_penalty
        parts["residual_mse"] = float(residual_mse.detach().cpu())
        parts["delta_penalty"] = float(delta_penalty.detach().cpu())
        parts["loss"] = float(loss.detach().cpu())
        loss.backward()
        if args.clip_grad and args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.corrector.parameters(), args.clip_grad)
        optimizer.step()
        scheduler.step()

        for key, value in parts.items():
            accum.setdefault(key, []).append(float(value))
        if step % 20 == 0:
            row = {
                "iteration": step,
                "lr": optimizer.param_groups[0]["lr"],
                **{key: float(np.mean(values[-20:])) for key, values in accum.items()},
            }
            train_log.append(row)
            print("TRAIN " + json.dumps(row, sort_keys=True), flush=True)

        if step % args.eval_interval == 0 or step == args.updates:
            summaries = evaluate_alphas(
                model,
                val_loader,
                device,
                alphas=alphas,
                abs_widths=abs_widths,
                rel_widths=rel_widths,
                max_batches=args.max_eval_batches,
            )
            top = summaries[0]
            top["iteration"] = step
            eval_log.append({"iteration": step, "summaries": summaries[:5]})
            current_score = float(top["best_final_est"])
            print("EVAL_TOP " + json.dumps(top, sort_keys=True), flush=True)
            save_checkpoint(
                args.out_dir / "model_latest.pth",
                model,
                iteration=step,
                best_score=current_score,
                best_alpha=float(top["alpha"]),
                best_bound_abs=float(top["best_bound_abs"]),
                best_bound_rel=float(top["best_bound_rel"]),
                train_log=train_log,
                eval_log=eval_log,
                run_config=run_config,
            )
            if current_score > best_score:
                best_score = current_score
                best_iter = step
                best_alpha = float(top["alpha"])
                best_bound_abs = float(top["best_bound_abs"])
                best_bound_rel = float(top["best_bound_rel"])
                save_checkpoint(
                    args.out_dir / "model_best.pth",
                    model,
                    iteration=step,
                    best_score=best_score,
                    best_alpha=best_alpha,
                    best_bound_abs=best_bound_abs,
                    best_bound_rel=best_bound_rel,
                    train_log=train_log,
                    eval_log=eval_log,
                    run_config=run_config,
                )
                print(
                    f"BEST iteration={best_iter} final_est={best_score:.6f} "
                    f"alpha={best_alpha} abs={best_bound_abs} rel={best_bound_rel}",
                    flush=True,
                )
            (args.out_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "best_iter": best_iter,
                        "best_score": best_score,
                        "best_alpha": best_alpha,
                        "best_bound_abs": best_bound_abs,
                        "best_bound_rel": best_bound_rel,
                        "latest_top": top,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

    save_checkpoint(
        args.out_dir / "model_final.pth",
        model,
        iteration=args.updates,
        best_score=best_score,
        best_alpha=best_alpha,
        best_bound_abs=best_bound_abs,
        best_bound_rel=best_bound_rel,
        train_log=train_log,
        eval_log=eval_log,
        run_config=run_config,
    )
    print(
        f"DONE out_dir={args.out_dir} best_iter={best_iter} "
        f"best_score={best_score:.6f} alpha={best_alpha} abs={best_bound_abs} rel={best_bound_rel}",
        flush=True,
    )


if __name__ == "__main__":
    main()
