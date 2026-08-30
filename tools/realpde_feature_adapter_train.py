#!/usr/bin/env python3
"""Train a small feature adapter in front of a RealPDE Track 1 CNO checkpoint.

This is the lowest-risk feature-engineering experiment for the current CNO
line.  It does not change the competition I/O format.  The adapter receives
derived features such as speed, temporal deltas, vorticity, divergence, and
coordinates, projects them back to three channels, and then feeds the original
CNO model.

By default the CNO backbone is frozen and the final adapter layer is initialized
to zero, so iteration 0 is effectively the original checkpoint.  Use
``--train-base`` only after the adapter-only run has shown validation gain.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from realpdebench.data.fluid_dataset import Foil
from realpdebench.model.load_model import load_model
from realpdebench.utils.utils import add_args_from_config, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from realpde_calibrate_bounds import (  # noqa: E402
    finalize_scores,
    init_sps_candidates,
    measured_channels,
    mvpe_rel_l2_per_sample,
    rel_l2_per_sample,
    tke_rel_l2_per_sample,
    update_metric_sums,
    update_sps_candidates,
)
from realpde_feature_engineering import augment_torch, feature_names  # noqa: E402
from realpde_tke_finetune import BAD_TRAIN_FILES, FilteredDataset, physics_loss  # noqa: E402


class PointwiseFeatureAdapter(nn.Module):
    """Residual per-point projection from engineered features to u/v/p input."""

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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Keep the adapter exactly residual-zero at initialization.  The first
        # forward pass is therefore the original CNO input, which makes the
        # validation baseline directly comparable to previous submissions.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = augment_torch(x, include_pressure=self.include_pressure)
        delta = self.net(features)
        if x.shape[-1] >= 3:
            raw = x[..., :3]
        else:
            raw = torch.cat([x[..., :2], torch.zeros_like(x[..., :1])], dim=-1)
        return raw + delta


class FeatureAdapterModel(nn.Module):
    """A RealPDE backbone preceded by ``PointwiseFeatureAdapter``."""

    def __init__(self, base_model: nn.Module, adapter: PointwiseFeatureAdapter) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(self.adapter(x))


def make_args(config: str, dataset_root: str, checkpoint_path: str, results_path: str):
    old_argv = sys.argv[:]
    try:
        sys.argv = ["feature_adapter", "--config", config, "--train_data_type", "real", "--is_finetune"]
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default=config)
        parser.add_argument("--gpu", type=int, default=0)
        parser.add_argument("--train_data_type", type=str, default="real")
        parser.add_argument("--is_finetune", action="store_true")
        args = parser.parse_args()
        args = add_args_from_config(args)
        args.dataset_root = dataset_root
        args.checkpoint_path = checkpoint_path
        args.results_path = results_path
        args.normalizer = "none"
        args.is_use_tb = False
        args.num_workers = min(int(getattr(args, "num_workers", 8)), 8)
        return args
    finally:
        sys.argv = old_argv


@torch.no_grad()
def evaluate(model, loader, device, abs_widths, rel_widths, max_batches=None):
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
        start = time.perf_counter()
        pred = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        target_np = y.detach().cpu().numpy().astype(np.float32)
        pred_np[..., 2] = 0.0
        c = measured_channels(target_np)
        rel = rel_l2_per_sample(pred_np, target_np, c)
        tke = tke_rel_l2_per_sample(pred_np, target_np, c)
        mvpe = mvpe_rel_l2_per_sample(pred_np, target_np)
        update_metric_sums(metric_sums, pred_np, target_np, elapsed)
        update_sps_candidates(candidates, pred_np, target_np, c, rel, tke, mvpe)
    return finalize_scores(metric_sums, candidates)


def save_feature_checkpoint(
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


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", default="/root/autodl-tmp/realpde/RealPDEBench/realpdebench/configs/foil/cno.yaml")
    cli.add_argument("--dataset-root", default="/root/autodl-tmp/realpde_competition_h5")
    cli.add_argument("--checkpoint", required=True, help="Base CNO checkpoint to adapt.")
    cli.add_argument("--out-root", default="/root/autodl-fs/realpde_runs")
    cli.add_argument("--run-name", default=None)
    cli.add_argument("--num-update", type=int, default=1200)
    cli.add_argument("--eval-interval", type=int, default=100)
    cli.add_argument("--batch-size", type=int, default=12)
    cli.add_argument("--test-batch-size", type=int, default=64)
    cli.add_argument("--adapter-lr", type=float, default=3e-4)
    cli.add_argument("--base-lr", type=float, default=1e-7)
    cli.add_argument("--weight-decay", type=float, default=1e-6)
    cli.add_argument("--hidden", type=int, default=32)
    cli.add_argument("--dropout", type=float, default=0.0)
    cli.add_argument("--train-base", action="store_true", help="Also fine-tune the CNO backbone.")
    cli.add_argument("--drop-pressure-feature", action="store_true")
    cli.add_argument("--seed", type=int, default=41)
    cli.add_argument("--max-eval-batches", type=int, default=None)
    cli.add_argument("--point", type=float, default=1.0)
    cli.add_argument("--mse", type=float, default=0.05)
    cli.add_argument("--tke", type=float, default=0.08)
    cli.add_argument("--temporal", type=float, default=0.04)
    cli.add_argument("--grad", type=float, default=0.02)
    cli.add_argument("--p-zero", type=float, default=0.01)
    cli.add_argument("--clip-grad", type=float, default=1.0)
    args_cli = cli.parse_args()

    set_seed(args_cli.seed)
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_name = args_cli.run_name or f"cno_feature_adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args_cli.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_args = make_args(args_cli.config, args_cli.dataset_root, args_cli.checkpoint, str(out_dir))
    train_base = Foil(
        dataset_name=cfg_args.dataset_name,
        dataset_root=cfg_args.dataset_root,
        mode="train",
        dataset_type="real",
        mask_prob=cfg_args.mask_prob,
        noise_scale=0.0,
    )
    val_base = Foil(
        dataset_name=cfg_args.dataset_name,
        dataset_root=cfg_args.dataset_root,
        mode="val",
        dataset_type="real",
    )
    train_dataset = FilteredDataset(train_base, BAD_TRAIN_FILES)
    val_dataset = val_base

    train_loader = DataLoader(
        train_dataset,
        batch_size=args_cli.batch_size,
        shuffle=True,
        num_workers=cfg_args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args_cli.test_batch_size,
        shuffle=False,
        num_workers=cfg_args.num_workers,
        pin_memory=True,
    )

    base_model = load_model(train_base, device=device, **vars(cfg_args))
    meta = base_model.load_checkpoint(args_cli.checkpoint, device) or {}
    adapter = PointwiseFeatureAdapter(
        hidden=args_cli.hidden,
        include_pressure=not args_cli.drop_pressure_feature,
        dropout=args_cli.dropout,
    ).to(device)
    model = FeatureAdapterModel(base_model, adapter).to(device)

    groups = trainable_parameter_groups(
        model,
        train_base=args_cli.train_base,
        adapter_lr=args_cli.adapter_lr,
        base_lr=args_cli.base_lr,
        weight_decay=args_cli.weight_decay,
    )
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args_cli.num_update)
    weights = {
        "point": args_cli.point,
        "mse": args_cli.mse,
        "tke": args_cli.tke,
        "temporal": args_cli.temporal,
        "grad": args_cli.grad,
        "p_zero": args_cli.p_zero,
    }
    abs_widths = np.array([0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030, 0.040], dtype=np.float32)
    rel_widths = np.array([0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20], dtype=np.float32)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    run_config = {
        "device": str(device),
        "out_dir": str(out_dir),
        "config": args_cli.config,
        "checkpoint": args_cli.checkpoint,
        "checkpoint_meta": {
            "iteration": meta.get("iteration"),
            "best_iteration": meta.get("best_iteration"),
            "best_val_loss": str(meta.get("best_val_loss")),
        },
        "adapter_config": adapter.config(),
        "train_base": args_cli.train_base,
        "train_len": len(train_dataset),
        "train_len_before_filter": len(train_base),
        "val_len": len(val_dataset),
        "batch_size": args_cli.batch_size,
        "test_batch_size": args_cli.test_batch_size,
        "adapter_lr": args_cli.adapter_lr,
        "base_lr": args_cli.base_lr,
        "weight_decay": args_cli.weight_decay,
        "num_update": args_cli.num_update,
        "eval_interval": args_cli.eval_interval,
        "max_eval_batches": args_cli.max_eval_batches,
        "weights": weights,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    print(json.dumps(run_config, default=str, indent=2), flush=True)

    train_log: list[dict] = []
    eval_log: list[dict] = []

    summary = evaluate(model, val_loader, device, abs_widths, rel_widths, args_cli.max_eval_batches)
    summary["iteration"] = 0
    eval_log.append(summary)
    best_score = summary["best_bounds"][0]["final_est"]
    best_iter = 0
    best_summary = summary
    print("EVAL " + json.dumps(summary, sort_keys=True), flush=True)
    save_feature_checkpoint(
        out_dir / "model_best.pth",
        model,
        iteration=0,
        best_score=best_score,
        train_log=train_log,
        eval_log=eval_log,
        run_config=run_config,
    )

    loader_iter = iter(train_loader)
    smooth: dict[str, float] = {}
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    model.train()
    for it in range(1, args_cli.num_update + 1):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss, parts = physics_loss(pred, y, weights)
        loss.backward()
        if args_cli.clip_grad and args_cli.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args_cli.clip_grad)
        opt.step()
        sched.step()

        for key, value in parts.items():
            smooth[key] = 0.98 * smooth.get(key, value) + 0.02 * value
        if it % 20 == 0:
            row = {
                "iteration": it,
                "adapter_lr": opt.param_groups[0]["lr"],
                "base_lr": opt.param_groups[1]["lr"] if len(opt.param_groups) > 1 else 0.0,
                **smooth,
            }
            train_log.append(row)
            print("TRAIN " + json.dumps(row, sort_keys=True), flush=True)
        if it % args_cli.eval_interval == 0 or it == args_cli.num_update:
            summary = evaluate(model, val_loader, device, abs_widths, rel_widths, args_cli.max_eval_batches)
            summary["iteration"] = it
            eval_log.append(summary)
            current_score = summary["best_bounds"][0]["final_est"]
            print("EVAL " + json.dumps(summary, sort_keys=True), flush=True)
            save_feature_checkpoint(
                out_dir / "model_latest.pth",
                model,
                iteration=it,
                best_score=current_score,
                train_log=train_log,
                eval_log=eval_log,
                run_config=run_config,
            )
            if current_score > best_score:
                best_score = current_score
                best_iter = it
                best_summary = summary
                save_feature_checkpoint(
                    out_dir / "model_best.pth",
                    model,
                    iteration=it,
                    best_score=best_score,
                    train_log=train_log,
                    eval_log=eval_log,
                    run_config=run_config,
                )
                print(f"BEST iteration={best_iter} final_est={best_score:.6f}", flush=True)
            (out_dir / "summary.json").write_text(
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

    save_feature_checkpoint(
        out_dir / "model_final.pth",
        model,
        iteration=args_cli.num_update,
        best_score=best_score,
        train_log=train_log,
        eval_log=eval_log,
        run_config=run_config,
    )
    print(f"DONE out_dir={out_dir} best_iter={best_iter} best_score={best_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
