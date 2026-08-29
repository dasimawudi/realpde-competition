#!/usr/bin/env python3
"""Fine-tune CNO for RealPDE Track 1 with physics-aware losses.

The script keeps the competition-compatible raw input/output convention and
uses the released validation split to select checkpoints by estimated
leaderboard score with calibrated SPS bounds.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from realpdebench.data.fluid_dataset import Foil
from realpdebench.model.load_model import load_model
from realpdebench.utils.utils import add_args_from_config, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
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


BAD_TRAIN_FILES = {"7575_0.h5"}


class FilteredDataset(Dataset):
    def __init__(self, base: Foil, bad_files: set[str] | None = None):
        self.base = base
        bad_files = bad_files or set()
        self.indices = [
            i
            for i, sid in enumerate(base.sim_id_mapping[base.mode])
            if sid not in bad_files
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

    def __getattr__(self, name):
        return getattr(self.base, name)


def make_args(config: str, dataset_root: str, checkpoint_path: str, results_path: str):
    old_argv = sys.argv[:]
    try:
        sys.argv = ["tke_finetune", "--config", config, "--train_data_type", "real", "--is_finetune"]
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


def kinetic_energy_torch(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 0]
    v = x[..., 1]
    u_prime = ((u - u.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
    v_prime = ((v - v.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
    return 0.5 * (u_prime + v_prime)


def rel_l2_loss_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    b = pred.shape[0]
    p = pred.reshape(b, -1)
    t = target.reshape(b, -1)
    return (torch.linalg.norm(p - t, dim=1) / torch.linalg.norm(t, dim=1).clamp_min(eps)).mean()


def temporal_rel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return rel_l2_loss_torch(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])


def spatial_grad_rel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    targ_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    targ_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return 0.5 * rel_l2_loss_torch(pred_dx, targ_dx) + 0.5 * rel_l2_loss_torch(pred_dy, targ_dy)


def physics_loss(pred: torch.Tensor, target: torch.Tensor, weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    pred_uv = pred[..., :2]
    target_uv = target[..., :2]
    point = rel_l2_loss_torch(pred_uv, target_uv)
    mse = torch.mean((pred_uv - target_uv) ** 2)
    pred_ke = kinetic_energy_torch(pred_uv)
    target_ke = kinetic_energy_torch(target_uv)
    tke = rel_l2_loss_torch(pred_ke, target_ke)
    temp = temporal_rel_loss(pred_uv, target_uv)
    grad = spatial_grad_rel_loss(pred_uv, target_uv)
    p_zero = torch.mean(pred[..., 2] ** 2) if pred.shape[-1] > 2 else pred.new_tensor(0.0)
    loss = (
        weights["point"] * point
        + weights["mse"] * mse
        + weights["tke"] * tke
        + weights["temporal"] * temp
        + weights["grad"] * grad
        + weights["p_zero"] * p_zero
    )
    parts = {
        "loss": float(loss.detach().cpu()),
        "point_rel": float(point.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "tke_rel": float(tke.detach().cpu()),
        "temporal_rel": float(temp.detach().cpu()),
        "grad_rel": float(grad.detach().cpu()),
        "p_zero": float(p_zero.detach().cpu()),
    }
    return loss, parts


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


def save_checkpoint(path: Path, model, iteration: int, best_score: float, train_log: list[dict], eval_log: list[dict]):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "train_losses": train_log,
        "val_losses": eval_log,
        "iteration": iteration,
        "best_iteration": iteration,
        "best_val_loss": -best_score,
    }
    torch.save(ckpt, path)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", default="/root/autodl-tmp/realpde/RealPDEBench/realpdebench/configs/foil/cno.yaml")
    cli.add_argument("--dataset-root", default="/root/autodl-tmp/realpde_competition_h5")
    cli.add_argument("--checkpoint", required=True)
    cli.add_argument("--out-root", default="/root/autodl-fs/realpde_runs")
    cli.add_argument("--run-name", default=None)
    cli.add_argument("--num-update", type=int, default=1800)
    cli.add_argument("--eval-interval", type=int, default=100)
    cli.add_argument("--batch-size", type=int, default=16)
    cli.add_argument("--test-batch-size", type=int, default=64)
    cli.add_argument("--lr", type=float, default=5e-5)
    cli.add_argument("--seed", type=int, default=1)
    cli.add_argument("--max-eval-batches", type=int, default=None)
    cli.add_argument("--point", type=float, default=1.0)
    cli.add_argument("--mse", type=float, default=0.05)
    cli.add_argument("--tke", type=float, default=0.10)
    cli.add_argument("--temporal", type=float, default=0.05)
    cli.add_argument("--grad", type=float, default=0.03)
    cli.add_argument("--p-zero", type=float, default=0.01)
    args_cli = cli.parse_args()

    set_seed(args_cli.seed)
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_name = args_cli.run_name or f"cno_tke_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    cfg_args.train_batch_size = args_cli.batch_size
    cfg_args.test_batch_size = args_cli.test_batch_size
    cfg_args.lr = args_cli.lr

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

    model = load_model(train_base, device=device, **vars(cfg_args))
    meta = model.load_checkpoint(args_cli.checkpoint, device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args_cli.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args_cli.num_update)
    weights = {
        "point": args_cli.point,
        "mse": args_cli.mse,
        "tke": args_cli.tke,
        "temporal": args_cli.temporal,
        "grad": args_cli.grad,
        "p_zero": args_cli.p_zero,
    }
    abs_widths = np.array(
        [0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030, 0.040],
        dtype=np.float32,
    )
    rel_widths = np.array([0.0, 0.02, 0.05, 0.08, 0.10], dtype=np.float32)

    print(
        json.dumps(
            {
                "device": str(device),
                "out_dir": str(out_dir),
                "checkpoint": args_cli.checkpoint,
                "checkpoint_meta": {
                    "iteration": meta.get("iteration"),
                    "best_iteration": meta.get("best_iteration"),
                    "best_val_loss": str(meta.get("best_val_loss")),
                },
                "train_len": len(train_dataset),
                "train_len_before_filter": len(train_base),
                "val_len": len(val_dataset),
                "weights": weights,
            },
            default=str,
            indent=2,
        ),
        flush=True,
    )

    train_log: list[dict] = []
    eval_log: list[dict] = []
    best_score = -float("inf")
    best_iter = 0
    best_summary = None

    # Baseline eval before any update.
    summary = evaluate(model, val_loader, device, abs_widths, rel_widths, args_cli.max_eval_batches)
    summary["iteration"] = 0
    eval_log.append(summary)
    print("EVAL " + json.dumps(summary, sort_keys=True), flush=True)
    best_score = summary["best_bounds"][0]["final_est"]
    best_iter = 0
    best_summary = summary
    save_checkpoint(out_dir / "model_best.pth", model, 0, best_score, train_log, eval_log)

    loader_iter = iter(train_loader)
    smooth = {}
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in parts.items():
            smooth[k] = 0.98 * smooth.get(k, v) + 0.02 * v
        if it % 20 == 0:
            row = {"iteration": it, "lr": sched.get_last_lr()[0], **smooth}
            train_log.append(row)
            print("TRAIN " + json.dumps(row, sort_keys=True), flush=True)
        if it % args_cli.eval_interval == 0 or it == args_cli.num_update:
            summary = evaluate(model, val_loader, device, abs_widths, rel_widths, args_cli.max_eval_batches)
            summary["iteration"] = it
            eval_log.append(summary)
            current_score = summary["best_bounds"][0]["final_est"]
            print("EVAL " + json.dumps(summary, sort_keys=True), flush=True)
            save_checkpoint(out_dir / "model_latest.pth", model, it, current_score, train_log, eval_log)
            if current_score > best_score:
                best_score = current_score
                best_iter = it
                best_summary = summary
                save_checkpoint(out_dir / "model_best.pth", model, it, best_score, train_log, eval_log)
                print(f"BEST iteration={best_iter} final_est={best_score:.6f}", flush=True)
            (out_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "best_score": best_score,
                        "best_iter": best_iter,
                        "best_summary": best_summary,
                        "latest_summary": summary,
                        "weights": weights,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    save_checkpoint(out_dir / "model_final.pth", model, args_cli.num_update, best_score, train_log, eval_log)
    print(f"DONE out_dir={out_dir} best_iter={best_iter} best_score={best_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
