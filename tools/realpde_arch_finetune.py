#!/usr/bin/env python3
"""Architecture-generic fine-tuning for RealPDE Track 1.

This extends the CNO-only tuning workflow so we can fairly test non-CNO
checkpoints with model-specific config overrides and optional normalization.
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
from torch.utils.data import DataLoader

from realpdebench.data.fluid_dataset import Foil
from realpdebench.model.load_model import load_model
from realpdebench.utils.utils import add_args_from_config, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from realpde_calibrate_bounds import (  # noqa: E402
    build_normalizer,
    finalize_scores,
    init_sps_candidates,
    measured_channels,
    mvpe_rel_l2_per_sample,
    rel_l2_per_sample,
    tke_rel_l2_per_sample,
    update_metric_sums,
    update_sps_candidates,
)
from realpde_tke_finetune import (  # noqa: E402
    BAD_TRAIN_FILES,
    FilteredDataset,
    physics_loss,
    save_checkpoint,
)


def parse_value(text: str):
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def make_args(config: str, dataset_root: str, checkpoint_path: str, results_path: str):
    old_argv = sys.argv[:]
    try:
        sys.argv = ["arch_finetune", "--config", config, "--train_data_type", "real", "--is_finetune"]
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
        args.is_use_tb = False
        args.num_workers = min(int(getattr(args, "num_workers", 8)), 8)
        return args
    finally:
        sys.argv = old_argv


def apply_overrides(args, overrides: list[str]) -> dict[str, object]:
    applied: dict[str, object] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        parsed = parse_value(value.strip())
        setattr(args, key, parsed)
        applied[key] = parsed
    return applied


def forward_for_loss(model, normalizer, x, y, use_normalized: bool):
    if use_normalized:
        x_model, y_model = normalizer.preprocess(x, y)
        pred_model = model(x_model)
        _, pred_raw = normalizer.postprocess(x_model, pred_model)
        _, target_raw = normalizer.postprocess(x_model, y_model)
        return pred_raw, target_raw
    pred_raw = model(x)
    return pred_raw, y


@torch.no_grad()
def evaluate_arch(model, loader, device, normalizer, use_normalized: bool, abs_widths, rel_widths, max_batches=None):
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
        if use_normalized:
            x_model, y_model = normalizer.preprocess(x, y)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            pred_model = model(x_model)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            _, pred = normalizer.postprocess(x_model, pred_model)
            _, target = normalizer.postprocess(x_model, y_model)
        else:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            pred = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            target = y

        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        target_np = target.detach().cpu().numpy().astype(np.float32)
        pred_np[..., 2] = 0.0
        c = measured_channels(target_np)
        rel = rel_l2_per_sample(pred_np, target_np, c)
        tke = tke_rel_l2_per_sample(pred_np, target_np, c)
        mvpe = mvpe_rel_l2_per_sample(pred_np, target_np)
        update_metric_sums(metric_sums, pred_np, target_np, elapsed)
        update_sps_candidates(candidates, pred_np, target_np, c, rel, tke, mvpe)
    return finalize_scores(metric_sums, candidates)


def save_normalizer(out_dir: Path, normalizer) -> None:
    payload = {}
    for name in ["mean_inputs", "mean_targets", "std_inputs", "std_targets", "max_inputs", "max_targets"]:
        if hasattr(normalizer, name):
            payload[name] = getattr(normalizer, name).detach().cpu()
    if payload:
        torch.save(payload, out_dir / "normalizer.pt")


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", required=True)
    cli.add_argument("--dataset-root", default="/root/autodl-tmp/realpde_competition_h5")
    cli.add_argument("--checkpoint", required=True)
    cli.add_argument("--out-root", default="/root/autodl-fs/realpde_runs")
    cli.add_argument("--run-name", default=None)
    cli.add_argument("--normalizer", choices=["config", "none", "gaussian", "range"], default="config")
    cli.add_argument("--override", action="append", default=[])
    cli.add_argument("--num-update", type=int, default=800)
    cli.add_argument("--eval-interval", type=int, default=200)
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument("--test-batch-size", type=int, default=8)
    cli.add_argument("--lr", type=float, default=1e-5)
    cli.add_argument("--seed", type=int, default=3)
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
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg_args = make_args(args_cli.config, args_cli.dataset_root, args_cli.checkpoint, args_cli.out_root)
    overrides = apply_overrides(cfg_args, args_cli.override)
    if args_cli.normalizer != "config":
        cfg_args.normalizer = args_cli.normalizer
    cfg_args.train_batch_size = args_cli.batch_size
    cfg_args.test_batch_size = args_cli.test_batch_size
    cfg_args.lr = args_cli.lr

    run_name = args_cli.run_name or (
        f"{getattr(cfg_args, 'model_name', 'model')}_{getattr(cfg_args, 'normalizer', 'na')}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args_cli.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

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

    normalizer_name = getattr(cfg_args, "normalizer", "none")
    normalizer = build_normalizer(normalizer_name, train_base, device)
    save_normalizer(out_dir, normalizer)
    use_normalized = normalizer_name != "none"

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
    meta = model.load_checkpoint(args_cli.checkpoint, device) or {}
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
    abs_widths = np.array([0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030, 0.040], dtype=np.float32)
    rel_widths = np.array([0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20], dtype=np.float32)

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
        "model_name": getattr(cfg_args, "model_name", None),
        "normalizer": normalizer_name,
        "overrides": overrides,
        "train_len": len(train_dataset),
        "train_len_before_filter": len(train_base),
        "val_len": len(val_dataset),
        "batch_size": args_cli.batch_size,
        "test_batch_size": args_cli.test_batch_size,
        "lr": args_cli.lr,
        "num_update": args_cli.num_update,
        "eval_interval": args_cli.eval_interval,
        "max_eval_batches": args_cli.max_eval_batches,
        "weights": weights,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    print(json.dumps(run_config, default=str, indent=2), flush=True)

    train_log: list[dict] = []
    eval_log: list[dict] = []

    summary = evaluate_arch(
        model, val_loader, device, normalizer, use_normalized, abs_widths, rel_widths, args_cli.max_eval_batches
    )
    summary["iteration"] = 0
    eval_log.append(summary)
    best_score = summary["best_bounds"][0]["final_est"]
    best_iter = 0
    best_summary = summary
    print("EVAL " + json.dumps(summary, sort_keys=True), flush=True)
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
        pred, target = forward_for_loss(model, normalizer, x, y, use_normalized)
        loss, parts = physics_loss(pred, target, weights)
        loss.backward()
        if args_cli.clip_grad and args_cli.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args_cli.clip_grad)
        opt.step()
        sched.step()

        for key, value in parts.items():
            smooth[key] = 0.98 * smooth.get(key, value) + 0.02 * value
        if it % 20 == 0:
            row = {"iteration": it, "lr": sched.get_last_lr()[0], **smooth}
            train_log.append(row)
            print("TRAIN " + json.dumps(row, sort_keys=True), flush=True)
        if it % args_cli.eval_interval == 0 or it == args_cli.num_update:
            summary = evaluate_arch(
                model, val_loader, device, normalizer, use_normalized, abs_widths, rel_widths, args_cli.max_eval_batches
            )
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
                        "run_config": run_config,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    save_checkpoint(out_dir / "model_final.pth", model, args_cli.num_update, best_score, train_log, eval_log)
    print(f"DONE out_dir={out_dir} best_iter={best_iter} best_score={best_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
