#!/usr/bin/env python3
"""Evaluate a RealPDE Track 1 checkpoint and calibrate SPS intervals on val data.

This script is intentionally self-contained so it can be copied to the remote
training box and run inside the RealPDEBench checkout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from realpdebench.data.data_normalizer import GaussianNormalizer, IdentityNormalizer, RangeNormalizer
from realpdebench.data.fluid_dataset import Foil
from realpdebench.model.load_model import load_model
from realpdebench.utils.utils import add_args_from_config, set_seed


SIGMA_GLOBAL = 0.0563870259
T_NUMERICAL_SEC = 0.72896


def measured_channels(target: np.ndarray) -> int:
    active = [not np.allclose(target[..., idx], 0.0) for idx in range(target.shape[-1])]
    return max(1, int(sum(active)))


def rel_l2_per_sample(pred: np.ndarray, target: np.ndarray, c: int) -> np.ndarray:
    p = pred[..., :c].reshape(pred.shape[0], -1)
    t = target[..., :c].reshape(target.shape[0], -1)
    denom = np.linalg.norm(t, axis=1).clip(min=1e-8)
    return np.linalg.norm(p - t, axis=1) / denom


def kinetic_energy(x: np.ndarray) -> np.ndarray:
    u = x[..., 0]
    v = x[..., 1]
    u_prime = np.mean((u - np.mean(u, axis=1, keepdims=True)) ** 2, axis=1)
    v_prime = np.mean((v - np.mean(v, axis=1, keepdims=True)) ** 2, axis=1)
    return 0.5 * (u_prime + v_prime)


def tke_rel_l2_per_sample(pred: np.ndarray, target: np.ndarray, c: int) -> np.ndarray:
    if c < 2:
        return np.zeros((pred.shape[0],), dtype=np.float32)
    pred_ke = kinetic_energy(pred[..., :c])
    target_ke = kinetic_energy(target[..., :c])
    p = pred_ke.reshape(pred.shape[0], -1)
    t = target_ke.reshape(target.shape[0], -1)
    denom = np.linalg.norm(t, axis=1).clip(min=1e-8)
    return np.linalg.norm(p - t, axis=1) / denom


def mvpe_rel_l2_per_sample(pred: np.ndarray, target: np.ndarray, sub_s_real: int = 2) -> np.ndarray:
    d = 16
    center_x = 10
    center_y = 32
    n_probe = 9
    n, _, h, w, _ = pred.shape
    probe_center_y = int(center_y / sub_s_real)
    interval_y = min(2, int(h / (n_probe + 1)))
    probe_y = [
        probe_center_y + interval_y * j
        for j in range(-(n_probe - 1) // 2, n_probe - (n_probe - 1) // 2)
    ]
    probe_y = [y for y in probe_y if 0 <= y < h]
    if not probe_y:
        return np.zeros((n,), dtype=np.float32)
    errors = []
    for i in range(4):
        if int((2 * d + center_x) / sub_s_real) < w:
            probe_x = int(((i + 1) * d + center_x) / sub_s_real)
        else:
            probe_x = int((0.5 * (i + 2) * d + center_x) / sub_s_real)
        if not 0 <= probe_x < w:
            continue
        pp = pred[:, :, probe_y, probe_x, :2].mean(axis=1).reshape(n, -1)
        tt = target[:, :, probe_y, probe_x, :2].mean(axis=1).reshape(n, -1)
        denom = np.linalg.norm(tt, axis=1).clip(min=1e-8)
        errors.append(np.linalg.norm(pp - tt, axis=1) / denom)
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
    r = float(t_neural) / T_NUMERICAL_SEC
    return float(100.0 / (1.0 + (r / r_min) ** 0.5))


def make_args(config: str, dataset_root: str, checkpoint_path: str, results_path: str):
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "calibrate",
            "--config",
            config,
            "--train_data_type",
            "real",
            "--is_finetune",
        ]
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
        args.num_workers = min(int(getattr(args, "num_workers", 8)), 8)
        args.test_batch_size = min(int(getattr(args, "test_batch_size", 64)), 64)
        return args
    finally:
        sys.argv = old_argv


def build_normalizer(name: str, normalizer_dataset, device):
    if name == "none":
        return IdentityNormalizer(device=device)
    if name == "gaussian":
        return GaussianNormalizer(normalizer_dataset, device=device)
    if name == "range":
        return RangeNormalizer(normalizer_dataset, device=device)
    raise ValueError(f"unsupported normalizer {name}")


def update_metric_sums(sums: dict, pred: np.ndarray, target: np.ndarray, elapsed: float):
    c = measured_channels(target)
    rel = rel_l2_per_sample(pred, target, c)
    tke = tke_rel_l2_per_sample(pred, target, c)
    mvpe = mvpe_rel_l2_per_sample(pred, target)
    n = pred.shape[0]
    sums["n"] += n
    sums["rel_l2_sum"] += float(np.sum(rel))
    sums["tke_sum"] += float(np.sum(tke))
    sums["mvpe_sum"] += float(np.sum(mvpe))
    sums["time_sum"] += float(elapsed)
    sums["time_n"] += n
    sums.setdefault("rel_all", []).append(rel.astype(np.float32))
    sums.setdefault("tke_all", []).append(tke.astype(np.float32))
    sums.setdefault("mvpe_all", []).append(mvpe.astype(np.float32))
    return c, rel, tke, mvpe


def init_sps_candidates(abs_widths, rel_widths):
    candidates = []
    for abs_w in abs_widths:
        for rel_w in rel_widths:
            candidates.append(
                {
                    "abs": float(abs_w),
                    "rel": float(rel_w),
                    "sum_dm": 0.0,
                    "sum_tke": 0.0,
                    "sum_mvpe": 0.0,
                    "inside": 0,
                    "count": 0,
                }
            )
    return candidates


def update_sps_candidates(candidates, pred, target, c, rel, tke, mvpe):
    p = pred[..., :c]
    t = target[..., :c]
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
        nil = (2.0 * half) / SIGMA_GLOBAL
        tight = np.exp(-nil)
        elem_mask = inside * tight
        cand["sum_dm"] += float(np.sum(w_dm * elem_mask))
        cand["sum_tke"] += float(np.sum(w_tke * elem_mask))
        cand["sum_mvpe"] += float(np.sum(w_mvpe * elem_mask))
        cand["inside"] += int(np.sum(inside))
        cand["count"] += count


def finalize_scores(metric_sums: dict, candidates: list[dict], logistic_sps: bool = False):
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
        cnt = max(cand["count"], 1)
        sps_raw = 0.5 * cand["sum_dm"] / cnt + 0.3 * cand["sum_tke"] / cnt + 0.2 * cand["sum_mvpe"] / cnt
        if logistic_sps:
            sps_score = 100.0 / (1.0 + math.exp(-max(-60.0, min(60.0, sps_raw))))
        else:
            sps_score = 100.0 * sps_raw
        final = float(np.mean([base["rel_l2_score"], base["tke_score"], base["mvpe_score"], base["time_score"], sps_score]))
        row = {
            "abs": cand["abs"],
            "rel": cand["rel"],
            "coverage": cand["inside"] / cnt,
            "sps_raw": sps_raw,
            "sps_score_linear": 100.0 * sps_raw,
            "sps_score_used": sps_score,
            "final_est": final,
        }
        rows.append(row)
    rows.sort(key=lambda x: x["final_est"], reverse=True)
    base["best_bounds"] = rows[:20]
    return base


def evaluate_mode(args, checkpoint_path: str, mode_name: str, device, max_batches: int | None, abs_widths, rel_widths):
    train_dataset = Foil(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        mode="train",
        dataset_type="real",
        mask_prob=args.mask_prob,
        noise_scale=args.noise_scale,
    )
    val_dataset = Foil(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        mode="val",
        dataset_type="real",
    )
    norm_dataset = Foil(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        mode="train",
        dataset_type="numerical",
    )
    normalizer = build_normalizer(args.normalizer, norm_dataset, device)
    model = load_model(train_dataset, device=device, **vars(args))
    meta = model.load_checkpoint(checkpoint_path, device)
    model.eval()
    loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    metric_sums = {
        "n": 0,
        "rel_l2_sum": 0.0,
        "tke_sum": 0.0,
        "mvpe_sum": 0.0,
        "time_sum": 0.0,
        "time_n": 0,
    }
    candidates = init_sps_candidates(abs_widths, rel_widths)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            torch.cuda.synchronize() if device.type == "cuda" else None
            start = time.perf_counter()
            if mode_name == "normalized":
                x_norm, y_norm = normalizer.preprocess(x, y)
                pred_norm = model(x_norm)
                _, pred = normalizer.postprocess(x_norm, pred_norm)
                _, target = normalizer.postprocess(x_norm, y_norm)
            elif mode_name == "raw":
                pred = model(x)
                target = y
            else:
                raise ValueError(mode_name)
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = time.perf_counter() - start
            pred_np = pred.detach().cpu().numpy().astype(np.float32)
            target_np = target.detach().cpu().numpy().astype(np.float32)
            pred_np[..., 2] = 0.0
            c, rel, tke, mvpe = update_metric_sums(metric_sums, pred_np, target_np, elapsed)
            update_sps_candidates(candidates, pred_np, target_np, c, rel, tke, mvpe)
            if batch_idx % 10 == 0:
                print(f"[{mode_name}] batch {batch_idx+1}/{len(loader)} n={metric_sums['n']}", flush=True)
    result = finalize_scores(metric_sums, candidates)
    result["mode"] = mode_name
    meta = meta or {}
    result["checkpoint_meta"] = {
        "iteration": meta.get("iteration"),
        "best_iteration": meta.get("best_iteration"),
        "best_val_loss": float(meta.get("best_val_loss", float("nan"))),
    }
    result["max_batches"] = max_batches
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/foil/cno.yaml")
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/realpde_competition_h5")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--results-path", default="/root/autodl-fs/realpde_runs/calibration_tmp")
    parser.add_argument("--out", default="/root/autodl-fs/realpde_runs/bounds_calibration.json")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=["raw", "normalized"], choices=["raw", "normalized"])
    args_cli = parser.parse_args()

    set_seed(args_cli.seed)
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = make_args(args_cli.config, args_cli.dataset_root, args_cli.checkpoint, args_cli.results_path)
    Path(args_cli.results_path).mkdir(parents=True, exist_ok=True)
    Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)

    abs_widths = np.array(
        [0.0, 0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060, 0.080],
        dtype=np.float32,
    )
    rel_widths = np.array([0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20], dtype=np.float32)

    all_results = []
    print(f"device={device} checkpoint={args_cli.checkpoint}", flush=True)
    for mode in args_cli.modes:
        result = evaluate_mode(args, args_cli.checkpoint, mode, device, args_cli.max_batches, abs_widths, rel_widths)
        all_results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    payload = {
        "checkpoint": args_cli.checkpoint,
        "dataset_root": args_cli.dataset_root,
        "results": all_results,
    }
    Path(args_cli.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args_cli.out}")


if __name__ == "__main__":
    main()
