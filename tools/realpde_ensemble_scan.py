#!/usr/bin/env python3
"""Scan simple ensembles between the tuned CNO and selected official models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from realpde_calibrate_bounds import (  # noqa: E402
    build_normalizer,
    finalize_scores,
    init_sps_candidates,
    make_args,
    measured_channels,
    update_metric_sums,
    update_sps_candidates,
)
from realpde_compare_architectures import SPECS, checkpoint_path  # noqa: E402
from realpdebench.data.fluid_dataset import Foil  # noqa: E402
from realpdebench.model.load_model import load_model  # noqa: E402


BASE_CNO_CKPT = "/root/autodl-fs/realpde_runs/cno_tke_ft_gentle_20260827_1045/model_best.pth"
BASE_CNO_CONFIG = "realpdebench/configs/foil/cno_competition_realft_full.yaml"


def build_predictor(config, checkpoint, mode, dataset_root, results_path, device, overrides=None):
    args = make_args(config, dataset_root, str(checkpoint), str(results_path))
    if overrides:
        for key, value in overrides.items():
            setattr(args, key, value)
    train_dataset = Foil(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        mode="train",
        dataset_type="real",
        mask_prob=args.mask_prob,
        noise_scale=args.noise_scale,
    )
    norm_dataset = Foil(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        mode="train",
        dataset_type="numerical",
    )
    normalizer = build_normalizer(args.normalizer, norm_dataset, device)
    model = load_model(train_dataset, device=device, **vars(args))
    model.load_checkpoint(str(checkpoint), device)
    model.eval()
    return {"args": args, "model": model, "normalizer": normalizer, "mode": mode}


def predict(predictor, x, y):
    model = predictor["model"]
    mode = predictor["mode"]
    normalizer = predictor["normalizer"]
    if mode == "normalized":
        x_norm, y_norm = normalizer.preprocess(x, y)
        pred_norm = model(x_norm)
        _, pred = normalizer.postprocess(x_norm, pred_norm)
    elif mode == "raw":
        pred = model(x)
    else:
        raise ValueError(mode)
    pred = pred.detach().cpu().numpy().astype(np.float32)
    pred[..., 2] = 0.0
    return pred


def fresh_sums():
    return {
        "n": 0,
        "rel_l2_sum": 0.0,
        "tke_sum": 0.0,
        "mvpe_sum": 0.0,
        "time_sum": 0.0,
        "time_n": 0,
    }


def scan_candidate(name, mode, *, dataset_root, model_dir, results_path, device, alphas, abs_widths, rel_widths, batch_size, workers):
    spec = SPECS[name]
    ckpt = checkpoint_path(spec, model_dir, spec.__dict__.get("repo_id", "AI4Science-WestlakeU/RealPDEBench-models"))
    base = build_predictor(BASE_CNO_CONFIG, BASE_CNO_CKPT, "raw", dataset_root, results_path / "base_cno", device)
    cand = build_predictor(spec.config, ckpt, mode, dataset_root, results_path / name / mode, device, spec.overrides)
    val_dataset = Foil(dataset_name="foil", dataset_root=dataset_root, mode="val", dataset_type="real")
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    states = {}
    for alpha in alphas:
        states[float(alpha)] = {
            "metric_sums": fresh_sums(),
            "candidates": init_sps_candidates(abs_widths, rel_widths),
            "time_sum": 0.0,
        }

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()
            pred_base = predict(base, x, y)
            pred_cand = predict(cand, x, y)
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = time.perf_counter() - t0
            target = y.detach().cpu().numpy().astype(np.float32)
            c = measured_channels(target)
            for alpha, state in states.items():
                pred = (1.0 - alpha) * pred_base + alpha * pred_cand
                _, rel, tke, mvpe = update_metric_sums(state["metric_sums"], pred, target, elapsed)
                update_sps_candidates(state["candidates"], pred, target, c, rel, tke, mvpe)
            if batch_idx % 10 == 0:
                print(f"{name}:{mode} batch {batch_idx + 1}/{len(loader)}", flush=True)

    rows = []
    for alpha, state in states.items():
        result = finalize_scores(state["metric_sums"], state["candidates"])
        best = result["best_bounds"][0]
        rows.append(
            {
                "candidate": name,
                "candidate_arch": spec.arch,
                "candidate_mode": mode,
                "alpha_candidate": alpha,
                "final_est": best["final_est"],
                "rel_l2_score": result["rel_l2_score"],
                "tke_score": result["tke_score"],
                "mvpe_score": result["mvpe_score"],
                "time_score": result["time_score"],
                "sps_score": best["sps_score_used"],
                "bound_abs": best["abs"],
                "bound_rel": best["rel"],
                "coverage": best["coverage"],
                "candidate_checkpoint_mb": ckpt.stat().st_size / (1024 * 1024),
            }
        )
    rows.sort(key=lambda row: row["final_est"], reverse=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/realpde_competition_h5")
    parser.add_argument("--model-dir", default="/root/autodl-fs/realpde_models")
    parser.add_argument("--results-path", default="/root/autodl-fs/realpde_runs/ensemble_scan_tmp")
    parser.add_argument("--out", default="/root/autodl-fs/realpde_runs/ensemble_scan.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    results_path = Path(args.results_path)
    alphas = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    abs_widths = [0.005, 0.0075, 0.010, 0.0125, 0.015]
    rel_widths = [0.0, 0.02, 0.05]
    targets = [
        ("unet_official", "normalized"),
        ("mwt_official", "raw"),
        ("transolver_official", "raw"),
    ]
    all_rows = []
    for name, mode in targets:
        print(f"START ensemble {name}:{mode}", flush=True)
        rows = scan_candidate(
            name,
            mode,
            dataset_root=args.dataset_root,
            model_dir=model_dir,
            results_path=results_path,
            device=device,
            alphas=alphas,
            abs_widths=abs_widths,
            rel_widths=rel_widths,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        all_rows.extend(rows)
        print("BEST " + json.dumps(rows[0], ensure_ascii=False, sort_keys=True), flush=True)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    all_rows.sort(key=lambda row: row["final_est"], reverse=True)
    payload = {"rows": all_rows, "targets": targets, "alphas": alphas, "abs_widths": abs_widths, "rel_widths": rel_widths}
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
