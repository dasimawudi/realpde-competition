#!/usr/bin/env python3
"""CNO weight-space interpolation scan on direct-HDF5 RealPDE Track 1 data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from realpde_h5_feature_adapter_train import (  # noqa: E402
    BAD_TRAIN_FILES,
    H5WindowDataset,
    evaluate,
    list_h5,
    load_cno_class,
    split_paths,
    unwrap_state,
)


DEFAULT_ALPHAS = "-0.35,-0.25,-0.18,-0.12,-0.08,-0.05,-0.03,0,0.02,0.05,0.08,0.12,0.18,0.25"


def parse_alphas(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one alpha is required")
    return values


def normalized_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw = unwrap_state(checkpoint)
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        new_key = key
        for prefix in ("module.", "model.", "base_model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        if new_key.startswith("adapter."):
            continue
        out[new_key] = value.detach().cpu() if torch.is_tensor(value) else value
    return out


def interpolate_state(
    base: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    alpha: float,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Return ``(1-alpha)*base + alpha*target`` where shapes match."""

    state: dict[str, torch.Tensor] = {}
    stats = {"matched": 0, "base_only": 0, "shape_mismatch": 0, "non_float": 0}
    for key, base_value in base.items():
        target_value = target.get(key)
        if target_value is None:
            state[key] = base_value.clone() if torch.is_tensor(base_value) else base_value
            stats["base_only"] += 1
            continue
        if not torch.is_tensor(base_value) or not torch.is_tensor(target_value):
            state[key] = base_value
            stats["non_float"] += 1
            continue
        if base_value.shape != target_value.shape:
            state[key] = base_value.clone()
            stats["shape_mismatch"] += 1
            continue
        if not torch.is_floating_point(base_value):
            state[key] = base_value.clone()
            stats["non_float"] += 1
            continue
        state[key] = ((1.0 - alpha) * base_value.float() + alpha * target_value.float()).to(base_value.dtype)
        stats["matched"] += 1
    return state, stats


def save_cno_checkpoint(path: Path, state: dict[str, torch.Tensor], row: dict, config: dict) -> None:
    torch.save(
        {
            "model_state_dict": state,
            "scan_row": row,
            "scan_config": config,
            "best_val_loss": -float(row["final_est"]),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="Current best CNO checkpoint.")
    parser.add_argument("--target", type=Path, required=True, help="Second CNO checkpoint to interpolate/extrapolate toward.")
    parser.add_argument("--realpdebench-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--in-steps", type=int, default=20)
    parser.add_argument("--out-steps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--sub-sample", type=int, default=2)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"out_dir already exists, refusing to overwrite: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    alphas = parse_alphas(args.alphas)
    paths = list_h5(args.real_root, BAD_TRAIN_FILES)
    _, val_paths = split_paths(paths, args.val_fraction, args.seed)
    val_set = H5WindowDataset(
        val_paths,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        stride=args.stride,
        sub_sample=args.sub_sample,
        include_pressure=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_loader = DataLoader(
        val_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    CNO3d = load_cno_class(args.realpdebench_root)
    model = CNO3d(
        in_dim=3,
        out_dim=3,
        out_dim_mult=1,
        in_size=64,
        N_layers=3,
        activation="LeakyReLU",
    ).to(device)
    abs_widths = np.array([0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030, 0.040], dtype=np.float32)
    rel_widths = np.array([0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20], dtype=np.float32)

    base_state = normalized_state_dict(args.base)
    target_state = normalized_state_dict(args.target)
    config = {
        "real_root": str(args.real_root),
        "base": str(args.base),
        "target": str(args.target),
        "realpdebench_root": str(args.realpdebench_root),
        "out_dir": str(args.out_dir),
        "alphas": alphas,
        "val_windows": len(val_set),
        "val_trajectories": len(val_paths),
        "max_eval_batches": args.max_eval_batches,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)

    rows = []
    best_row = None
    best_state = None
    for alpha in alphas:
        state, stats = interpolate_state(base_state, target_state, alpha)
        load_result = model.load_state_dict({k: v.to(device) if torch.is_tensor(v) else v for k, v in state.items()}, strict=False)
        missing = list(getattr(load_result, "missing_keys", []))
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        if missing or unexpected:
            raise RuntimeError(f"load mismatch at alpha={alpha}: missing={missing[:20]}, unexpected={unexpected[:20]}")
        result = evaluate(model, val_loader, device, abs_widths, rel_widths, args.max_eval_batches)
        top = result["best_bounds"][0]
        row = {
            "alpha": alpha,
            "final_est": top["final_est"],
            "bound_abs": top["abs"],
            "bound_rel": top["rel"],
            "rel_l2": result["rel_l2"],
            "tke": result["tke"],
            "mvpe": result["mvpe"],
            "time_score": result["time_score"],
            "sps_score": top["sps_score_used"],
            "coverage": top["coverage"],
            "stats": stats,
        }
        rows.append(row)
        print("ROW " + json.dumps(row, sort_keys=True), flush=True)
        if best_row is None or row["final_est"] > best_row["final_est"]:
            best_row = row
            best_state = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in state.items()}
            save_cno_checkpoint(args.out_dir / "model_best.pth", best_state, best_row, config)
            print(f"BEST alpha={alpha} final_est={row['final_est']:.6f}", flush=True)

    summary = {"config": config, "best": best_row, "rows": rows}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best_state is not None and best_row is not None:
        save_cno_checkpoint(args.out_dir / "model_best.pth", best_state, best_row, config)
    print("DONE " + json.dumps(best_row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
