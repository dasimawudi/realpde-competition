#!/usr/bin/env python3
"""Evaluate a trained RealPDE residual-corrected CNO checkpoint on HDF5 data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from realpde_h5_feature_adapter_train import (  # noqa: E402
    BAD_TRAIN_FILES,
    H5WindowDataset,
    list_h5,
    load_cno_class,
    split_paths,
)
from realpde_h5_residual_corrector_train import (  # noqa: E402
    CorrectorConfig,
    ResidualCorrectionModel,
    ResidualCorrector3D,
    evaluate_alphas,
    parse_float_list,
)


def flexible_load(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    fixed = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        fixed[new_key] = value
    result = module.load_state_dict(fixed, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(
            "residual checkpoint mismatch: "
            f"missing_keys={missing[:20]}, unexpected_keys={unexpected[:20]}"
        )


def load_model(checkpoint_path: Path, realpdebench_root: Path, device: torch.device) -> ResidualCorrectionModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"expected residual checkpoint with model_state_dict: {checkpoint_path}")
    config_dict = dict(checkpoint.get("corrector_config") or {})
    config = CorrectorConfig(
        hidden=int(config_dict.get("hidden", 32)),
        blocks=int(config_dict.get("blocks", 2)),
        dropout=float(config_dict.get("dropout", 0.0)),
        include_pressure=bool(config_dict.get("include_pressure", True)),
        max_delta=float(config_dict.get("max_delta", 0.05)),
    )
    CNO3d = load_cno_class(realpdebench_root)
    base_model = CNO3d(
        in_dim=3,
        out_dim=3,
        out_dim_mult=1,
        in_size=64,
        N_layers=3,
        activation="LeakyReLU",
    )
    model = ResidualCorrectionModel(base_model, ResidualCorrector3D(config)).to(device)
    flexible_load(model, checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--realpdebench-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--in-steps", type=int, default=20)
    parser.add_argument("--out-steps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--sub-sample", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-windows-per-trajectory", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--include-pressure-data", action="store_true")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--eval-alphas", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--bound-abs", default="0.005,0.0075,0.01,0.0125,0.015")
    parser.add_argument("--bound-rel", default="0,0.0025,0.005,0.0075,0.01,0.0125,0.015,0.02,0.025,0.03")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    paths = list_h5(args.real_root, BAD_TRAIN_FILES)
    _, val_paths = split_paths(paths, args.val_fraction, args.seed)
    dataset = H5WindowDataset(
        val_paths,
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        stride=args.stride,
        sub_sample=args.sub_sample,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        include_pressure=args.include_pressure_data,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(args.checkpoint, args.realpdebench_root, device)
    summaries = evaluate_alphas(
        model,
        loader,
        device,
        alphas=parse_float_list(args.eval_alphas),
        abs_widths=parse_float_list(args.bound_abs),
        rel_widths=parse_float_list(args.bound_rel),
        max_batches=args.max_eval_batches,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "val_windows": len(dataset),
        "top": summaries[0],
        "summaries": summaries,
    }
    print(json.dumps(result, indent=2, default=str), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
