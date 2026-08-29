#!/usr/bin/env python3
"""Compare RealPDE Track 1 foil checkpoints across model architectures.

The script is intended to run inside a RealPDEBench checkout.  It reuses the
metric and SPS-calibration helpers from realpde_calibrate_bounds.py so scores are
comparable with the CNO tuning workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from huggingface_hub import hf_hub_download


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from realpde_calibrate_bounds import evaluate_mode, make_args, set_seed  # noqa: E402


REPO_ID = "AI4Science-WestlakeU/RealPDEBench-models"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    arch: str
    config: str
    batch_size: int
    hf_path: str | None = None
    local_checkpoint: str | None = None
    modes: tuple[str, ...] = ("raw", "normalized")
    overrides: dict[str, int | float | str] | None = None
    note: str = ""


SPECS: dict[str, ModelSpec] = {
    "cno_official": ModelSpec(
        name="cno_official",
        arch="CNO",
        config="realpdebench/configs/foil/cno.yaml",
        batch_size=64,
        hf_path="foil/cno/finetune.pth",
        modes=("raw", "normalized"),
        note="official foil/cno finetune checkpoint",
    ),
    "cno_tke_ours": ModelSpec(
        name="cno_tke_ours",
        arch="CNO",
        config="realpdebench/configs/foil/cno_competition_realft_full.yaml",
        batch_size=64,
        local_checkpoint="/root/autodl-fs/realpde_runs/cno_tke_ft_gentle_20260827_1045/model_best.pth",
        modes=("raw",),
        note="our TKE/SPS-aware continuation checkpoint",
    ),
    "transolver_official": ModelSpec(
        name="transolver_official",
        arch="Transolver",
        config="realpdebench/configs/foil/trainsolver.yaml",
        batch_size=16,
        hf_path="foil/transolver/finetune.pth",
        modes=("raw", "normalized"),
        overrides={"H": 32, "W": 64, "D": 20},
        note="small and submission-friendly",
    ),
    "unet_official": ModelSpec(
        name="unet_official",
        arch="UNet",
        config="realpdebench/configs/foil/unet.yaml",
        batch_size=12,
        hf_path="foil/unet/finetune.pth",
        modes=("raw", "normalized"),
        overrides={"dim": 64},
        note="medium-size convolutional baseline",
    ),
    "deeponet_official": ModelSpec(
        name="deeponet_official",
        arch="DeepONet",
        config="realpdebench/configs/foil/deeponet.yaml",
        batch_size=64,
        hf_path="foil/deeponet/finetune.pth",
        modes=("raw", "normalized"),
        note="very small checkpoint",
    ),
    "mwt_official": ModelSpec(
        name="mwt_official",
        arch="MWT",
        config="realpdebench/configs/foil/mwt.yaml",
        batch_size=16,
        hf_path="foil/mwt/finetune.pth",
        modes=("raw", "normalized"),
        note="small wavelet-transformer baseline",
    ),
    "dpot_s_official": ModelSpec(
        name="dpot_s_official",
        arch="DPOT-S",
        config="realpdebench/configs/foil/dpot_s.yaml",
        batch_size=4,
        hf_path="foil/dpot_s/finetune.pth",
        modes=("raw", "normalized"),
        note="large but still under 256 MB as a single checkpoint",
    ),
    "fno_official": ModelSpec(
        name="fno_official",
        arch="FNO",
        config="realpdebench/configs/foil/fno.yaml",
        batch_size=8,
        hf_path="foil/fno/finetune.pth",
        modes=("raw", "normalized"),
        note="strong classic neural operator, but fp32 checkpoint exceeds submit limit",
    ),
}


ABS_WIDTHS = [0.0, 0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060, 0.080]
REL_WIDTHS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]


def parse_model_list(value: str) -> list[str]:
    if value == "small":
        return [
            "cno_official",
            "cno_tke_ours",
            "transolver_official",
            "unet_official",
            "deeponet_official",
            "mwt_official",
            "dpot_s_official",
        ]
    if value == "all":
        return list(SPECS)
    return [part.strip() for part in value.split(",") if part.strip()]


def checkpoint_path(spec: ModelSpec, model_dir: Path, repo_id: str) -> Path:
    if spec.local_checkpoint:
        path = Path(spec.local_checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"local checkpoint missing: {path}")
        return path
    if not spec.hf_path:
        raise ValueError(f"spec {spec.name} has neither hf_path nor local_checkpoint")
    local_target = model_dir / spec.hf_path
    if local_target.exists() and local_target.stat().st_size > 0:
        return local_target
    model_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=spec.hf_path,
        repo_type="model",
        local_dir=str(model_dir),
    )
    return Path(path)


def summarize_result(spec: ModelSpec, mode: str, ckpt: Path, result: dict, elapsed_wall_s: float) -> dict:
    best = (result.get("best_bounds") or [{}])[0]
    size_mb = ckpt.stat().st_size / (1024 * 1024)
    return {
        "name": spec.name,
        "arch": spec.arch,
        "mode": mode,
        "status": "ok",
        "checkpoint": str(ckpt),
        "checkpoint_mb": size_mb,
        "batch_size": spec.batch_size,
        "n": result.get("n"),
        "final_est": best.get("final_est"),
        "rel_l2": result.get("rel_l2"),
        "tke": result.get("tke"),
        "mvpe": result.get("mvpe"),
        "rel_l2_score": result.get("rel_l2_score"),
        "tke_score": result.get("tke_score"),
        "mvpe_score": result.get("mvpe_score"),
        "time_score": result.get("time_score"),
        "sps_score": best.get("sps_score_used"),
        "bound_abs": best.get("abs"),
        "bound_rel": best.get("rel"),
        "coverage": best.get("coverage"),
        "mean_t_neural_s": result.get("mean_t_neural_s"),
        "wall_s": elapsed_wall_s,
        "note": spec.note,
    }


def write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_one(
    spec: ModelSpec,
    *,
    dataset_root: str,
    model_dir: Path,
    results_path: Path,
    repo_id: str,
    max_batches: int | None,
    workers: int,
    modes_override: tuple[str, ...] | None,
    out_jsonl: Path,
) -> list[dict]:
    ckpt = checkpoint_path(spec, model_dir, repo_id)
    modes = modes_override or spec.modes
    rows: list[dict] = []
    for mode in modes:
        started = time.perf_counter()
        try:
            args = make_args(spec.config, dataset_root, str(ckpt), str(results_path / spec.name / mode))
            if spec.overrides:
                for key, value in spec.overrides.items():
                    setattr(args, key, value)
            args.test_batch_size = spec.batch_size
            args.num_workers = workers
            result = evaluate_mode(
                args,
                str(ckpt),
                mode,
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                max_batches,
                ABS_WIDTHS,
                REL_WIDTHS,
            )
            row = summarize_result(spec, mode, ckpt, result, time.perf_counter() - started)
        except Exception as exc:
            row = {
                "name": spec.name,
                "arch": spec.arch,
                "mode": mode,
                "status": "error",
                "checkpoint": str(ckpt) if "ckpt" in locals() else None,
                "batch_size": spec.batch_size,
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=8),
                "wall_s": time.perf_counter() - started,
                "note": spec.note,
            }
        rows.append(row)
        write_jsonl(out_jsonl, row)
        print("ROW " + json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="small", help="'small', 'all', or comma-separated spec names")
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/realpde_competition_h5")
    parser.add_argument("--model-dir", default="/root/autodl-fs/realpde_models")
    parser.add_argument("--results-path", default="/root/autodl-fs/realpde_runs/arch_compare_tmp")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--out", default="/root/autodl-fs/realpde_runs/architecture_compare.json")
    parser.add_argument("--jsonl", default="/root/autodl-fs/realpde_runs/architecture_compare.jsonl")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--modes", default=None, help="override modes, e.g. raw or normalized or raw,normalized")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    names = parse_model_list(args.models)
    unknown = [name for name in names if name not in SPECS]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; available={sorted(SPECS)}")

    model_dir = Path(args.model_dir)
    results_path = Path(args.results_path)
    out_path = Path(args.out)
    jsonl_path = Path(args.jsonl)
    if jsonl_path.exists():
        jsonl_path.unlink()

    modes_override = tuple(part.strip() for part in args.modes.split(",") if part.strip()) if args.modes else None

    all_rows: list[dict] = []
    for name in names:
        spec = SPECS[name]
        print(f"START {spec.name} arch={spec.arch}", flush=True)
        all_rows.extend(
            run_one(
                spec,
                dataset_root=args.dataset_root,
                model_dir=model_dir,
                results_path=results_path,
                repo_id=args.repo_id,
                max_batches=args.max_batches,
                workers=args.workers,
                modes_override=modes_override,
                out_jsonl=jsonl_path,
            )
        )

    all_rows.sort(key=lambda row: (row.get("status") == "ok", row.get("final_est") or -1), reverse=True)
    payload = {
        "created_at_unix": time.time(),
        "dataset_root": args.dataset_root,
        "max_batches": args.max_batches,
        "rows": all_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()
