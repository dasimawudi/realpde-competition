#!/usr/bin/env python3
"""Package a residual-corrected CNO checkpoint for Codabench submission."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import textwrap
import zipfile
from pathlib import Path

import torch


def extract_cno(template_zip: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(template_zip, "r") as zf:
        for name in ("rpde_baselines/__init__.py", "rpde_baselines/cno.py"):
            if name not in zf.namelist():
                raise FileNotFoundError(f"{template_zip} does not contain {name}")
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))


def checkpoint_meta(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return {}
    return {
        "corrector_config": checkpoint.get("corrector_config", {}),
        "best_alpha": checkpoint.get("best_alpha"),
        "best_bound_abs": checkpoint.get("best_bound_abs"),
        "best_bound_rel": checkpoint.get("best_bound_rel"),
        "best_score": checkpoint.get("best_score"),
        "best_iteration": checkpoint.get("best_iteration"),
    }


def submission_source(
    *,
    bound_abs: float,
    bound_rel: float,
    alpha: float,
    hidden: int,
    blocks: int,
    dropout: float,
    include_pressure: bool,
    max_delta: float,
    history_context: bool,
) -> str:
    return textwrap.dedent(
        f"""
        import os
        import traceback

        import numpy as np


        _MODEL = None
        _MODEL_ERROR = None
        _BOUND_ABS = {float(bound_abs)!r}
        _BOUND_REL = {float(bound_rel)!r}
        _CORRECTION_ALPHA = {float(alpha)!r}
        _HIDDEN = {int(hidden)!r}
        _BLOCKS = {int(blocks)!r}
        _DROPOUT = {float(dropout)!r}
        _INCLUDE_PRESSURE = {bool(include_pressure)!r}
        _MAX_DELTA = {float(max_delta)!r}
        _HISTORY_CONTEXT = {bool(history_context)!r}


        def _persistence(input_array):
            x = np.asarray(input_array)
            pred = np.repeat(x[:, -1:, :, :, :], 20, axis=1).astype(np.float32, copy=False)
            if pred.shape[-1] >= 3:
                pred[..., 2] = 0.0
            return pred


        def _with_bounds(pred):
            pred = np.asarray(pred, dtype=np.float32)
            half_width = (_BOUND_ABS + _BOUND_REL * np.abs(pred)).astype(np.float32)
            if pred.shape[-1] >= 3:
                half_width[..., 2] = 0.0
            return {{"prediction": pred, "lower": pred - half_width, "upper": pred + half_width}}


        def _load_state_dict_flexible(module, state):
            fixed = {{}}
            for key, value in state.items():
                new_key = key
                for prefix in ("module.", "model."):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                fixed[new_key] = value
            result = module.load_state_dict(fixed, strict=False)
            missing = list(getattr(result, "missing_keys", []))
            unexpected = list(getattr(result, "unexpected_keys", []))
            if missing or unexpected:
                raise RuntimeError(
                    "checkpoint did not match residual-corrected CNO; "
                    f"missing_keys={{missing[:20]}}, unexpected_keys={{unexpected[:20]}}"
                )


        def _load_model(device):
            import torch
            from torch import nn
            from rpde_baselines.cno import CNO3d
            from realpde_feature_engineering import (
                augment_torch,
                feature_names,
                future_context_feature_count,
                future_context_torch,
            )

            torch.backends.cudnn.benchmark = False

            def ensure_three_channels(x):
                if x.shape[-1] >= 3:
                    return x[..., :3]
                return torch.cat([x[..., :2], torch.zeros_like(x[..., :1])], dim=-1)

            def zero_pressure(x):
                if x.shape[-1] < 3:
                    return x
                y = x.clone()
                y[..., 2] = 0.0
                return y

            def future_linear_extrapolation(x, out_steps):
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

            def future_feature_count(include_pressure, history_context):
                count = 2 * len(feature_names(include_pressure=include_pressure)) + 9
                if history_context:
                    count += future_context_feature_count()
                return count

            def build_future_features(x, base_pred, include_pressure, history_context):
                base = zero_pressure(ensure_three_channels(base_pred))
                out_steps = int(base.shape[1])
                last_raw = ensure_three_channels(x[:, -1:]).expand(-1, out_steps, -1, -1, -1)
                last_raw = zero_pressure(last_raw)
                linear = future_linear_extrapolation(x, out_steps)
                base_features = augment_torch(base, include_pressure=include_pressure)
                past_features = augment_torch(ensure_three_channels(x), include_pressure=include_pressure)
                last_features = past_features[:, -1:].expand(-1, out_steps, -1, -1, -1)
                pieces = [base_features, last_features, linear, base - last_raw, base - linear]
                if history_context:
                    pieces.append(future_context_torch(x, out_steps))
                return torch.cat(pieces, dim=-1)

            def norm_groups(channels):
                for groups in (8, 4, 2):
                    if channels % groups == 0:
                        return groups
                return 1

            class ResidualBlock3D(nn.Module):
                def __init__(self, channels, dropout=0.0):
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

                def forward(self, x):
                    return self.act(x + self.net(x))

            class ResidualCorrector3D(nn.Module):
                def __init__(self):
                    super().__init__()
                    in_channels = future_feature_count(
                        include_pressure=_INCLUDE_PRESSURE,
                        history_context=_HISTORY_CONTEXT,
                    )
                    self.input_norm = nn.LayerNorm(in_channels)
                    layers = [
                        nn.Conv3d(in_channels, _HIDDEN, kernel_size=3, padding=1),
                        nn.GroupNorm(norm_groups(_HIDDEN), _HIDDEN),
                        nn.SiLU(),
                    ]
                    for _ in range(_BLOCKS):
                        layers.append(ResidualBlock3D(_HIDDEN, dropout=_DROPOUT))
                    layers.append(nn.Conv3d(_HIDDEN, 3, kernel_size=1))
                    self.net = nn.Sequential(*layers)

                def forward(self, x, base_pred):
                    features = build_future_features(
                        x,
                        base_pred,
                        include_pressure=_INCLUDE_PRESSURE,
                        history_context=_HISTORY_CONTEXT,
                    )
                    features = self.input_norm(features)
                    z = features.permute(0, 4, 1, 2, 3).contiguous()
                    raw_delta = self.net(z).permute(0, 2, 3, 4, 1).contiguous()
                    if _MAX_DELTA > 0:
                        delta = _MAX_DELTA * torch.tanh(raw_delta / _MAX_DELTA)
                    else:
                        delta = raw_delta
                    delta = delta.clone()
                    delta[..., 2] = 0.0
                    return delta

            class ResidualCorrectionModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.base_model = CNO3d(
                        in_dim=3,
                        out_dim=3,
                        out_dim_mult=1,
                        in_size=64,
                        N_layers=3,
                        activation="LeakyReLU",
                    )
                    self.corrector = ResidualCorrector3D()

                def forward(self, x):
                    x = ensure_three_channels(x)
                    base = self.base_model(x)
                    base = zero_pressure(ensure_three_channels(base))
                    delta = self.corrector(x, base)
                    pred = base + _CORRECTION_ALPHA * delta
                    return zero_pressure(pred)

            submission_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(submission_dir, "model.pth")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model = ResidualCorrectionModel().to(device)
            _load_state_dict_flexible(model, state)
            model.eval()
            return model


        def predict(input_array, metadata=None):
            global _MODEL, _MODEL_ERROR
            try:
                import torch

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if _MODEL is None and _MODEL_ERROR is None:
                    try:
                        _MODEL = _load_model(device)
                    except Exception:
                        _MODEL_ERROR = traceback.format_exc()
                        print("residual-corrected CNO failed to load; falling back to persistence.")
                        print(_MODEL_ERROR)

                if _MODEL is None:
                    return _with_bounds(_persistence(input_array))

                x = np.asarray(input_array, dtype=np.float32)
                with torch.inference_mode():
                    tensor = torch.from_numpy(x).to(device)
                    pred = _MODEL(tensor)
                    pred = pred.detach().cpu().numpy().astype(np.float32, copy=False)

                if pred.shape != (x.shape[0], 20, 32, 64, 3):
                    print(f"residual-corrected CNO returned unexpected shape {{pred.shape}}; falling back to persistence.")
                    return _with_bounds(_persistence(input_array))
                pred[..., 2] = 0.0
                if not np.isfinite(pred).all():
                    print("residual-corrected CNO returned non-finite values; falling back to persistence.")
                    return _with_bounds(_persistence(input_array))
                return _with_bounds(pred)

            except Exception:
                print("predict() failed; falling back to persistence.")
                print(traceback.format_exc())
                return _with_bounds(_persistence(input_array))
        """
    ).lstrip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--template-zip", type=Path, required=True)
    parser.add_argument("--feature-file", type=Path, default=Path("tools/realpde_feature_engineering.py"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=Path("_residual_corrector_build"))
    parser.add_argument("--bound-abs", type=float, default=None)
    parser.add_argument("--bound-rel", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--include-pressure", action="store_true")
    parser.add_argument("--drop-pressure-feature", action="store_true")
    parser.add_argument("--max-delta", type=float, default=None)
    parser.add_argument("--history-context", action="store_true")
    parser.add_argument("--no-history-context", action="store_true")
    parser.add_argument("--max-size-mb", type=float, default=256.0)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.template_zip.exists():
        raise FileNotFoundError(args.template_zip)
    if not args.feature_file.exists():
        raise FileNotFoundError(args.feature_file)

    meta = checkpoint_meta(args.checkpoint)
    config = dict(meta.get("corrector_config") or {})
    alpha = args.alpha if args.alpha is not None else float(meta.get("best_alpha") or 0.0)
    bound_abs = args.bound_abs if args.bound_abs is not None else float(meta.get("best_bound_abs") or 0.0075)
    bound_rel = args.bound_rel if args.bound_rel is not None else float(meta.get("best_bound_rel") or 0.015)
    hidden = args.hidden if args.hidden is not None else int(config.get("hidden", 32))
    blocks = args.blocks if args.blocks is not None else int(config.get("blocks", 2))
    dropout = args.dropout if args.dropout is not None else float(config.get("dropout", 0.0))
    max_delta = args.max_delta if args.max_delta is not None else float(config.get("max_delta", 0.05))
    if args.history_context and args.no_history_context:
        raise ValueError("choose only one of --history-context or --no-history-context")
    if args.history_context:
        history_context = True
    elif args.no_history_context:
        history_context = False
    else:
        history_context = bool(config.get("history_context", False))
    if args.drop_pressure_feature:
        include_pressure = False
    elif args.include_pressure:
        include_pressure = True
    else:
        include_pressure = bool(config.get("include_pressure", True))

    if args.build_dir.exists():
        resolved = args.build_dir.resolve()
        if not resolved.name.startswith("_residual_corrector_build"):
            raise RuntimeError(f"refusing to remove non-temporary build dir: {resolved}")
        shutil.rmtree(resolved)
    args.build_dir.mkdir(parents=True)

    shutil.copy2(args.checkpoint, args.build_dir / "model.pth")
    shutil.copy2(args.feature_file, args.build_dir / "realpde_feature_engineering.py")
    extract_cno(args.template_zip, args.build_dir)
    (args.build_dir / "submission.py").write_text(
        submission_source(
            bound_abs=bound_abs,
            bound_rel=bound_rel,
            alpha=alpha,
            hidden=hidden,
            blocks=blocks,
            dropout=dropout,
            include_pressure=include_pressure,
            max_delta=max_delta,
            history_context=history_context,
        ),
        encoding="utf-8",
    )
    (args.build_dir / "residual_submission_meta.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_meta": meta,
                "alpha": alpha,
                "bound_abs": bound_abs,
                "bound_rel": bound_rel,
                "hidden": hidden,
                "blocks": blocks,
                "dropout": dropout,
                "include_pressure": include_pressure,
                "max_delta": max_delta,
                "history_context": history_context,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(args.build_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(args.build_dir).as_posix())
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.2f} MB)")
    print(
        json.dumps(
            {
                "alpha": alpha,
                "bound_abs": bound_abs,
                "bound_rel": bound_rel,
                "hidden": hidden,
                "blocks": blocks,
                "include_pressure": include_pressure,
                "max_delta": max_delta,
                "history_context": history_context,
                "size_mb": size_mb,
            },
            indent=2,
        )
    )
    if size_mb > args.max_size_mb:
        raise RuntimeError(f"submission exceeds {args.max_size_mb} MB: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
