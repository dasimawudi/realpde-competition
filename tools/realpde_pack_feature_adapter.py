#!/usr/bin/env python3
"""Package a trained feature-adapter CNO checkpoint for Codabench submission."""

from __future__ import annotations

import argparse
import os
import shutil
import textwrap
import zipfile
from pathlib import Path


def extract_cno(template_zip: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(template_zip, "r") as zf:
        for name in ("rpde_baselines/__init__.py", "rpde_baselines/cno.py"):
            if name not in zf.namelist():
                raise FileNotFoundError(f"{template_zip} does not contain {name}")
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))


def submission_source(bound_abs: float, bound_rel: float, hidden: int, include_pressure: bool) -> str:
    return textwrap.dedent(
        f"""
        import os
        import traceback

        import numpy as np


        _MODEL = None
        _MODEL_ERROR = None
        _BOUND_ABS = {bound_abs!r}
        _BOUND_REL = {bound_rel!r}
        _ADAPTER_HIDDEN = {int(hidden)!r}
        _INCLUDE_PRESSURE = {bool(include_pressure)!r}


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


        def _central_diff_numpy(arr, axis):
            axis = axis % arr.ndim
            out = np.empty_like(arr)
            n = arr.shape[axis]
            if n <= 1:
                out[...] = 0.0
                return out
            middle = [slice(None)] * arr.ndim
            before = [slice(None)] * arr.ndim
            after = [slice(None)] * arr.ndim
            middle[axis] = slice(1, -1)
            before[axis] = slice(None, -2)
            after[axis] = slice(2, None)
            out[tuple(middle)] = 0.5 * (arr[tuple(after)] - arr[tuple(before)])
            first = [slice(None)] * arr.ndim
            second = [slice(None)] * arr.ndim
            first[axis] = 0
            second[axis] = 1
            out[tuple(first)] = arr[tuple(second)] - arr[tuple(first)]
            last = [slice(None)] * arr.ndim
            prev = [slice(None)] * arr.ndim
            last[axis] = -1
            prev[axis] = -2
            out[tuple(last)] = arr[tuple(last)] - arr[tuple(prev)]
            return out


        def _backward_diff_numpy(arr, axis):
            axis = axis % arr.ndim
            out = np.zeros_like(arr)
            if arr.shape[axis] <= 1:
                return out
            current = [slice(None)] * arr.ndim
            previous = [slice(None)] * arr.ndim
            current[axis] = slice(1, None)
            previous[axis] = slice(None, -1)
            out[tuple(current)] = arr[tuple(current)] - arr[tuple(previous)]
            return out


        def _coordinate_features_numpy(shape_without_channels, dtype):
            t, h, w = shape_without_channels[-3:]
            leading = (1,) * (len(shape_without_channels) - 3)
            x = np.linspace(-1.0, 1.0, w, dtype=dtype).reshape(leading + (1, 1, w))
            y = np.linspace(-1.0, 1.0, h, dtype=dtype).reshape(leading + (1, h, 1))
            tt = np.linspace(-1.0, 1.0, t, dtype=dtype).reshape(leading + (t, 1, 1))
            return (
                np.broadcast_to(x, shape_without_channels),
                np.broadcast_to(y, shape_without_channels),
                np.broadcast_to(tt, shape_without_channels),
            )


        def _feature_names(include_pressure=True):
            raw = ["u", "v", "p"] if include_pressure else ["u", "v"]
            return raw + [
                "speed",
                "kinetic_energy",
                "du_dt",
                "dv_dt",
                "vorticity",
                "divergence",
                "strain_magnitude",
                "x_coord",
                "y_coord",
                "t_coord",
            ]


        def _augment_numpy(x, include_pressure=True, eps=1e-6):
            arr = np.asarray(x, dtype=np.float32)
            if arr.ndim < 5 or arr.shape[-1] < 2:
                raise ValueError(f"unexpected input shape {{arr.shape}}")
            u = arr[..., 0]
            v = arr[..., 1]
            p = arr[..., 2] if arr.shape[-1] >= 3 else np.zeros_like(u)
            du_dt = _backward_diff_numpy(u, axis=-3)
            dv_dt = _backward_diff_numpy(v, axis=-3)
            du_dy = _central_diff_numpy(u, axis=-2)
            du_dx = _central_diff_numpy(u, axis=-1)
            dv_dy = _central_diff_numpy(v, axis=-2)
            dv_dx = _central_diff_numpy(v, axis=-1)
            speed = np.sqrt(u * u + v * v + eps).astype(np.float32, copy=False)
            kinetic = (0.5 * (u * u + v * v)).astype(np.float32, copy=False)
            vorticity = (dv_dx - du_dy).astype(np.float32, copy=False)
            divergence = (du_dx + dv_dy).astype(np.float32, copy=False)
            strain = np.sqrt((du_dx - dv_dy) ** 2 + (du_dy + dv_dx) ** 2 + eps).astype(np.float32, copy=False)
            x_coord, y_coord, t_coord = _coordinate_features_numpy(u.shape, arr.dtype)
            channels = [u, v]
            if include_pressure:
                channels.append(p)
            channels.extend([speed, kinetic, du_dt, dv_dt, vorticity, divergence, strain, x_coord, y_coord, t_coord])
            return np.stack(channels, axis=-1).astype(np.float32, copy=False)


        def _central_diff_torch(arr, axis):
            import torch

            axis = axis % arr.ndim
            out = torch.empty_like(arr)
            n = arr.shape[axis]
            if n <= 1:
                out.zero_()
                return out
            middle = [slice(None)] * arr.ndim
            before = [slice(None)] * arr.ndim
            after = [slice(None)] * arr.ndim
            middle[axis] = slice(1, -1)
            before[axis] = slice(None, -2)
            after[axis] = slice(2, None)
            out[tuple(middle)] = 0.5 * (arr[tuple(after)] - arr[tuple(before)])
            first = [slice(None)] * arr.ndim
            second = [slice(None)] * arr.ndim
            first[axis] = 0
            second[axis] = 1
            out[tuple(first)] = arr[tuple(second)] - arr[tuple(first)]
            last = [slice(None)] * arr.ndim
            prev = [slice(None)] * arr.ndim
            last[axis] = -1
            prev[axis] = -2
            out[tuple(last)] = arr[tuple(last)] - arr[tuple(prev)]
            return out


        def _backward_diff_torch(arr, axis):
            import torch

            axis = axis % arr.ndim
            out = torch.zeros_like(arr)
            if arr.shape[axis] <= 1:
                return out
            current = [slice(None)] * arr.ndim
            previous = [slice(None)] * arr.ndim
            current[axis] = slice(1, None)
            previous[axis] = slice(None, -1)
            out[tuple(current)] = arr[tuple(current)] - arr[tuple(previous)]
            return out


        def _coordinate_features_torch(u):
            import torch

            shape = tuple(u.shape)
            t, h, w = shape[-3:]
            leading = (1,) * (len(shape) - 3)
            x = torch.linspace(-1.0, 1.0, w, device=u.device, dtype=u.dtype).reshape(leading + (1, 1, w))
            y = torch.linspace(-1.0, 1.0, h, device=u.device, dtype=u.dtype).reshape(leading + (1, h, 1))
            tt = torch.linspace(-1.0, 1.0, t, device=u.device, dtype=u.dtype).reshape(leading + (t, 1, 1))
            return x.expand(shape), y.expand(shape), tt.expand(shape)


        def _augment_torch(x, include_pressure=True, eps=1e-6):
            import torch

            if x.dtype != torch.float32:
                x = x.float()
            u = x[..., 0]
            v = x[..., 1]
            p = x[..., 2] if x.shape[-1] >= 3 else torch.zeros_like(u)
            du_dt = _backward_diff_torch(u, axis=-3)
            dv_dt = _backward_diff_torch(v, axis=-3)
            du_dy = _central_diff_torch(u, axis=-2)
            du_dx = _central_diff_torch(u, axis=-1)
            dv_dy = _central_diff_torch(v, axis=-2)
            dv_dx = _central_diff_torch(v, axis=-1)
            speed = torch.sqrt(u * u + v * v + eps)
            kinetic = 0.5 * (u * u + v * v)
            vorticity = dv_dx - du_dy
            divergence = du_dx + dv_dy
            strain = torch.sqrt((du_dx - dv_dy) ** 2 + (du_dy + dv_dx) ** 2 + eps)
            x_coord, y_coord, t_coord = _coordinate_features_torch(u)
            channels = [u, v]
            if include_pressure:
                channels.append(p)
            channels.extend([speed, kinetic, du_dt, dv_dt, vorticity, divergence, strain, x_coord, y_coord, t_coord])
            return torch.stack(channels, dim=-1)


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
                    "checkpoint did not match feature-adapter CNO; "
                    f"missing_keys={{missing[:20]}}, unexpected_keys={{unexpected[:20]}}"
                )


        def _load_model(device):
            import torch
            from torch import nn
            from rpde_baselines.cno import CNO3d

            class PointwiseFeatureAdapter(nn.Module):
                def __init__(self, hidden=_ADAPTER_HIDDEN, include_pressure=_INCLUDE_PRESSURE, dropout=0.0):
                    super().__init__()
                    self.include_pressure = bool(include_pressure)
                    n_features = len(_feature_names(self.include_pressure))
                    self.net = nn.Sequential(
                        nn.LayerNorm(n_features),
                        nn.Linear(n_features, int(hidden)),
                        nn.SiLU(),
                        nn.Dropout(float(dropout)),
                        nn.Linear(int(hidden), 3),
                    )

                def forward(self, x):
                    features = _augment_torch(x, include_pressure=self.include_pressure)
                    delta = self.net(features)
                    raw = x[..., :3] if x.shape[-1] >= 3 else torch.cat([x[..., :2], torch.zeros_like(x[..., :1])], dim=-1)
                    return raw + delta

            class FeatureAdapterModel(nn.Module):
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
                    self.adapter = PointwiseFeatureAdapter()

                def forward(self, x):
                    return self.base_model(self.adapter(x))

            submission_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(submission_dir, "model.pth")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model = FeatureAdapterModel().to(device)
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
                        print("feature-adapter CNO failed to load; falling back to persistence.")
                        print(_MODEL_ERROR)

                if _MODEL is None:
                    return _with_bounds(_persistence(input_array))

                x = np.asarray(input_array, dtype=np.float32)
                with torch.inference_mode():
                    tensor = torch.from_numpy(x).to(device)
                    pred = _MODEL(tensor)
                    pred = pred.detach().cpu().numpy().astype(np.float32, copy=False)

                if pred.shape != (x.shape[0], 20, 32, 64, 3):
                    print(f"feature-adapter CNO returned unexpected shape {{pred.shape}}; falling back to persistence.")
                    return _with_bounds(_persistence(input_array))
                pred[..., 2] = 0.0
                if not np.isfinite(pred).all():
                    print("feature-adapter CNO returned non-finite values; falling back to persistence.")
                    return _with_bounds(_persistence(input_array))
                return _with_bounds(pred)

            except Exception:
                print("predict() failed; falling back to persistence.")
                print(traceback.format_exc())
                return _with_bounds(_persistence(input_array))
        """
    ).lstrip()


def package_submission(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    template_zip = Path(args.template_zip)
    out_zip = Path(args.out)
    build_dir = Path(args.build_dir)

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not template_zip.exists():
        raise FileNotFoundError(template_zip)
    if build_dir.exists():
        resolved = build_dir.resolve()
        if not resolved.name.startswith("_feature_adapter_build"):
            raise RuntimeError(f"refusing to remove non-temporary build dir: {resolved}")
        shutil.rmtree(resolved)
    build_dir.mkdir(parents=True)

    shutil.copy2(checkpoint, build_dir / "model.pth")
    extract_cno(template_zip, build_dir)
    (build_dir / "submission.py").write_text(
        submission_source(
            bound_abs=args.bound_abs,
            bound_rel=args.bound_rel,
            hidden=args.hidden,
            include_pressure=not args.drop_pressure_feature,
        ),
        encoding="utf-8",
    )

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(build_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(build_dir).as_posix())

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"wrote {out_zip} ({size_mb:.2f} MB)")
    if size_mb > args.max_size_mb:
        raise RuntimeError(f"submission exceeds {args.max_size_mb} MB: {size_mb:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="FeatureAdapterModel checkpoint.")
    parser.add_argument(
        "--template-zip",
        default="submission_cno_tke4100_lam215_microa020_abs0075_rel0075_nobench_20260830.zip",
        help="Existing CNO submission zip to reuse rpde_baselines/cno.py from.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--build-dir", default="_feature_adapter_build")
    parser.add_argument("--bound-abs", type=float, default=0.0075)
    parser.add_argument("--bound-rel", type=float, default=0.0075)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--drop-pressure-feature", action="store_true")
    parser.add_argument("--max-size-mb", type=float, default=256.0)
    args = parser.parse_args()
    package_submission(args)


if __name__ == "__main__":
    main()
