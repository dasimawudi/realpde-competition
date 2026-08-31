#!/usr/bin/env python3
"""Create a CNO Codabench zip by replacing model.pth in a known-good template."""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path


def patch_bounds(source: str, bound_abs: float, bound_rel: float) -> str:
    source = re.sub(r"^_BOUND_ABS\s*=\s*[-+0-9.eE]+", f"_BOUND_ABS = {bound_abs!r}", source, flags=re.MULTILINE)
    source = re.sub(r"^_BOUND_REL\s*=\s*[-+0-9.eE]+", f"_BOUND_REL = {bound_rel!r}", source, flags=re.MULTILINE)
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-zip", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=Path("_cno_template_build"))
    parser.add_argument("--bound-abs", type=float, default=0.0075)
    parser.add_argument("--bound-rel", type=float, default=0.0075)
    parser.add_argument("--max-size-mb", type=float, default=256.0)
    args = parser.parse_args()

    if not args.template_zip.exists():
        raise FileNotFoundError(args.template_zip)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.build_dir.exists():
        resolved = args.build_dir.resolve()
        if not resolved.name.startswith("_cno_template_build"):
            raise RuntimeError(f"refusing to remove non-temporary build dir: {resolved}")
        shutil.rmtree(resolved)
    args.build_dir.mkdir(parents=True)

    with zipfile.ZipFile(args.template_zip, "r") as zf:
        zf.extractall(args.build_dir)
    shutil.copy2(args.checkpoint, args.build_dir / "model.pth")
    submission_path = args.build_dir / "submission.py"
    submission_path.write_text(
        patch_bounds(submission_path.read_text(encoding="utf-8"), args.bound_abs, args.bound_rel),
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
    if size_mb > args.max_size_mb:
        raise RuntimeError(f"submission exceeds {args.max_size_mb} MB: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
