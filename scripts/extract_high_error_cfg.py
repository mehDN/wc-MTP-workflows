#!/usr/bin/env python3
"""Extract high force-error configurations for force-focused MTP retrain.

Compares a DFT reference .cfg with an MTP-predicted .cfg (from `mlp calc-efs`)
and writes the DFT configs with the largest per-config force RMSE.

Usage:
  mlp calc-efs pot.mtp train.cfg pred.cfg
  ./scripts/extract_high_error_cfg.py train.cfg pred.cfg high_err.cfg \\
      --top-frac 0.15 --min-force-rmse 0.15
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def read_configs(path: str) -> list[list[str]]:
    configs: list[list[str]] = []
    current: list[str] = []
    in_cfg = False
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped == "BEGIN_CFG":
                if current:
                    configs.append(current)
                current = [line]
                in_cfg = True
            elif stripped == "END_CFG" and in_cfg:
                current.append(line)
                configs.append(current)
                current = []
                in_cfg = False
            elif in_cfg:
                current.append(line)
    if current:
        raise ValueError(f"Incomplete configuration block at end of {path}")
    return configs


def write_configs(path: str, configs: list[list[str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for cfg in configs:
            fh.writelines(cfg)


def parse_forces(lines: list[str]) -> list[tuple[float, float, float]]:
    forces: list[tuple[float, float, float]] = []
    in_atomdata = False
    has_fx = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("AtomData:"):
            in_atomdata = True
            headers = line.lower().split()
            has_fx = "fx" in headers
            continue
        if in_atomdata:
            if (
                line.startswith("Energy")
                or line.startswith("PlusStress")
                or line.startswith("Feature")
                or line == "END_CFG"
            ):
                break
            if not has_fx:
                continue
            toks = line.split()
            if len(toks) >= 8:
                try:
                    forces.append((float(toks[5]), float(toks[6]), float(toks[7])))
                except ValueError:
                    pass
    return forces


def force_rmse(
    ref: list[tuple[float, float, float]],
    pred: list[tuple[float, float, float]],
) -> float | None:
    if not ref or not pred or len(ref) != len(pred):
        return None
    s = 0.0
    n = 0
    for (fx, fy, fz), (px, py, pz) in zip(ref, pred):
        dx, dy, dz = fx - px, fy - py, fz - pz
        s += dx * dx + dy * dy + dz * dz
        n += 3
    if n == 0:
        return None
    return math.sqrt(s / n)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select high force-error DFT configs for force-weighted retrain."
    )
    parser.add_argument("dft_cfg", help="Reference DFT .cfg")
    parser.add_argument("mtp_cfg", help="MTP-predicted .cfg from calc-efs")
    parser.add_argument("output_cfg", help="Output high-error DFT subset .cfg")
    parser.add_argument(
        "--top-frac",
        type=float,
        default=0.15,
        help="Keep this top fraction by force RMSE (default 0.15)",
    )
    parser.add_argument(
        "--min-force-rmse",
        type=float,
        default=0.15,
        help="Also keep any config with force RMSE >= this (eV/A)",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="Optional hard cap on number of configs (0 = no cap)",
    )
    parser.add_argument(
        "--scores",
        default="",
        help="Optional TSV of index, force_rmse for debugging",
    )
    args = parser.parse_args()

    if not (0.0 < args.top_frac <= 1.0):
        parser.error("--top-frac must be in (0, 1]")

    dft = read_configs(args.dft_cfg)
    mtp = read_configs(args.mtp_cfg)
    if len(dft) != len(mtp):
        print(
            f"Config count mismatch: DFT={len(dft)} MTP={len(mtp)}",
            file=sys.stderr,
        )
        return 1
    if not dft:
        print("Empty configuration files", file=sys.stderr)
        return 1

    scores: list[tuple[int, float]] = []
    for i, (d_lines, m_lines) in enumerate(zip(dft, mtp)):
        rmse = force_rmse(parse_forces(d_lines), parse_forces(m_lines))
        if rmse is None:
            continue
        scores.append((i, rmse))

    if not scores:
        print("Could not compute force errors for any configuration", file=sys.stderr)
        return 1

    scores.sort(key=lambda t: t[1], reverse=True)
    n_top = max(1, int(math.ceil(len(scores) * args.top_frac)))
    selected_idx = {idx for idx, _ in scores[:n_top]}
    for idx, rmse in scores:
        if rmse >= args.min_force_rmse:
            selected_idx.add(idx)

    ordered = [idx for idx, _ in scores if idx in selected_idx]
    if args.max_configs > 0:
        ordered = ordered[: args.max_configs]

    out = [dft[i] for i in ordered]
    write_configs(args.output_cfg, out)

    kept_rmses = [r for i, r in scores if i in selected_idx]
    print(
        f"Selected {len(out)}/{len(dft)} high-error configs "
        f"(top_frac={args.top_frac}, min_force_rmse={args.min_force_rmse})"
    )
    print(
        f"Force RMSE among selected: "
        f"min={min(kept_rmses):.6g} median={sorted(kept_rmses)[len(kept_rmses)//2]:.6g} "
        f"max={max(kept_rmses):.6g}"
    )
    print(f"Wrote {args.output_cfg}")

    if args.scores:
        Path(args.scores).parent.mkdir(parents=True, exist_ok=True)
        with open(args.scores, "w") as fh:
            fh.write("index\tforce_rmse\tselected\n")
            for idx, rmse in scores:
                fh.write(f"{idx}\t{rmse:.8g}\t{1 if idx in selected_idx else 0}\n")
        print(f"Wrote scores to {args.scores}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
