#!/usr/bin/env python3
"""Filter bad or irrelevant DFT configurations from an MLIP .cfg training set.

Drops configs that are likely harmful to MTP fit quality:
  - missing Energy or Forces
  - unphysically short min pair distance
  - extreme force magnitudes (DFT blow-ups / failed SCF)
  - energy-per-atom outliers relative to the dataset median
  - explicit bad labels (e.g. Feature EFS_by = VASP_not_converged)

Usage:
  ./scripts/filter_cfg.py train.cfg train_clean.cfg \\
      --max-force 50 --min-dist 0.5 --max-epa-outlier 5.0
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from pathlib import Path


# Default tags treated as bad / non-reference DFT.
DEFAULT_BAD_EFS_TAGS = (
    "VASP_not_converged",
    "not_converged",
    "failed",
    "bad",
    "irrelevant",
    "exclude",
)


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


def parse_cfg_stats(lines: list[str]) -> dict:
    """Extract size, energy, max |F|, mindist, and EFS_by tag from one config."""
    size = None
    energy = None
    max_force = 0.0
    mindist = None
    efs_by = None
    has_forces = False
    in_atomdata = False
    atomdata_has_force = False

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue

        if line == "Size" and i + 1 < len(lines):
            try:
                size = int(lines[i + 1].strip().split()[0])
            except (ValueError, IndexError):
                pass
            continue

        # Energy is typically on its own line; value on the next line.
        if line == "Energy" or line.startswith("Energy "):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    energy = float(parts[1])
                except ValueError:
                    pass
            elif i + 1 < len(lines):
                try:
                    energy = float(lines[i + 1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            continue

        if line.startswith("Feature"):
            # "Feature   mindist\t2.197" or "Feature   EFS_by\tVASP"
            rest = line[len("Feature") :].strip()
            # Split on whitespace (spaces/tabs)
            toks = rest.split()
            if len(toks) >= 2:
                key = toks[0].lower()
                val = toks[1]
                if key == "mindist":
                    try:
                        mindist = float(val)
                    except ValueError:
                        pass
                elif key in ("efs_by", "efs-by", "source"):
                    efs_by = val
            continue

        if line.startswith("AtomData:"):
            in_atomdata = True
            headers = line.lower().split()
            atomdata_has_force = "fx" in headers
            continue

        if in_atomdata:
            if (
                line.startswith("Energy")
                or line.startswith("PlusStress")
                or line.startswith("Feature")
                or line == "END_CFG"
            ):
                in_atomdata = False
                # re-process this line in outer logic next iteration — but we
                # already advanced; handle Energy/Feature above on next loops.
                # Force: break atomdata only; Energy is usually after AtomData.
                if line.startswith("Energy"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            energy = float(parts[1])
                        except ValueError:
                            pass
                    elif i + 1 < len(lines):
                        try:
                            energy = float(lines[i + 1].strip().split()[0])
                        except (ValueError, IndexError):
                            pass
                continue

            if atomdata_has_force:
                toks = line.split()
                # id type x y z fx fy fz
                if len(toks) >= 8:
                    try:
                        fx, fy, fz = float(toks[5]), float(toks[6]), float(toks[7])
                        fmag = math.sqrt(fx * fx + fy * fy + fz * fz)
                        if fmag > max_force:
                            max_force = fmag
                        has_forces = True
                    except ValueError:
                        pass

    if mindist is None:
        mindist = estimate_mindist(lines)

    epa = None
    if energy is not None and size and size > 0:
        epa = energy / size

    return {
        "size": size,
        "energy": energy,
        "epa": epa,
        "max_force": max_force if has_forces else None,
        "mindist": mindist,
        "efs_by": efs_by,
        "has_energy": energy is not None,
        "has_forces": has_forces,
    }


def estimate_mindist(lines: list[str]) -> float | None:
    """Cheap mindist from cartesians; samples large cells."""
    coords: list[tuple[float, float, float]] = []
    in_atomdata = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("AtomData:"):
            in_atomdata = True
            continue
        if in_atomdata:
            if (
                line.startswith("Energy")
                or line.startswith("PlusStress")
                or line.startswith("Feature")
                or line == "END_CFG"
            ):
                break
            toks = line.split()
            if len(toks) >= 5:
                try:
                    coords.append((float(toks[2]), float(toks[3]), float(toks[4])))
                except ValueError:
                    pass
    if len(coords) < 2:
        return None
    if len(coords) > 400:
        step = max(1, len(coords) // 200)
        coords = coords[::step]
    md = float("inf")
    n = len(coords)
    for i in range(n):
        xi, yi, zi = coords[i]
        for j in range(i + 1, n):
            dx = xi - coords[j][0]
            dy = yi - coords[j][1]
            dz = zi - coords[j][2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < md:
                md = d2
    if md == float("inf"):
        return None
    return math.sqrt(md)


def is_bad_efs_tag(tag: str | None, bad_tags: tuple[str, ...]) -> bool:
    if not tag:
        return False
    low = tag.lower()
    for b in bad_tags:
        if b.lower() in low:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter bad/irrelevant DFT configs from an MLIP .cfg file."
    )
    parser.add_argument("input_cfg", help="Input training .cfg")
    parser.add_argument("output_cfg", help="Filtered output .cfg")
    parser.add_argument(
        "--rejected-cfg",
        default="",
        help="Optional path to write rejected configurations",
    )
    parser.add_argument(
        "--max-force",
        type=float,
        default=50.0,
        help="Drop configs with any |F| above this (eV/A). 0 disables.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.5,
        help="Drop configs with mindist below this (A). 0 disables.",
    )
    parser.add_argument(
        "--max-epa-outlier",
        type=float,
        default=5.0,
        help="Drop configs with |E/atom - median| above this (eV). 0 disables.",
    )
    parser.add_argument(
        "--allow-no-forces",
        action="store_true",
        help="Keep configs without forces",
    )
    parser.add_argument(
        "--keep-not-converged",
        action="store_true",
        help="Do not drop Feature EFS_by = VASP_not_converged (etc.)",
    )
    parser.add_argument(
        "--bad-efs-tag",
        action="append",
        default=[],
        help="Extra EFS_by substring to reject (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts only; do not write output",
    )
    args = parser.parse_args()

    require_forces = not args.allow_no_forces
    bad_tags = DEFAULT_BAD_EFS_TAGS + tuple(args.bad_efs_tag)

    configs = read_configs(args.input_cfg)
    if not configs:
        print(f"No configurations in {args.input_cfg}", file=sys.stderr)
        return 1

    stats = [parse_cfg_stats(c) for c in configs]
    epas = [s["epa"] for s in stats if s["epa"] is not None]
    median_epa = statistics.median(epas) if epas else None

    kept: list[list[str]] = []
    rejected: list[list[str]] = []
    reasons: dict[str, int] = {}

    for cfg, s in zip(configs, stats):
        why = None
        if (
            not args.keep_not_converged
            and is_bad_efs_tag(s.get("efs_by"), bad_tags)
        ):
            why = f"bad_efs_by:{s.get('efs_by')}"
        elif require_forces and not s["has_forces"]:
            why = "no_forces"
        elif not s["has_energy"]:
            why = "no_energy"
        elif args.min_dist > 0 and s["mindist"] is not None and s["mindist"] < args.min_dist:
            why = "short_mindist"
        elif (
            args.max_force > 0
            and s["max_force"] is not None
            and s["max_force"] > args.max_force
        ):
            why = "huge_force"
        elif (
            args.max_epa_outlier > 0
            and median_epa is not None
            and s["epa"] is not None
            and abs(s["epa"] - median_epa) > args.max_epa_outlier
        ):
            why = "epa_outlier"

        if why:
            rejected.append(cfg)
            # group bad_efs_by:* under one key for summary
            key = "bad_efs_by" if why.startswith("bad_efs_by") else why
            reasons[key] = reasons.get(key, 0) + 1
        else:
            kept.append(cfg)

    print(f"Input:    {len(configs)} configs from {args.input_cfg}")
    print(f"Kept:     {len(kept)}")
    print(f"Rejected: {len(rejected)}")
    if reasons:
        for k, v in sorted(reasons.items()):
            print(f"  {k:16s} {v}")
    if median_epa is not None:
        print(f"Median E/atom: {median_epa:.6g} eV")

    if args.dry_run:
        return 0

    if not kept:
        print("All configurations rejected; refusing to write empty set.", file=sys.stderr)
        return 1

    write_configs(args.output_cfg, kept)
    print(f"Wrote {len(kept)} configs to {args.output_cfg}")
    if args.rejected_cfg and rejected:
        write_configs(args.rejected_cfg, rejected)
        print(f"Wrote {len(rejected)} rejected configs to {args.rejected_cfg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
