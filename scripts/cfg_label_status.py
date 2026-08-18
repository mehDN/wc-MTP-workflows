#!/usr/bin/env python3
"""Classify MLIP .cfg blocks as DFT-labeled (Energy + forces) or not.

Used by the active-learning loop so already-labeled AIMD/OUTCAR frames are
merged and retrained instead of being sent back to VASP.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def read_configs(path):
    configs = []
    current = []
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


def _next_nonempty(lines, start):
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped:
            return i, stripped
        i += 1
    return None, ""


def is_labeled(lines):
    """True if the block has a numeric Energy and AtomData force columns."""
    has_energy = False
    has_forces = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "Energy" or stripped.startswith("Energy "):
            _, value = _next_nonempty(lines, i + 1)
            try:
                float(value.split()[0])
                has_energy = True
            except (ValueError, IndexError):
                pass
        if stripped.startswith("AtomData:") and "fx" in stripped and "fy" in stripped:
            has_forces = True
        if has_energy and has_forces:
            return True
    return False


def write_configs(path, configs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for cfg in configs:
            fh.writelines(cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Report / extract DFT-labeled vs unlabeled MLIP configs."
    )
    parser.add_argument("cfg", help="Input .cfg file")
    parser.add_argument(
        "--extract-labeled",
        metavar="OUT",
        help="Write configs that already have Energy + forces",
    )
    parser.add_argument(
        "--extract-unlabeled",
        metavar="OUT",
        help="Write configs that still need DFT",
    )
    args = parser.parse_args()

    path = Path(args.cfg)
    if not path.is_file():
        print(f"Missing cfg: {path}", file=sys.stderr)
        sys.exit(2)

    configs = read_configs(path)
    labeled = []
    unlabeled = []
    for cfg in configs:
        if is_labeled(cfg):
            labeled.append(cfg)
        else:
            unlabeled.append(cfg)

    print(f"N_CFG={len(configs)}")
    print(f"N_LABELED={len(labeled)}")
    print(f"N_UNLABELED={len(unlabeled)}")

    if args.extract_labeled:
        if labeled:
            write_configs(args.extract_labeled, labeled)
        elif Path(args.extract_labeled).exists():
            Path(args.extract_labeled).unlink()
    if args.extract_unlabeled:
        if unlabeled:
            write_configs(args.extract_unlabeled, unlabeled)
        elif Path(args.extract_unlabeled).exists():
            Path(args.extract_unlabeled).unlink()


if __name__ == "__main__":
    main()
