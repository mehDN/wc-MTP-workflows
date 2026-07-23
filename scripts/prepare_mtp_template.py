#!/usr/bin/env python3
"""Prepare a W-C MTP template from MLIP untrained level files."""

import argparse
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a binary W-C MTP template at a given MLIP level."
    )
    parser.add_argument(
        "--level",
        type=int,
        default=20,
        choices=[20, 22, 24, 26, 28, 18, 16, 14, 12, 10, 8, 6, 4, 2],
        help="MTP level (20-22 recommended for WC W-vacancy reconstruction)",
    )
    parser.add_argument("--species-count", type=int, default=2)
    parser.add_argument("--min-dist", type=float, default=1.2)
    parser.add_argument("--max-dist", type=float, default=6.0)
    parser.add_argument("--radial-basis-size", type=int, default=11)
    parser.add_argument(
        "--mlip-root",
        type=Path,
        default=Path.home() / "software" / "mlip-2",
        help="MLIP installation root (contains untrained_mtps/)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def patch_header(text, species_count, min_dist, max_dist, radial_basis_size):
    text = re.sub(
        r"(?m)^species_count = \d+",
        f"species_count = {species_count}",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^(\s*)min_dist = [\d.]+",
        rf"\1min_dist = {min_dist}",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^(\s*)max_dist = [\d.]+",
        rf"\1max_dist = {max_dist}",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^(\s*)radial_basis_size = \d+",
        rf"\1radial_basis_size = {radial_basis_size}",
        text,
        count=1,
    )
    return text


def main():
    args = parse_args()
    src = args.mlip_root / "untrained_mtps" / f"{args.level:02d}.mtp"
    if not src.is_file():
        src = args.mlip_root / "untrained_mtps" / f"{args.level}.mtp"
    if not src.is_file():
        print(f"Level template not found under {args.mlip_root}/untrained_mtps/", file=sys.stderr)
        sys.exit(1)

    raw = src.read_text()
    patched = patch_header(
        raw,
        species_count=args.species_count,
        min_dist=args.min_dist,
        max_dist=args.max_dist,
        radial_basis_size=args.radial_basis_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(patched)
    print(
        f"Wrote {args.output} "
        f"(level={args.level}, species={args.species_count}, "
        f"cutoff={args.min_dist}-{args.max_dist} A, rb={args.radial_basis_size})"
    )


if __name__ == "__main__":
    main()