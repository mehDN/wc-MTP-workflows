#!/usr/bin/env python3
"""Concatenate multiple MLIP .cfg files into one training set."""

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


def write_configs(path, configs):
    with open(path, "w") as fh:
        for cfg in configs:
            fh.writelines(cfg)


def main():
    parser = argparse.ArgumentParser(description="Merge MLIP .cfg files.")
    parser.add_argument("output_cfg", help="Merged output .cfg file")
    parser.add_argument("input_cfgs", nargs="+", help="Input .cfg files to merge")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop exact duplicate configuration blocks",
    )
    args = parser.parse_args()

    merged = []
    seen = set()
    per_file = {}

    for cfg_path in args.input_cfgs:
        path = Path(cfg_path)
        if not path.is_file():
            print(f"Skipping missing file: {path}", file=sys.stderr)
            continue
        configs = read_configs(path)
        kept = 0
        for cfg in configs:
            key = "".join(cfg)
            if args.dedupe and key in seen:
                continue
            seen.add(key)
            merged.append(cfg)
            kept += 1
        per_file[str(path)] = kept

    if not merged:
        print("No configurations to merge.", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output_cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_configs(out, merged)

    print(f"Wrote {len(merged)} configurations to {out}")
    for src, count in per_file.items():
        print(f"  {count:6d} from {src}")


if __name__ == "__main__":
    main()