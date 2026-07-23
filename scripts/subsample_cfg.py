#!/usr/bin/env python3
"""Subsample an MLIP .cfg file by keeping every Nth configuration."""

import argparse
import sys


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
    parser = argparse.ArgumentParser(
        description="Keep every Nth configuration from an MLIP .cfg training set."
    )
    parser.add_argument("input_cfg", help="Input .cfg file")
    parser.add_argument("output_cfg", help="Output .cfg file")
    parser.add_argument(
        "--stride",
        type=int,
        default=50,
        help="Keep one configuration every STRIDE frames (default: 50)",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional cap on number of kept configurations",
    )
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be >= 1")

    configs = read_configs(args.input_cfg)
    if not configs:
        print(f"No configurations found in {args.input_cfg}", file=sys.stderr)
        sys.exit(1)

    selected = configs[:: args.stride]
    if args.max_configs is not None:
        selected = selected[: args.max_configs]

    write_configs(args.output_cfg, selected)
    print(
        f"Wrote {len(selected)} configurations "
        f"(from {len(configs)}, stride={args.stride}) to {args.output_cfg}"
    )


if __name__ == "__main__":
    main()