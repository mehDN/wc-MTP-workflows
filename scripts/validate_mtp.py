#!/usr/bin/env python3
"""Parse mlp calc-errors output and check convergence thresholds."""

import argparse
import re
import sys


METRICS = {
    "force_rms": re.compile(r"Forces:[\s\S]*?RMS\s+absolute difference = ([0-9.eE+-]+)"),
    "force_mae": re.compile(r"Forces:[\s\S]*?Average absolute difference = ([0-9.eE+-]+)"),
    "energy_per_atom_rms": re.compile(
        r"Energy per atom:[\s\S]*?RMS\s+absolute difference = ([0-9.eE+-]+)"
    ),
    "energy_per_atom_mae": re.compile(
        r"Energy per atom:[\s\S]*?Average absolute difference = ([0-9.eE+-]+)"
    ),
    "stress_rms": re.compile(
        r"Stresses:[\s\S]*?RMS\s+absolute difference = ([0-9.eE+-]+)"
    ),
    "stress_mae": re.compile(
        r"Stresses:[\s\S]*?Average absolute difference = ([0-9.eE+-]+)"
    ),
}


def parse_metrics(text):
    out = {}
    for name, pattern in METRICS.items():
        match = pattern.search(text)
        if match:
            out[name] = float(match.group(1))
    return out


def main():
    parser = argparse.ArgumentParser(description="Validate MTP errors against thresholds.")
    parser.add_argument("errors_log", help="calc-errors log file")
    parser.add_argument("--force-rms-max", type=float, default=0.08)
    parser.add_argument("--force-mae-max", type=float, default=0.08)
    parser.add_argument("--energy-per-atom-max", type=float, default=0.005)
    parser.add_argument("--stress-rms-max", type=float, default=0.5)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as key=value lines suitable for shell parsing",
    )
    args = parser.parse_args()

    text = open(args.errors_log).read()
    metrics = parse_metrics(text)
    if not metrics:
        print(f"Could not parse metrics from {args.errors_log}", file=sys.stderr)
        sys.exit(2)

    checks = {
        "force_rms": metrics.get("force_rms", float("inf")) <= args.force_rms_max,
        "force_mae": metrics.get("force_mae", float("inf")) <= args.force_mae_max,
        "energy_per_atom_rms": metrics.get("energy_per_atom_rms", float("inf"))
        <= args.energy_per_atom_max,
    }
    if "stress_rms" in metrics:
        checks["stress_rms"] = metrics["stress_rms"] <= args.stress_rms_max

    passed = all(checks.values())

    if args.json:
        for key, value in metrics.items():
            print(f"{key}={value}")
        print(f"passed={'yes' if passed else 'no'}")
    else:
        print("Validation metrics:")
        for key, value in metrics.items():
            status = "OK" if checks.get(key, True) else "FAIL"
            print(f"  {key:22s} {value:.6g}  [{status}]")
        print()
        if passed:
            print("Convergence criteria met.")
        else:
            print("Convergence criteria NOT met.")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()