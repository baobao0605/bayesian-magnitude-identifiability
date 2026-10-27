#!/usr/bin/env python3
"""Reproduce descriptive tail diagnostics from the unchanged recovery CSVs.

No simulations, refits, interval changes, or exclusions are performed.
Defaults are resolved relative to this file, so the current directory is
irrelevant. Run with --output to preserve the shipped diagnostic JSON.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / "recovery_tail_diagnostics.json")
    args = parser.parse_args()
    with (args.results_dir / "recovery_intervals.csv").open(newline="") as stream:
        intervals = list(csv.DictReader(stream))
    with (args.results_dir / "recovery_estimates.csv").open(newline="") as stream:
        fits = list(csv.DictReader(stream))
    fit_by_key = {(int(row["replicate"]), row["regime"]): row for row in fits}
    output = {}
    for procedure in ("known_gain", "noisy_gain", "noisy_gain_ignored"):
        rows = [row for row in intervals if row["procedure"] == procedure
                and row["parameter"] == "v" and row["valid_interval"] == "True"]
        widths = np.array([float(row["width"]) for row in rows])
        largest = rows[int(np.argmax(widths))]
        regime = "noisy_gain" if procedure == "noisy_gain_ignored" else procedure
        fitted = fit_by_key[(int(largest["replicate"]), regime)]
        output[procedure] = {
            "valid_intervals": len(rows),
            "width_mean": float(np.mean(widths)),
            "width_median": float(np.median(widths)),
            "width_q90": float(np.quantile(widths, .9)),
            "width_q95": float(np.quantile(widths, .95)),
            "width_q99": float(np.quantile(widths, .99)),
            "width_max": float(np.max(widths)),
            "maximum_width_replicate": int(largest["replicate"]),
            "maximum_width_estimate_v": float(fitted["v"]),
            "maximum_width_estimate_w": float(fitted["w"]),
            "maximum_width_interval": [float(largest["lower"]), float(largest["upper"])],
            "information_condition_number": float(largest["information_condition_number"]),
        }
    output["boundary_fits"] = [
        {"replicate": int(row["replicate"]), "regime": row["regime"],
         "w": float(row["w"]), "v": float(row["v"]), "b": float(row["b"]),
         "g": float(row["g"]), "unconstrained_w": float(row["unconstrained_w"]),
         "active_coordinates": row["active_coordinates"]}
        for row in fits if row["near_boundary"] == "True"
    ]
    output["note"] = ("Post-run descriptive tail diagnostics from the fixed results; "
                      "no design, fit, interval, exclusion or random draw changed.")
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
