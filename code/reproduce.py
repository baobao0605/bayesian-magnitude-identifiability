#!/usr/bin/env python3
"""Reproduce checks, summaries, and all five figures from supplied outputs.

By default, no new simulation or optimization is performed. The optional
--rerun-simulation flag explicitly repeats the original fixed 200-replicate
study before validating and redrawing its outputs.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-simulation", action="store_true",
                        help="Explicitly overwrite recovery outputs by repeating the fixed original 200-replicate study")
    args = parser.parse_args()
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    run_folder = build / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    run_folder.mkdir(exist_ok=False)
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(build / "matplotlib")
    steps = [
        ("exact_and_deterministic_checks", ["code/rederive_core.py"]),
        ("symbolic_relations", ["code/check_claim_relations.py"]),
    ]
    if args.rerun_simulation:
        steps.append(("fixed_recovery_simulation", ["code/recovery_simulation.py", "--rerun-simulation", "--overwrite"]))
    steps.extend([
        ("saved_recovery_results", ["code/validate_saved_results.py"]),
        ("interval_tail_summary", ["code/recovery_tail_diagnostics.py"]),
        ("main_figures", ["code/make_revision_figures.py"]),
        ("saved_supplementary_illustrations", ["code/plot_saved_illustrations.py"]),
        ("recovery_figure", ["code/recovery_simulation.py", "--plot-only"]),
    ])
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "running",
        "python": platform.python_version(),
        "mode": "repeat fixed recovery study" if args.rerun_simulation else "supplied data and outputs only",
        "new_recovery_simulation_requested": args.rerun_simulation,
        "legacy_random_illustration_rerun": False,
        "steps": [],
    }
    for index, (name, arguments) in enumerate(steps, 1):
        command = [sys.executable] + arguments
        started = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stdout_name = f"{index:02d}_{name}.stdout.txt"
        stderr_name = f"{index:02d}_{name}.stderr.txt"
        (run_folder / stdout_name).write_text(result.stdout, encoding="utf-8")
        (run_folder / stderr_name).write_text(result.stderr, encoding="utf-8")
        report["steps"].append({"name": name, "command": ["python"] + arguments,
                                "return_code": result.returncode,
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                                "stdout": stdout_name, "stderr": stderr_name})
        print(f"{index}/{len(steps)} {name}: exit {result.returncode}", flush=True)
        if result.returncode:
            report["status"] = "failed"
            (run_folder / "run_record.json").write_text(json.dumps(report, indent=2) + "\n")
            raise SystemExit(result.returncode)
    report["status"] = "completed"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["new_simulation_replicates"] = 200 if args.rerun_simulation else 0
    report["new_joint_fits"] = 400 if args.rerun_simulation else 0
    report["legacy_random_parameter_draws"] = 0
    (run_folder / "run_record.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "completed", "run_record": str(run_folder.relative_to(ROOT) / "run_record.json"),
                      "mode": report["mode"]}, indent=2))


if __name__ == "__main__":
    main()
