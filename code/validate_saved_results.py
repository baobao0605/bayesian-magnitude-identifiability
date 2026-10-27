#!/usr/bin/env python3
"""Check the supplied recovery data, fits, intervals, and summaries without fitting.

The checks evaluate the existing likelihood at saved estimates, reconstruct the
reported intervals at those estimates, and recompute descriptive summaries.
No optimizer, random-number generator, new fit, or new simulation is called.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.special import logit, ndtr

import recovery_simulation as recovery

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_csv(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def typed_value(value):
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "":
        return value
    try:
        return float(value)
    except ValueError:
        return value


def compare(expected, observed, location="", differences=None):
    if differences is None:
        differences = []
    if isinstance(expected, dict):
        if set(expected) != set(observed):
            differences.append(location + ": different fields")
        else:
            for key in expected:
                compare(expected[key], observed[key], location + "/" + key, differences)
    elif isinstance(expected, list):
        if len(expected) != len(observed):
            differences.append(location + ": different lengths")
        else:
            for i, (a, b) in enumerate(zip(expected, observed)):
                compare(a, b, location + "/" + str(i), differences)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not isinstance(observed, (float, int)):
            differences.append(location + ": nonnumeric comparison")
        elif math.isnan(float(expected)) and math.isnan(float(observed)):
            pass
        elif not math.isclose(float(expected), float(observed), rel_tol=1e-8, abs_tol=1e-9):
            differences.append(location + ": numerical mismatch")
    elif expected != observed:
        differences.append(location + ": value mismatch")
    return differences


def main():
    config = json.loads((RESULTS / "recovery_design.json").read_text())
    saved_summary = json.loads((RESULTS / "recovery_summary.json").read_text())
    fits = json.loads((RESULTS / "recovery_fits.json").read_text())
    fit_csv = load_csv("recovery_estimates.csv")
    interval_csv = load_csv("recovery_intervals.csv")
    optimizer_csv = load_csv("recovery_optimizer.csv")
    n = config["replicates"]
    assert n == 200 and config["seed"] == 2026102617
    assert len(fits) == len(fit_csv) == 400
    assert len(interval_csv) == 3200
    assert len(optimizer_csv) == 1200
    expected_keys = {(i, regime) for i in range(n) for regime in ("known_gain", "noisy_gain")}
    assert {(row["replicate"], row["regime"]) for row in fits} == expected_keys
    for expected, row in zip(fits, fit_csv):
        observed = {key: typed_value(value) for key, value in row.items()}
        # This field is a semicolon-separated coordinate list, not a number.
        observed["active_coordinates"] = row["active_coordinates"]
        assert not compare(expected, observed), expected["replicate"]
    stored_intervals = [{key: typed_value(value) for key, value in row.items()} for row in interval_csv]
    design = recovery.design_arrays(config)
    truth = recovery.truth_parameters(config)
    assert not compare(saved_summary["truth"], truth)
    shapes = {}
    objective_errors = []
    rebuilt_intervals = []
    with np.load(RESULTS / "recovery_simulated_data.npz", allow_pickle=False) as arrays:
        for name in arrays.files:
            assert np.isfinite(arrays[name]).all(), name
            shapes[name] = list(arrays[name].shape)
        assert arrays["choices"].shape == (200, 25, 24)
        assert arrays["reproductions"].shape == (200, 100)
        assert arrays["calibrations"].shape == (200, 16)
        assert np.isin(arrays["choices"], [0, 1]).all()
        for name, key in [("delta_t", "dt"), ("delta_distance", "dl"), ("choice_t1", "t1"),
                          ("choice_t2", "t2"), ("reproduction_distance", "ell")]:
            assert np.array_equal(arrays[name], design[key]), name
        probability = ndtr((design["dt"] + truth["b"] * design["dl"]) / (truth["s"] * design["D"]))
        assert np.allclose(arrays["true_choice_probability"], probability, rtol=1e-14, atol=1e-14)
        for fit in fits:
            i = fit["replicate"]
            noisy = fit["regime"] == "noisy_gain"
            theta = np.array([math.log(fit["b"]), math.log(fit["s"]), float(logit(fit["w"]))]
                             + ([math.log(fit["g"])] if noisy else []))
            pars = recovery.unpack(theta, noisy, config["gain_calibration"]["known_gain"])
            for key in ("b", "s", "w", "v", "w_m", "g", "q"):
                assert math.isclose(pars[key], fit[key], rel_tol=1e-10, abs_tol=1e-10)
            counts = arrays["choices"][i].sum(axis=1)
            objective, _ = recovery.joint_objective(theta, counts, arrays["reproductions"][i],
                                                   arrays["calibrations"][i], design, config, noisy)
            objective_errors.append(abs(objective - fit["objective"]))
            assert math.isclose(objective, fit["objective"], rel_tol=1e-10, abs_tol=1e-8)
            alpha = float(arrays["reproductions"][i].mean() - fit["q"] * design["ell"].mean())
            assert math.isclose(alpha, fit["alpha"], rel_tol=1e-12, abs_tol=1e-12)
            traces = [row for row in optimizer_csv if int(row["replicate"]) == i and row["regime"] == fit["regime"]]
            assert len(traces) == 3 and {int(row["start"]) for row in traces} == {0, 1, 2}
            chosen = next(row for row in traces if int(row["start"]) == fit["selected_start"])
            assert math.isclose(float(chosen["objective"]), fit["objective"], rel_tol=1e-13, abs_tol=1e-10)
            assert math.isclose(min(float(row["objective"]) for row in traces if row["finite"] == "True"),
                                fit["objective"], rel_tol=1e-13, abs_tol=1e-10)
            rebuilt_intervals.extend(recovery.intervals(theta, fit, truth, design, config, noisy))
            if noisy:
                rebuilt_intervals.extend(recovery.intervals(theta, fit, truth, design, config, noisy, ignore_gain=True))
    interval_differences = compare(stored_intervals, rebuilt_intervals)
    assert not interval_differences, interval_differences[:10]
    summary = recovery.summarize(fits, stored_intervals, truth, n)
    summary_differences = compare(saved_summary["procedures"], summary)
    assert not summary_differences, summary_differences[:10]
    checks = recovery.numerical_checks(design, config, truth)
    recovery.write_json(RESULTS / "recovery_numerical_checks.json", checks)
    report = {
        "status": "completed",
        "scope": "Saved data and estimates only; likelihood and intervals evaluated at existing fitted points, and summaries recomputed. No fitting or sampling.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "replicates": n,
        "fit_rows": len(fits),
        "interval_rows": len(stored_intervals),
        "optimizer_start_rows": len(optimizer_csv),
        "data_shapes": shapes,
        "fits_json_and_csv_agree": True,
        "saved_estimates_satisfy_parameter_transformations": True,
        "saved_objectives_match_raw_data_likelihoods": True,
        "maximum_absolute_objective_difference": max(objective_errors),
        "selected_start_matches_minimum_saved_objective": True,
        "recomputed_intervals_match_saved_intervals": True,
        "recomputed_summaries_match_saved_summaries": True,
        "relative_comparison_tolerance": 1e-8,
        "absolute_comparison_tolerance": 1e-9,
        "deterministic_score_and_information_checks": checks,
        "speed_summary": {p: {"rmse": summary[p]["parameters"]["v"]["rmse"],
                              "covered_count": summary[p]["parameters"]["v"]["covered_count"],
                              "valid_interval_count": summary[p]["parameters"]["v"]["valid_interval_count"],
                              "near_boundary_count": summary[p]["near_boundary_count"]}
                          for p in ("known_gain", "noisy_gain", "noisy_gain_ignored")},
        "new_random_draws": 0,
        "new_fits": 0,
        "new_simulation_replicates": 0,
        "human_participant_data": False,
    }
    recovery.write_json(RESULTS / "saved_results_validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
