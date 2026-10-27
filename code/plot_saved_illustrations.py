#!/usr/bin/env python3
"""Redraw Figures S1 and S2 from the supplied ridge table and fixed equations.

This command makes no random draws and performs no fitting. The ridge table
was generated in the original 1,500-trial illustration, whose full observation
array was not retained. code/verify_equivalence.py provides its seeded generator.
"""
from pathlib import Path
from io import BytesIO
import json
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build" / "matplotlib"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
ridge = np.genfromtxt(ROOT / "results" / "synthetic_ridge.csv", delimiter=",", names=True)
ridge_w = ridge["weight"]
ridge_ll = ridge["synthetic_loglikelihood"]
base_w, base_v, base_wm = .8, .25, .2
base_beta = (1 - base_w) / (base_w * base_v)
base_s = base_wm / np.sqrt(base_w)
assert len(ridge) == 81
assert np.allclose(ridge["speed"], (1-ridge_w)/(ridge_w*base_beta), rtol=1e-13, atol=1e-14)
assert np.allclose(ridge["measurement_coefficient"], base_s*np.sqrt(ridge_w), rtol=1e-13, atol=1e-14)
assert np.ptp(ridge_ll) < 1e-9
ct = .8
cl = np.array([0., 1., 2., 4., 8., 16., 25.])
c_beta, c_wp, c_bias = .08, .15, .1
cw1, cw2 = .8, .5
cm1 = cw1 * ct + c_beta * cl
cm2 = cw2 * ct + c_beta * cl
cbias2 = c_bias + (cw1 - cw2) * ct
csd1, csd2 = c_wp * cm1, c_wp * cm2
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.labelsize": 9, "legend.fontsize": 8})
colors = ["#145b78", "#ba542f"]

def save_current_figure(fig, stem):
    # Serialize before replacing the destination, so a failed render cannot
    # leave an empty artifact that appears to be a completed PDF.
    for ext in ("pdf", "png"):
        buffer = BytesIO()
        fig.savefig(buffer, format=ext, dpi=220)
        payload = buffer.getvalue()
        if len(payload) < 1000:
            raise RuntimeError("Unexpectedly empty figure: " + stem)
        if ext == "pdf" and not payload.startswith(b"%PDF-"):
            raise RuntimeError("Invalid PDF figure: " + stem)
        destination = FIG / (stem + "." + ext)
        temporary = FIG / (stem + ".pending." + ext)
        temporary.write_bytes(payload)
        temporary.replace(destination)
    plt.close(fig)


# Supplementary Figure S1: the exact 2021 classical discrimination ridge.
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.65), constrained_layout=True)
axs[0].plot(ridge_w, (1-ridge_w)/(ridge_w*base_beta), color=colors[0])
axs[0].set(xlabel="Weight w", ylabel="Speed v (degrees/s)")
axs[0].set_yscale("log")
axs[0].scatter([base_w], [base_v], s=25, color=colors[1], zorder=3)
axs[0].set_title("A   Equivalent speeds", loc="left", fontweight="bold", fontsize=9)
axs[1].plot(ridge_w, base_s*np.sqrt(ridge_w), color=colors[0])
axs[1].scatter([base_w], [base_wm], s=25, color=colors[1], zorder=3)
axs[1].set(xlabel="Weight w", ylabel="Measurement coefficient $w_m$")
axs[1].set_title("B   Equivalent noise values", loc="left", fontweight="bold", fontsize=9)
axs[2].plot(ridge_w, ridge_ll-ridge_ll[0], color=colors[0])
axs[2].set(xlabel="Weight w", ylabel="Log-likelihood change", ylim=(-1e-9, 1e-9))
axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axs[2].set_title("C   Constant likelihood", loc="left", fontweight="bold", fontsize=9)
save_current_figure(fig, "figureS1_choice_ridge")

# Supplementary Figure S2: equal means alone do not give equal distributions.
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
axs[0].plot(cl, cm1+c_bias, "-", color=colors[0], label="w = 0.8")
axs[0].plot(cl, cm2+cbias2, "o", color=colors[1], mfc="white", label="w = 0.5")
axs[0].set(xlabel="Distance (degrees)", ylabel="Mean response (s)")
axs[0].legend(frameon=False)
axs[0].set_title("A   Bias compensation preserves means", loc="left", fontweight="bold", fontsize=9)
axs[1].plot(cl, csd1, "-o", color=colors[0], label="w = 0.8")
axs[1].plot(cl, csd2, "-s", color=colors[1], label="w = 0.5")
axs[1].set(xlabel="Distance (degrees)", ylabel="Response standard deviation (s)")
axs[1].legend(frameon=False)
axs[1].set_title("B   Noise retains information about w", loc="left", fontweight="bold", fontsize=9)
save_current_figure(fig, "figureS2_mean_vs_likelihood")


report = {"source": "Existing synthetic_ridge.csv and the original deterministic Figure S2 equations", "ridge_rows": int(len(ridge)), "ridge_loglikelihood_range": float(np.ptp(ridge_ll)), "figures": ["figureS1_choice_ridge", "figureS2_mean_vs_likelihood"], "new_random_draws": 0, "new_fits": 0}
(ROOT / "results" / "saved_illustration_redraw.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
