#!/usr/bin/env python3
"""Reproduce algebraic identities and synthetic illustrations in the manuscript.

No participant data are loaded or fitted. Numerical comparisons complement the
analytic proofs; they are not estimates of human performance.
"""

import argparse

def regenerate_original_checks():
    from pathlib import Path
    import json
    from io import BytesIO
    import os
    import platform
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1]/"build"/"matplotlib"))
    import numpy as np
    import scipy
    from scipy.special import ndtr, log_ndtr
    from scipy.stats import norm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    ROOT = Path(__file__).resolve().parents[1]
    FIG = ROOT / "figures"
    OUT = ROOT / "results"
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261026)
    
    
    def lcm_staged(t, ell, w, v, t0):
        latent = w * np.log1p(t / t0) + (1 - w) * np.log1p(ell / (v * t0))
        return t0 * np.expm1(latent)
    
    
    def lcm_power(t, ell, w, v, t0):
        return t0 * ((1 + t / t0) ** w * (1 + ell / (v * t0)) ** (1 - w) - 1)
    
    
    def classical_choice(t1, t2, ell1, ell2, w, v, wm):
        numerator = w * (t2 - t1) + (1 - w) * (ell2 - ell1) / v
        denominator = np.sqrt(w * wm ** 2 * (t1 ** 2 + t2 ** 2))
        return ndtr(numerator / denominator)
    
    
    def reduced_choice(t1, t2, ell1, ell2, beta, s):
        return ndtr(((t2 - t1) + beta * (ell2 - ell1)) /
                    (s * np.sqrt(t1 ** 2 + t2 ** 2)))
    
    
    def lcm_choice(t1, t2, ell1, ell2, w, v, t0, sigma):
        mu1 = w * np.log1p(t1 / t0) + (1 - w) * np.log1p(ell1 / (v * t0))
        mu2 = w * np.log1p(t2 / t0) + (1 - w) * np.log1p(ell2 / (v * t0))
        return ndtr((mu2 - mu1) / (np.sqrt(2 * w) * sigma))
    
    
    def direct_lcm_choice(t1, t2, ell1, ell2, gamma, a, t0, s):
        score = np.log((t0 + t2) / (t0 + t1)) + gamma * np.log((a + ell2) / (a + ell1))
        return ndtr(score / (np.sqrt(2) * s))
    
    
    # Identity checks cover interior parameter values; values at limiting boundaries
    # are treated analytically in the text rather than by division by zero.
    n = 10000
    t = rng.uniform(.1, 4, n)
    ell = rng.uniform(0, 32, n)
    w = rng.uniform(.05, .95, n)
    v = np.exp(rng.uniform(np.log(.03), np.log(5), n))
    t0 = np.exp(rng.uniform(np.log(.05), np.log(30), n))
    wp = rng.uniform(.03, .4, n)
    bias = rng.normal(0, .2, n)
    m1 = lcm_staged(t, ell, w, v, t0)
    m2 = lcm_power(t, ell, w, v, t0)
    scaled_error = np.max(np.abs(m1 - m2) / np.maximum(1, np.abs(m1)))
    y = m1 + bias + wp * m1 * rng.normal(size=n)
    logp1 = norm.logpdf(y, loc=m1 + bias, scale=wp * m1)
    logp2 = norm.logpdf(y, loc=m2 + bias, scale=wp * m2)
    
    t1 = rng.uniform(.3, 2, n)
    t2 = rng.uniform(.3, 2, n)
    ell1 = rng.uniform(0, 4, n)
    ell2 = rng.uniform(0, 4, n)
    wm = rng.uniform(.05, .5, n)
    beta = (1 - w) / (w * v)
    s = wm / np.sqrt(w)
    p1 = classical_choice(t1, t2, ell1, ell2, w, v, wm)
    p2 = reduced_choice(t1, t2, ell1, ell2, beta, s)
    wstar = rng.uniform(.1, .9, n)
    vstar = (1 - wstar) / (wstar * beta)
    wmstar = s * np.sqrt(wstar)
    pstar = classical_choice(t1, t2, ell1, ell2, wstar, vstar, wmstar)
    sigma = rng.uniform(.05, .5, n)
    plog = lcm_choice(t1, t2, ell1, ell2, w, v, t0, sigma)
    ppower = direct_lcm_choice(t1, t2, ell1, ell2, (1 - w) / w,
                               v * t0, t0, sigma / np.sqrt(w))
    
    # Synthetic Bernoulli observations make the flat likelihood ridge visible.
    nsyn = 1500
    st1 = rng.uniform(.5, 1.5, nsyn)
    st2 = rng.uniform(.5, 1.5, nsyn)
    sl1 = rng.uniform(0, 2, nsyn)
    sl2 = rng.uniform(0, 2, nsyn)
    base_w, base_v, base_wm = .8, .25, .2
    base_beta = (1 - base_w) / (base_w * base_v)
    base_s = base_wm / np.sqrt(base_w)
    sp = reduced_choice(st1, st2, sl1, sl2, base_beta, base_s)
    sy = rng.binomial(1, sp)
    
    
    def binary_ll(wv):
        vv = (1 - wv) / (wv * base_beta)
        wmv = base_s * np.sqrt(wv)
        z = (wv * (st2 - st1) + (1 - wv) * (sl2 - sl1) / vv) / (
            np.sqrt(wv) * wmv * np.sqrt(st1 ** 2 + st2 ** 2))
        return float(np.sum(np.where(sy, log_ndtr(z), log_ndtr(-z))))
    
    
    ridge_w = np.linspace(.1, .9, 81)
    ridge_ll = np.array([binary_ll(q) for q in ridge_w])
    
    # Counterexample to transferring a mean ridge to the reproduction likelihood.
    ct = .8
    cl = np.array([0., 1., 2., 4., 8., 16., 25.])
    c_beta, c_wp, c_bias = .08, .15, .1
    cw1, cw2 = .8, .5
    cm1 = cw1 * ct + c_beta * cl
    cm2 = cw2 * ct + c_beta * cl
    cbias2 = c_bias + (cw1 - cw2) * ct
    csd1, csd2 = c_wp * cm1, c_wp * cm2
    
    results = {
        "purpose": "Algebraic verification and synthetic illustrations; no empirical fit",
        "seed": 20261026,
        "parameter_draws": n,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "lcm_reproduction_scaled_mean_error_max": float(scaled_error),
        "lcm_reproduction_logdensity_error_max": float(np.max(np.abs(logp1-logp2))),
        "classic_choice_probability_error_max": float(np.max(np.abs(p1-p2))),
        "classic_choice_ridge_probability_error_max": float(np.max(np.abs(p1-pstar))),
        "lcm_choice_probability_error_max": float(np.max(np.abs(plog-ppower))),
        "synthetic_trials": nsyn,
        "ridge_loglikelihood_range": float(np.ptp(ridge_ll)),
        "reproduction_mean_ridge_mean_error_max": float(np.max(np.abs((cm1+c_bias)-(cm2+cbias2)))),
        "reproduction_mean_ridge_sd_difference_max_seconds": float(np.max(np.abs(csd1-csd2)))
    }
    assert scaled_error < 1e-12
    assert results["lcm_reproduction_logdensity_error_max"] < 1e-9
    assert results["classic_choice_probability_error_max"] < 1e-12
    assert results["classic_choice_ridge_probability_error_max"] < 1e-12
    assert results["lcm_choice_probability_error_max"] < 1e-12
    assert results["ridge_loglikelihood_range"] < 1e-9
    
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
    
    (OUT / "verification_results.json").write_text(json.dumps(results, indent=2)+"\n")
    np.savetxt(OUT / "synthetic_ridge.csv",
               np.column_stack([ridge_w, (1-ridge_w)/(ridge_w*base_beta),
                                base_s*np.sqrt(ridge_w), ridge_ll]),
               delimiter=",", header="weight,speed,measurement_coefficient,synthetic_loglikelihood", comments="")
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-illustration", action="store_true", help="Explicitly repeat the original seeded numerical checks and 1,500-trial illustration")
    args = parser.parse_args()
    if not args.rerun_illustration:
        parser.error("This command generates random values. To redraw supplied outputs without sampling, use code/plot_saved_illustrations.py. To repeat the original fixed checks, pass --rerun-illustration.")
    regenerate_original_checks()

if __name__ == "__main__":
    main()
