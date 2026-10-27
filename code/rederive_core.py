#!/usr/bin/env python3
"""Independent deterministic checks for the four mathematical results.

Uses only the Python standard library. Exact rational checks concern finite
instances of the analytic identities; floating-point checks concern the
displayed nonlinear expressions. Neither is a parameter-recovery study.
"""
from fractions import Fraction as F
from pathlib import Path
import itertools
import json
import math
import platform

HERE = Path(__file__).resolve().parents[1] / "results"
HERE.mkdir(exist_ok=True)


def maximum(values):
    return float(max(map(abs, values), default=0))


def cdf(z):
    return .5 * math.erfc(-z / math.sqrt(2))


def classical_choice(t1, t2, l1, l2, w, v, wm):
    num = w * (t2 - t1) + (1 - w) * (l2 - l1) / v
    den = math.sqrt(w) * wm * math.hypot(t1, t2)
    return cdf(num / den)


def log_choice(t1, t2, l1, l2, tau, a, gamma, s):
    z = (math.log((tau + t2) / (tau + t1))
         + gamma * math.log((a + l2) / (a + l1))) / (math.sqrt(2) * s)
    return cdf(z)


def power_mean(t, ell, tau, a, p, q, gain, bias):
    return gain * tau * ((1 + t / tau) ** p * (1 + ell / a) ** q - 1) + bias


exact = {}
V = [F(1, 5), F(2, 3), F(7, 4)]
T = [F(3, 10), F(4, 5), F(9, 4)]
W = [[tj / (tj + vi) for tj in T] for vi in V]
O = [[(1 - w) / w for w in row] for row in W]
exact["all_positive_weight_odds_minors"] = all(
    O[i][j] * O[k][l] == O[i][l] * O[k][j]
    for i, k, j, l in itertools.product(range(3), repeat=4))
scale = F(7, 3)
Vr = [scale * row[0] for row in O]
Tr = [scale * O[0][0] / q for q in O[0]]
exact["positive_factorization_reconstructs_all_cells"] = all(
    Vr[i] / Tr[j] == O[i][j] for i, j in itertools.product(range(3), repeat=2))

w, v, wm, gain = F(4, 5), F(1, 4), F(1, 5), F(23, 20)
b, s2 = (1 - w) / (w * v), wm * wm / w
beta = (1 - w) / v
beta_y = gain * beta
wr = beta_y / (gain * b)
vr = gain / beta_y - 1 / b
wm2r = s2 * wr
exact["calibrated_bridge_inverse"] = (wr == w and vr == v and wm2r == wm * wm)
ridge = []
for w_new in [F(1, 5), F(1, 2), F(9, 10)]:
    v_new = (1 - w_new) / (w_new * b)
    wm2_new = s2 * w_new
    gain_new = beta_y / (b * w_new)
    ridge.append((1 - w_new) / (w_new * v_new) == b and wm2_new / w_new == s2)
    ridge.append(gain_new * w_new == gain * w)
    ridge.append(gain_new * b * w_new == beta_y)
exact["choice_ridge_and_unknown_gain_mean_ridge"] = all(ridge)
exact["shared_context_line"] = all(
    wm * wm / q == wm * wm + wm * wm * v * ((1 - q) / (q * v))
    for q in [F(1, 4), F(3, 5), F(9, 10)])

t, beta, wp, w, bias = F(4, 5), F(2, 25), F(3, 20), F(4, 5), F(1, 10)
ells = [F(0), F(1), F(4), F(8)]
mus = [w * t + beta * ell + bias for ell in ells]
sds = [wp * (w * t + beta * ell) for ell in ells]
beta_r = (mus[-1] - mus[0]) / (ells[-1] - ells[0])
wp_r = (sds[-1] - sds[0]) / (beta_r * (ells[-1] - ells[0]))
w_r = (sds[0] / wp_r - beta_r * ells[0]) / t
bias_r = mus[0] - w_r * t - beta_r * ells[0]
exact["pre_bias_population_moment_inverse"] = (beta_r, wp_r, w_r, bias_r) == (beta, wp, w, bias)
w_new = F(1, 2)
bias_new = bias + (w - w_new) * t
exact["post_bias_full_gaussian_identity"] = all(
    (w * t + beta * ell + bias,
     (wp * (w * t + beta * ell + bias)) ** 2)
    == (w_new * t + beta * ell + bias_new,
        (wp * (w_new * t + beta * ell + bias_new)) ** 2)
    for ell in ells)
exact["pre_bias_sd_change"] = all(
    wp * ((w * t + beta * ell) - (w_new * t + beta * ell)) == F(9, 250)
    for ell in ells)

# A sparse two-duration design does not satisfy the rowwise design condition
# used to characterize the pre/post family intersection.
times = [F(1), F(2)]
distances = [F(1), F(3)]
pre = {"w": F(2, 5), "beta": F(1, 5), "wp": F(1, 10),
       "bias": [F(-3, 10), F(-7, 10)]}
post = {"w": F(1, 2), "beta": F(3, 20), "wp": F(1, 5),
        "bias": [F(-7, 20), F(-3, 4)]}
sparse_rows = []
for i, (tt, ell) in enumerate(zip(times, distances)):
    mp = pre["w"] * tt + pre["beta"] * ell
    mq = post["w"] * tt + post["beta"] * ell + post["bias"][i]
    mu_pre, sd_pre = mp + pre["bias"][i], pre["wp"] * mp
    mu_post, sd_post = mq, post["wp"] * mq
    sparse_rows.append({"t": str(tt), "ell": str(ell), "mean_pre": str(mu_pre),
                        "mean_post": str(mu_post), "sd_pre": str(sd_pre),
                        "sd_post": str(sd_post), "post_bias_over_t": str(post["bias"][i] / tt)})
exact["sparse_design_distributions_match"] = all(
    row["mean_pre"] == row["mean_post"] and row["sd_pre"] == row["sd_post"]
    for row in sparse_rows)
exact["sparse_design_bias_ratio_is_not_constant"] = post["bias"][0] / times[0] != post["bias"][1] / times[1]

float_checks = {}
grid = list(itertools.product([.5, .8, 1.2], [.5, .8, 1.2], [0., .1, .3], [0., .1, .3]))
old_choice = (.8, .25, .2)
b0, s0 = (1 - old_choice[0]) / (old_choice[0] * old_choice[1]), old_choice[2] / math.sqrt(old_choice[0])
w2 = .35
new_choice = (w2, (1 - w2) / (w2 * b0), s0 * math.sqrt(w2))
float_checks["classical_choice_probability_max_error"] = maximum(
    classical_choice(*x, *old_choice) - classical_choice(*x, *new_choice) for x in grid)

tau, a, p, q, gain = 1., .09, .83, .17, 1.2
tt = [.5, .8, 1.2, 2.]
ll = [0., .5, 2., 5.]
biases = [.1, .23, -.05, .3]
mu = [[power_mean(ti, lj, tau, a, p, q, gain, bi) for lj in ll] for ti, bi in zip(tt, biases)]
M = [[row[j] - row[0] for j in range(1, len(ll))] for row in mu]
float_checks["distance_contrast_minor_max_error"] = maximum(
    M[i][j] * M[k][l] - M[i][l] * M[k][j]
    for i, k in itertools.product(range(4), repeat=2)
    for j, l in itertools.product(range(3), repeat=2))
float_checks["temporal_contrast_ratio_max_error"] = maximum(
    M[i][j] / M[0][j] - ((tau + tt[i]) / (tau + tt[0])) ** p
    for i, j in itertools.product(range(1, 4), range(3)))

t1, t2, tau2 = .5, 1.2, 2.
L = lambda z: math.log((z + t2) / (z + t1))
p2 = p * L(tau) / L(tau2)
gain2 = gain * tau * (1 + t1 / tau) ** p / (tau2 * (1 + t1 / tau2) ** p2)
new_biases = [bi - gain * tau + gain2 * tau2 for bi in biases[:2]]
float_checks["free_scale_mean_path_max_error"] = maximum(
    power_mean(ti, ell, tau, a, p, q, gain, bi)
    - power_mean(ti, ell, tau2, a, p2, q, gain2, bn)
    for ti, bi, bn in zip([t1, t2], biases[:2], new_biases) for ell in ll)

gamma, scale = .3, .2
scale2 = scale * L(tau2) / L(tau)
gamma2 = gamma * scale2 / scale
float_checks["two_time_log_choice_path_max_error"] = maximum(
    log_choice(ta, tb, la, lb, tau, a, gamma, scale)
    - log_choice(ta, tb, la, lb, tau2, a, gamma2, scale2)
    for ta, tb in itertools.product([t1, t2], repeat=2)
    for la, lb in itertools.product(ll, repeat=2))

result = {
    "scope": "AI-assisted mathematical re-derivation with exact rational and deterministic numerical checks; not human peer review or recovery simulation",
    "python": platform.python_version(),
    "standard_library_only": True,
    "exact_rational_checks": exact,
    "all_exact_rational_checks_true": all(exact.values()),
    "floating_point_residuals": float_checks,
    "sparse_design_counterexample": sparse_rows,
    "free_scale_path": {"original_p_plus_q": p + q, "new_p_plus_q": p2 + q},
    "random_draws": 0,
    "fit_runs": 0,
    "parameter_recovery": False,
    "model_recovery": False,
}
output = HERE / "rederived_core_results.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
if not all(exact.values()):
    raise SystemExit("An exact rational check failed")
