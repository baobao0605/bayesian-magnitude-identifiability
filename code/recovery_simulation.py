#!/usr/bin/env python3
"""Fixed-design finite-sample recovery for the calibrated kappa task bridge.

This controlled simulation uses additive reproduction noise. It is not a
replication of the production-noise model fitted by Chen et al. (2024).
Run from any directory; all defaults are relative to this project.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, logit, ndtr

ROOT = Path(__file__).resolve().parents[1]
NORMAL_95 = 1.959963984540054
LOG_2PI = math.log(2.0 * math.pi)
PARAMETERS = ("b", "s", "w", "v", "w_m")



def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty results table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def design_arrays(config: dict) -> dict:
    choice = config["choice"]
    dt, dl = np.meshgrid(choice["delta_t_seconds"], choice["delta_distance_degrees"], indexing="ij")
    dt, dl = dt.ravel(), dl.ravel()
    t1 = choice["time_midpoint_seconds"] - dt / 2
    t2 = choice["time_midpoint_seconds"] + dt / 2
    ell = np.repeat(config["reproduction"]["distances_degrees"], config["reproduction"]["trials_per_distance"])
    return {"dt": dt, "dl": dl, "D": np.sqrt(t1*t1+t2*t2),
            "t1": t1, "t2": t2, "ell": ell, "ell_c": ell-ell.mean(),
            "choice_n": choice["trials_per_cell"]}


def truth_parameters(config: dict) -> dict:
    row = config["truth"]
    w, v, wm, g = row["w"], row["v_degrees_per_second"], row["w_m"], row["g"]
    return {"w": w, "v": v, "w_m": wm, "g": g,
            "b": (1-w)/(w*v), "s": wm/math.sqrt(w),
            "q": g*(1-w)/v,
            "alpha": g*w*config["reproduction"]["duration_seconds"]+row["B_seconds"]}


def transformed_bounds(config: dict, noisy: bool) -> list[tuple[float, float]]:
    domain = config["estimation"]["bounds"]
    bounds = [tuple(np.log(domain["b"])), tuple(np.log(domain["s"])), tuple(domain["logit_w"])]
    if noisy:
        bounds.append(tuple(np.log(domain["g"])))
    return bounds


def unpack(theta: np.ndarray, noisy: bool, known_gain: float = 1.0) -> dict:
    b, s, w = math.exp(theta[0]), math.exp(theta[1]), float(expit(theta[2]))
    g = math.exp(theta[3]) if noisy else known_gain
    return {"b": b, "s": s, "w": w, "g": g,
            "v": math.exp(-theta[0]-theta[2]), "w_m": s*math.sqrt(w), "q": g*b*w}


def choice_objective(theta_bs: np.ndarray, counts: np.ndarray, design: dict) -> tuple[float, np.ndarray]:
    b, s = np.exp(theta_bs)
    z = (design["dt"]+b*design["dl"])/(s*design["D"])
    log_p, log_1mp = log_ndtr(z), log_ndtr(-z)
    log_phi = -.5*z*z-.5*LOG_2PI
    n = design["choice_n"]
    objective = -np.sum(counts*log_p+(n-counts)*log_1mp)
    score_z = (n-counts)*np.exp(log_phi-log_1mp)-counts*np.exp(log_phi-log_p)
    derivatives = np.column_stack((b*design["dl"]/(s*design["D"]), -z))
    return float(objective), derivatives.T @ score_z


def joint_objective(theta: np.ndarray, counts: np.ndarray, responses: np.ndarray,
                    calibrations: np.ndarray, design: dict, config: dict, noisy: bool) -> tuple[float, np.ndarray]:
    pars = unpack(theta, noisy, config["gain_calibration"]["known_gain"])
    value, grad_bs = choice_objective(theta[:2], counts, design)
    gradient = np.zeros(len(theta))
    gradient[:2] = grad_bs
    sigma = config["reproduction"]["known_additive_sd_seconds"]
    y_c = responses-responses.mean()
    residual = pars["q"]*design["ell_c"]-y_c
    value += .5*np.sum(residual*residual)/(sigma*sigma)
    value += len(responses)*(math.log(sigma)+.5*LOG_2PI)
    dq = pars["q"]*np.array([1., 0., 1.-pars["w"]]+([1.] if noisy else []))
    gradient += float(residual @ design["ell_c"])/(sigma*sigma)*dq
    if noisy:
        cal_sigma = config["gain_calibration"]["known_observation_sd"]
        value += .5*np.sum((calibrations-pars["g"])**2)/(cal_sigma*cal_sigma)
        value += len(calibrations)*(math.log(cal_sigma)+.5*LOG_2PI)
        gradient[3] += (len(calibrations)*pars["g"]-calibrations.sum())*pars["g"]/(cal_sigma*cal_sigma)
    return float(value), gradient


def expected_information(theta: np.ndarray, design: dict, config: dict, noisy: bool) -> np.ndarray:
    pars = unpack(theta, noisy, config["gain_calibration"]["known_gain"])
    b, s, w, g, q = (pars[k] for k in ("b", "s", "w", "g", "q"))
    z = (design["dt"]+b*design["dl"])/(s*design["D"])
    weight = design["choice_n"]*np.exp(-z*z-LOG_2PI-log_ndtr(z)-log_ndtr(-z))
    derivative = np.zeros((len(z), len(theta)))
    derivative[:, 0] = b*design["dl"]/(s*design["D"])
    derivative[:, 1] = -z
    info = derivative.T @ (weight[:, None]*derivative)
    sigma = config["reproduction"]["known_additive_sd_seconds"]
    dq = q*np.array([1., 0., 1.-w]+([1.] if noisy else []))
    info += float(design["ell_c"] @ design["ell_c"])/(sigma*sigma)*np.outer(dq, dq)
    if noisy:
        calibration = config["gain_calibration"]
        info[3, 3] += calibration["independent_observations"]*g*g/(calibration["known_observation_sd"]**2)
    return (info+info.T)/2


def projected_score(theta: np.ndarray, gradient: np.ndarray, bounds: list[tuple[float, float]]) -> float:
    score = gradient.copy()
    for j, (lower, upper) in enumerate(bounds):
        if theta[j]-lower <= 1e-6 and gradient[j] > 0:
            score[j] = 0.
        if upper-theta[j] <= 1e-6 and gradient[j] < 0:
            score[j] = 0.
    return float(np.max(np.abs(score)))


def optimize_choice(counts: np.ndarray, design: dict, config: dict):
    bounds = transformed_bounds(config, False)[:2]
    result = minimize(choice_objective, np.log([1., .25]), args=(counts, design),
                      jac=True, method="L-BFGS-B", bounds=bounds,
                      options={k:config["estimation"][k] for k in ("ftol", "gtol", "maxiter")})
    score = projected_score(result.x, result.jac, bounds)
    return result, bool(np.isfinite(result.fun) and (result.success or score <= config["estimation"]["max_absolute_score_for_numerical_success"]))


def fit_joint(rep: int, counts: np.ndarray, responses: np.ndarray, calibrations: np.ndarray,
              choice_result, choice_success: bool, design: dict, config: dict, noisy: bool):
    regime = "noisy_gain" if noisy else "known_gain"
    bounds = transformed_bounds(config, noisy)
    b0, s0 = np.exp(choice_result.x)
    q0 = float((responses-responses.mean()) @ design["ell_c"]/(design["ell_c"] @ design["ell_c"]))
    unconstrained_g = float(calibrations.mean()) if noisy else config["gain_calibration"]["known_gain"]
    free_w = q0/(unconstrained_g*b0) if unconstrained_g != 0 else math.nan
    valid_inverse = bool(choice_success and np.isfinite(free_w) and b0 > 0 and s0 > 0 and q0 > 0 and unconstrained_g > 0 and 0 < free_w < 1)
    # Clipping below is solely an optimizer starting point; it never defines an estimate.
    g0 = float(np.clip(unconstrained_g, config["estimation"]["bounds"]["g"][0]+.001,
                       config["estimation"]["bounds"]["g"][1]-.001)) if noisy else unconstrained_g
    init_w = float(np.clip(q0/(g0*b0), 1e-4, 1-1e-4))
    initial_weights = [init_w, .3, .8]
    candidates, optimization_rows = [], []
    for start, weight in enumerate(initial_weights):
        theta0 = np.array([math.log(b0), math.log(s0), float(logit(weight))]+([math.log(g0)] if noisy else []))
        result = minimize(joint_objective, theta0, args=(counts, responses, calibrations, design, config, noisy),
                          jac=True, method="L-BFGS-B", bounds=bounds,
                          options={k:config["estimation"][k] for k in ("ftol", "gtol", "maxiter")})
        score = projected_score(result.x, result.jac, bounds)
        finite = bool(np.isfinite(result.fun) and np.all(np.isfinite(result.x)))
        optimization_rows.append({"replicate":rep,"regime":regime,"start":start,"scipy_success":bool(result.success),
                                  "status":int(result.status),"message":str(result.message),"iterations":int(result.nit),
                                  "objective":float(result.fun),"absolute_score_max":float(np.max(np.abs(result.jac))),
                                  "projected_score_max":score,"finite":finite})
        if finite:
            candidates.append((result, score, start))
    if not candidates:
        raise RuntimeError(f"All fits nonfinite at replicate {rep}, {regime}; simulation halted without dropping this replicate")
    result, score, selected = min(candidates, key=lambda item:float(item[0].fun))
    success = bool(result.success or score <= config["estimation"]["max_absolute_score_for_numerical_success"])
    tol = config["estimation"]["near_bound_tolerance_transformed"]
    active = [j for j,(lower,upper) in enumerate(bounds) if min(result.x[j]-lower,upper-result.x[j])<=tol]
    pars = unpack(result.x, noisy, config["gain_calibration"]["known_gain"])
    alpha = float(responses.mean()-pars["q"]*design["ell"].mean())
    row = {"replicate":rep,"regime":regime,**pars,"alpha":alpha,
           "B":alpha-pars["g"]*pars["w"]*config["reproduction"]["duration_seconds"],
           "fit_success":success,"scipy_success":bool(result.success),"selected_start":selected,
           "objective":float(result.fun),"projected_score_max":score,"near_boundary":bool(active),
           "active_coordinates":";".join(map(str,active)),"unconstrained_w":float(free_w),
           "unconstrained_gain":unconstrained_g,"unconstrained_q":q0,"valid_unconstrained_inverse":valid_inverse,
           "choice_fit_success":choice_success}
    return result.x, row, optimization_rows


def intervals(theta: np.ndarray, fit: dict, truth: dict, design: dict, config: dict, noisy: bool, ignore_gain: bool=False) -> list[dict]:
    procedure = "noisy_gain_ignored" if ignore_gain else fit["regime"]
    info = expected_information(theta, design, config, noisy)
    use_theta = theta[:3] if ignore_gain else theta
    if ignore_gain:
        info = info[:3,:3]
    eigenvalues = np.linalg.eigvalsh(info)
    condition = float(np.linalg.cond(info))
    base_valid = bool(fit["fit_success"] and not fit["near_boundary"] and
                      eigenvalues[0]>0 and condition<config["estimation"]["max_information_condition_number"])
    covariance = np.linalg.inv(info) if base_valid else None
    names = PARAMETERS+(("g",) if noisy and not ignore_gain else ())
    rows = []
    w = fit["w"]
    for name in names:
        derivative = np.zeros(len(use_theta))
        if name == "b":
            center, derivative[0], inverse = theta[0], 1., math.exp
        elif name == "s":
            center, derivative[1], inverse = theta[1], 1., math.exp
        elif name == "w":
            center, derivative[2], inverse = theta[2], 1., expit
        elif name == "v":
            center, inverse = -theta[0]-theta[2], math.exp
            derivative[0] = derivative[2] = -1.
        elif name == "w_m":
            center, inverse = theta[1]+.5*math.log(w), math.exp
            derivative[1], derivative[2] = 1., .5*(1.-w)
        else:
            center, derivative[3], inverse = theta[3], 1., math.exp
        lower = upper = variance = math.nan
        valid = base_valid
        if valid:
            variance = float(derivative @ covariance @ derivative)
            valid = bool(np.isfinite(variance) and variance>=0)
        if valid:
            margin = NORMAL_95*math.sqrt(variance)
            try:
                lower, upper = float(inverse(center-margin)), float(inverse(center+margin))
                valid = bool(np.isfinite(lower) and np.isfinite(upper))
            except OverflowError:
                valid = False
        rows.append({"replicate":fit["replicate"],"procedure":procedure,"parameter":name,
                     "truth":truth[name],"estimate":fit[name],"lower":lower,"upper":upper,
                     "valid_interval":valid,"covers_truth":bool(valid and lower<=truth[name]<=upper),
                     "width":upper-lower if valid else math.nan,
                     "information_condition_number":condition,"information_min_eigenvalue":float(eigenvalues[0])})
    return rows


def numerical_checks(design: dict, config: dict, truth: dict) -> dict:
    theta = np.array([math.log(truth["b"]),math.log(truth["s"]),float(logit(truth["w"])),math.log(truth["g"])])
    z = (design["dt"]+truth["b"]*design["dl"])/(truth["s"]*design["D"])
    expected_counts = design["choice_n"]*ndtr(z)
    responses = truth["alpha"]+truth["q"]*design["ell"]
    calibrations = np.full(config["gain_calibration"]["independent_observations"],truth["g"])
    offset_theta = theta+np.array([.07,-.04,.12,.03])
    args = (expected_counts,responses,calibrations,design,config,True)
    h = 1e-5
    numerical_grad = np.array([(joint_objective(offset_theta+np.eye(4)[j]*h,*args)[0]-joint_objective(offset_theta-np.eye(4)[j]*h,*args)[0])/(2*h) for j in range(4)])
    analytic_grad = joint_objective(offset_theta,*args)[1]
    numerical_hessian = np.column_stack([(joint_objective(theta+np.eye(4)[j]*h,*args)[1]-joint_objective(theta-np.eye(4)[j]*h,*args)[1])/(2*h) for j in range(4)])
    information = expected_information(theta,design,config,True)
    gradient_relative_error = float(np.max(np.abs(numerical_grad-analytic_grad))/max(1.,np.max(np.abs(analytic_grad))))
    information_relative_error = float(np.max(np.abs(numerical_hessian-information))/np.max(np.abs(information)))
    if gradient_relative_error>1e-6 or information_relative_error>1e-6:
        raise RuntimeError("Analytical score/information failed deterministic finite-difference verification")
    return {"method":"deterministic central differences at fixed test parameters; no simulation data used",
            "step":h,"gradient_max_relative_error":gradient_relative_error,
            "expected_information_max_relative_error":information_relative_error,
            "required_relative_tolerance":1e-6,"passed":True}


def summarize(fits: list[dict], ci_rows: list[dict], truth: dict, repetitions: int) -> dict:
    output = {}
    for procedure in ("known_gain","noisy_gain","noisy_gain_ignored"):
        regime = "noisy_gain" if procedure == "noisy_gain_ignored" else procedure
        selected = [r for r in fits if r["regime"]==regime]
        successful = [r for r in selected if r["fit_success"]]
        metrics = {}
        names = PARAMETERS+(("g",) if procedure=="noisy_gain" else ())
        for name in names:
            errors = np.array([r[name]-truth[name] for r in successful])
            rows = [r for r in ci_rows if r["procedure"]==procedure and r["parameter"]==name]
            valid = [r for r in rows if r["valid_interval"]]
            covered = sum(r["covers_truth"] for r in rows)
            fraction = covered/repetitions
            conditional = covered/len(valid) if valid else None
            widths = np.array([r["width"] for r in valid])
            metrics[name] = {"truth":truth[name],"point_estimate_count":len(errors),
                             "bias":float(errors.mean()) if len(errors) else None,
                             "bias_mcse":float(errors.std(ddof=1)/math.sqrt(len(errors))) if len(errors)>1 else None,
                             "rmse":float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                             "median_absolute_error":float(np.median(np.abs(errors))) if len(errors) else None,
                             "valid_interval_count":len(valid),"covered_count":int(covered),
                             "success_and_coverage_all_replicates":fraction,
                             "coverage_all_mcse":math.sqrt(fraction*(1-fraction)/repetitions),
                             "coverage_given_valid":conditional,
                             "coverage_given_valid_mcse":math.sqrt(conditional*(1-conditional)/len(valid)) if valid else None,
                             "median_interval_width":float(np.median(widths)) if len(widths) else None,
                             "mean_interval_width":float(widths.mean()) if len(widths) else None}
        output[procedure] = {"replicates":repetitions,"fit_successes":len(successful),
                             "optimizer_failures":repetitions-len(successful),
                             "near_boundary_count":sum(r["near_boundary"] for r in selected),
                             "valid_unconstrained_inverse_count":sum(r["valid_unconstrained_inverse"] for r in selected),
                             "parameters":metrics}
    return output


def make_figure(fits: list[dict], summary: dict, destination: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR",tempfile.mkdtemp(prefix="kappa-recovery-mpl-"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})
    fig, axes = plt.subplots(1,3,figsize=(10.1,3.25),constrained_layout=True)
    colors = ["#1B6B9E","#C65B32","#8A6BA3"]
    for j,regime in enumerate(("known_gain","noisy_gain")):
        values = np.array([r["v"] for r in fits if r["regime"]==regime and r["fit_success"]])
        # Deterministic spacing is used only to display overlapping fitted values.
        jitter = .10*np.sin(np.arange(len(values))*2.399963229728653)
        axes[0].scatter(j+jitter,values,s=9,alpha=.38,color=colors[j],edgecolors="none")
        axes[0].plot([j-.20,j+.20],[np.median(values)]*2,color=colors[j],linewidth=2.5)
    axes[0].axhline(.5,color=".35",linestyle="--",linewidth=1)
    axes[0].set(xticks=[0,1],xticklabels=["Known gain","Noisy gain"],ylabel="Fitted speed (degree/s)",title="A  Finite-sample estimates")
    procedures = ("known_gain","noisy_gain","noisy_gain_ignored")
    labels = ["Known","Noisy:\npropagated","Noisy:\nignored"]
    rates = [summary[p]["parameters"]["v"]["success_and_coverage_all_replicates"] for p in procedures]
    errors = [1.96*summary[p]["parameters"]["v"]["coverage_all_mcse"] for p in procedures]
    widths = [summary[p]["parameters"]["v"]["median_interval_width"] for p in procedures]
    axes[1].bar(range(3),rates,color=colors,width=.65,yerr=errors,capsize=3)
    axes[1].axhline(.95,color=".35",linestyle="--",linewidth=1)
    axes[1].set(xticks=range(3),xticklabels=labels,ylim=(0,1.05),ylabel="Covered / all 200 replicates",title="B  Speed interval coverage")
    axes[2].bar(range(3),widths,color=colors,width=.65)
    axes[2].set(xticks=range(3),xticklabels=labels,ylabel="Median 95% width (degree/s)",title="C  Speed interval width")
    for ax in axes:
        ax.tick_params(axis="x",length=0)
        ax.grid(axis="y",alpha=.13)
        ax.set_axisbelow(True)
    fig.savefig(destination/"figureS3_recovery.pdf",metadata={"Title":"Controlled finite-sample recovery of a calibrated task bridge","Creator":"Matplotlib from recovery_simulation.py"})
    fig.savefig(destination/"figureS3_recovery.png",dpi=240)
    plt.close(fig)


def report_text(summary: dict, config: dict, checks: dict) -> str:
    lines = ["# 校准跨任务连接：固定设计的有限样本恢复结果","",
             "本报告对应预先写定的200次配对重复、一个参数点。再现噪声为已知加性Gaussian噪声，属于识别桥接的受控模拟，不是Chen 2024生产噪声的复现。任务共享与独立gain校准在生成时成立；这里没有人类数据、模型选择或计算机制的经验验证。","",
             "## 实际设计与估计","",
             "真值为 w=.65、v=.5 degree/s、w_m=.2、g=1。每次生成600个二项选择、100个再现反应；有噪声gain校准另含16个N(g,.6²)观测，平均标准误=.15。两情形共享同一行为样本。完整似然重新拟合，截距解析剖面化，数值边界及多起点规则见冻结设计。","",
             "95%区间采用联合MLE处的期望Fisher信息，在log/logit尺度建立再逆变换。近边界或信息不合格时不报告常规Wald区间。下表覆盖率一律以全部200次为分母，未得到有效区间的重复不计为覆盖；括号是二项Monte Carlo标准误，不是新的行为测量误差。","",
             "## 点估计与区间","",
             "| 情形 / 区间处理 | 参数 | 偏差 | RMSE | 覆盖 / 200 | 覆盖率（MCSE） | 有效区间数 | 中位宽度 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    labels = {"known_gain":"gain准确已知","noisy_gain":"gain有噪声，传播误差","noisy_gain_ignored":"同一有噪声拟合，忽略gain误差"}
    for procedure in labels:
        for parameter in PARAMETERS:
            m=summary[procedure]["parameters"][parameter]
            lines.append(f"| {labels[procedure]} | {parameter} | {m['bias']:.5f} | {m['rmse']:.5f} | {m['covered_count']} | {m['success_and_coverage_all_replicates']:.3f} ({m['coverage_all_mcse']:.3f}) | {m['valid_interval_count']} | {m['median_interval_width']:.5f} |")
    lines.extend(["","## 失败、边界与逆解可行性","",
                  "| 情形 | 优化成功 / 200 | 近边界 | 无约束桥接逆解有效 / 200 |",
                  "|---|---:|---:|---:|"])
    for procedure in ("known_gain","noisy_gain"):
        r=summary[procedure]
        lines.append(f"| {labels[procedure]} | {r['fit_successes']} | {r['near_boundary_count']} | {r['valid_unconstrained_inverse_count']} |")
    lines.extend(["","不可行的约化逆解未被当作有效心理参数。正式点估计由联合约束似然获得；近边界点仍保留在点误差统计中，区间是否可用另列。CSV保留每次起点、状态、目标值、梯度、逆解标志和区间。JSON还给出有效区间条件覆盖率、平均宽度和偏差Monte Carlo标准误。","",
                  "## 实现核验","",
                  f"正式生成数据前，用确定性例子核验解析梯度与期望信息：中心差分步长{checks['step']:g}，梯度最大相对差{checks['gradient_max_relative_error']:.3g}，信息矩阵最大相对差{checks['expected_information_max_relative_error']:.3g}；均低于预定1e-6容差。该检查不是参数恢复结果。","",
                  "## 论证含义与范围","",
                  "条件性逆式决定可识别对象；有限样本误差和校准精度决定其估计质量。将gain当成已知会删除实际存在的校准不确定性。完整传播的覆盖和宽度，应与忽略该误差的对照一起阅读。Wald区间本身是近似推断，200次重复的覆盖估计仍有Monte Carlo误差。","",
                  "本项结果局限于一个固定参数点、一个设计、规定的加性再现噪声与正确共享假设。它不证明其他参数区间、原始生产噪声、失配校准或跨任务不共享时也有同样恢复性能；不构成心理速度或某种实现的经验支持。","",
                  "![跨任务恢复](../figures/figureS3_recovery.png)","",
                  "图：A为两种数据情形的速度估计，虚线为真值；B为全200次的区间覆盖，误差棒为±1.96 Monte Carlo标准误，虚线为.95；C为有效区间宽度中位数。Noisy: ignored 使用同一联合估计但在区间计算中将gain视为固定。","",
                  "## 复现","",
                  "`python3 code/recovery_simulation.py --rerun-simulation --overwrite` 使用冻结设计重做这项模拟。只重绘已有结果可用 `--plot-only`。也可用 `--output-dir` 和 `--figure-dir` 指向新目录以保留交付结果。代码仅使用Python标准库、NumPy、SciPy和Matplotlib；运行版本记录在JSON中。",""
                  ])
    return "\n".join(lines)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results")
    parser.add_argument("--figure-dir",type=Path,default=ROOT/"figures")
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--rerun-simulation",action="store_true",help="Explicitly regenerate and refit the fixed 200-replicate study")
    parser.add_argument("--plot-only",action="store_true")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    args.figure_dir.mkdir(parents=True,exist_ok=True)
    design_path=ROOT/"results/recovery_design.json"
    config=json.loads(design_path.read_text())
    summary_path=args.output_dir/"recovery_summary.json"
    if args.plot_only:
        saved=json.loads(summary_path.read_text())
        fits=json.loads((args.output_dir/"recovery_fits.json").read_text())
        make_figure(fits,saved["procedures"],args.figure_dir)
        return
    if not args.rerun_simulation:
        parser.error("No new data or fits were requested. Use --plot-only to redraw saved results, or explicitly --rerun-simulation to repeat the fixed study.")
    if summary_path.exists() and not args.overwrite:
        raise RuntimeError("Existing recovery results retained; use a new --output-dir or explicit --overwrite")
    design=design_arrays(config)
    truth=truth_parameters(config)
    checks=numerical_checks(design,config,truth)
    write_json(args.output_dir/"recovery_numerical_checks.json",checks)
    n=config["replicates"]
    child_seeds=np.random.SeedSequence(config["seed"]).spawn(n)
    z=(design["dt"]+truth["b"]*design["dl"])/(truth["s"]*design["D"])
    probability=ndtr(z)
    responses_mean=truth["alpha"]+truth["q"]*design["ell"]
    choices=np.empty((n,len(z),design["choice_n"]),dtype=np.uint8)
    reproductions=np.empty((n,len(design["ell"])))
    calibrations=np.empty((n,config["gain_calibration"]["independent_observations"]))
    for i, seed in enumerate(child_seeds):
        a,b,c=seed.spawn(3)
        choices[i]=np.random.default_rng(a).binomial(1,probability[:,None],size=choices[i].shape)
        reproductions[i]=np.random.default_rng(b).normal(responses_mean,config["reproduction"]["known_additive_sd_seconds"])
        calibrations[i]=np.random.default_rng(c).normal(truth["g"],config["gain_calibration"]["known_observation_sd"],size=calibrations.shape[1])
    data_path=args.output_dir/"recovery_simulated_data.npz"
    np.savez_compressed(data_path,choices=choices,reproductions=reproductions,calibrations=calibrations,
                        delta_t=design["dt"],delta_distance=design["dl"],choice_t1=design["t1"],choice_t2=design["t2"],
                        reproduction_distance=design["ell"],true_choice_probability=probability)
    fits,ci_rows,optimizer_rows=[],[],[]
    choice_ridge_differences=[]
    for i in range(n):
        counts=choices[i].sum(axis=1)
        choice_result,choice_success=optimize_choice(counts,design,config)
        b_hat,s_hat=np.exp(choice_result.x)
        base_ll=choice_objective(choice_result.x,counts,design)[0]
        ridge_ll=[]
        for weight in np.linspace(.1,.9,41):
            speed=(1-weight)/(weight*b_hat)
            measurement=s_hat*math.sqrt(weight)
            recovered=np.array([math.log((1-weight)/(weight*speed)),math.log(measurement/math.sqrt(weight))])
            ridge_ll.append(choice_objective(recovered,counts,design)[0])
        choice_ridge_differences.append(float(np.max(np.abs(np.array(ridge_ll)-base_ll))))
        for noisy in (False,True):
            theta,fit,trace=fit_joint(i,counts,reproductions[i],calibrations[i],choice_result,choice_success,design,config,noisy)
            fits.append(fit);optimizer_rows.extend(trace)
            ci_rows.extend(intervals(theta,fit,truth,design,config,noisy))
            if noisy:
                ci_rows.extend(intervals(theta,fit,truth,design,config,noisy,ignore_gain=True))
        if (i+1)%25==0:
            print(f"Completed {i+1}/{n} paired replicates",flush=True)
    procedures=summarize(fits,ci_rows,truth,n)
    import matplotlib
    result={"design_id":config["design_id"],
            "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__},
            "seed":config["seed"],"replicates":n,"truth":truth,
            "scope":{"one_parameter_point":True,"paired_behavioral_samples":True,"additive_reproduction_noise":True,
                     "human_data":False,"model_recovery":False,"parameter_recovery":True,"design_changed_after_results":False},
            "choice_only_ridge_loglikelihood_max_absolute_error":max(choice_ridge_differences),
            "procedures":procedures}
    write_json(summary_path,result)
    write_json(args.output_dir/"recovery_fits.json",fits)
    write_csv(args.output_dir/"recovery_estimates.csv",fits)
    write_csv(args.output_dir/"recovery_intervals.csv",ci_rows)
    write_csv(args.output_dir/"recovery_optimizer.csv",optimizer_rows)
    (args.output_dir/"recovery_report_CN.md").write_text(report_text(procedures,config,checks))
    make_figure(fits,procedures,args.figure_dir)
    print(json.dumps({"status":"completed","paired_replicates":n,
                      "summary":str(summary_path),
                      "optimizer_failures":{p:procedures[p]["optimizer_failures"] for p in ("known_gain","noisy_gain")},
                      "near_boundary_count":{p:procedures[p]["near_boundary_count"] for p in ("known_gain","noisy_gain")}},indent=2))


if __name__=="__main__":
    main()
