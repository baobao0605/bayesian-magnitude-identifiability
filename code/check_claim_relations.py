#!/usr/bin/env python3
"""Check the manuscript's analytical relations.

All checks use symbolic identities or selected, deterministic parameter values.
There is no response sampling, optimization, parameter recovery, or model recovery.
"""
from pathlib import Path
import json
import platform
import numpy as np
import scipy
from scipy.stats import norm
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "results" / "claim_relation_checks.json"
out.parent.mkdir(exist_ok=True)
checks = {}

def check(name, expression):
    residual = sp.simplify(expression)
    if residual != 0:
        raise AssertionError((name, residual))
    checks[name] = str(residual)

V1, V2, T1, T2 = sp.symbols("V1 V2 T1 T2", positive=True)
check("weight_odds_minor", (V1/T1)*(V2/T2)-(V1/T2)*(V2/T1))
O11, O12, O21, scale = sp.symbols("O11 O12 O21 scale", positive=True)
check("positive_factorization_converse", scale*O21/(scale*O11/O12)-O21*O12/O11)
w, v, wm, g, beta, wp, t = sp.symbols("w v wm g beta wp t", positive=True)
dl, dt, ell, B = sp.symbols("dl dt ell B", real=True)
R = sp.symbols("R", positive=True)
b = (1-w)/(w*v)
s = wm/sp.sqrt(w)
betaY = g*(1-w)/v
check("classical_choice_reduction",
      (w*dt+(1-w)*dl/v)/(sp.sqrt(w)*wm*R)-(dt+b*dl)/(s*R))
check("calibrated_bridge_weight", betaY/(g*b)-w)
check("calibrated_bridge_speed", g/betaY-1/b-v)
check("calibrated_bridge_noise_squared", s**2*betaY/(g*b)-wm**2)
check("shared_context_line", s**2-wm**2-wm**2*v*b)
u = sp.symbols("u", positive=True)
newv = (1-u)/(u*b)
newwm2 = s**2*u
newg = betaY/(b*u)
check("unknown_gain_preserves_reproduction_time", newg*u-g*w)
check("unknown_gain_preserves_reproduction_slope", newg*(1-u)/newv-betaY)
m1, m2 = w*t+beta*ell, u*t+beta*ell
newB = B+(w-u)*t
check("post_bias_mean_equality", m2+newB-m1-B)
check("post_bias_variance_equality", wp**2*(m2+newB)**2-wp**2*(m1+B)**2)
check("pre_bias_sd_change", wp*m1-wp*m2-wp*(w-u)*t)
check("pre_post_overlap_estimate", (w+B/t)*t+beta*ell-(m1+B))
muA, varA = sp.symbols("muA varA", real=True)
Ey2 = (1+wp**2)*(varA+muA**2)+2*B*muA+B**2
check("random_estimate_total_variance", Ey2-(muA+B)**2-((1+wp**2)*varA+wp**2*muA**2))
A1, A2, C0, C1, C2 = sp.symbols("A1 A2 C0 C1 C2", real=True)
check("distance_contrast_minor",
      A1*(C1-C0)*A2*(C2-C0)-A1*(C2-C0)*A2*(C1-C0))
a, ws, amp = sp.symbols("a ws amp", positive=True)
x, offset = sp.symbols("x offset", real=True)
curve = amp*(1+x/a)**(1-ws)+offset
check("logarithmic_curvature", -sp.diff(curve,x,2)/sp.diff(curve,x)-ws/(a+x))
rho, lx, lz, lam = sp.symbols("rho lx lz lam", real=True)
cgf = sp.log(lam*sp.exp(rho*lx)+(1-lam)*sp.exp(rho*lz))
check("generalized_mean_first_order",
      sp.diff(cgf,rho,2).subs(rho,0)/2-lam*(1-lam)*(lx-lz)**2/2)
t0 = sp.symbols("t0", positive=True)
check("arithmetic_mean_scale_reduction",
      t0*(w*(1+t/t0)+(1-w)*(1+ell/(v*t0))-1)-(w*t+(1-w)*ell/v))

distances=np.array([0.,1.,2.,4.,8.,16.,25.])
tval,bval,wpval=.8,.08,.15
ma=.8*tval+bval*distances
mb=.5*tval+bval*distances
mua,mub=ma+.1,mb+.34
probe=mua+.23
post_error=float(np.max(abs(norm.logpdf(probe,mua,wpval*mua)-norm.logpdf(probe,mub,wpval*mub))))
times=np.array([.5,.8,1.2,2.])[:,None]
ls=np.array([0.,.5,2.,5.,10.])[None,:]
bias=np.array([.1,.23,-.05,.3])[:,None]
wsval=.83
means=(1+times)**wsval*(1+ls/.09)**(1-wsval)-1+bias
contrasts=means[:,1:]-means[:,[0]]
minors=[contrasts[i,u]*contrasts[j,z]-contrasts[i,z]*contrasts[j,u]
        for i in range(4) for j in range(i+1,4)
        for u in range(4) for z in range(u+1,4)]
numeric={
 "post_bias_max_log_density_error":post_error,
 "pre_bias_sd_difference_seconds":float(np.max(abs(wpval*ma-wpval*mb))),
 "log_contrast_max_minor_error":float(np.max(abs(np.array(minors)))),
}
assert post_error < 1e-12
assert numeric["log_contrast_max_minor_error"] < 1e-12
report={
 "purpose":"Symbolic identities and deterministic illustrations, not fitted recovery",
 "versions":{"python":platform.python_version(),"numpy":np.__version__,
             "scipy":scipy.__version__,"sympy":sp.__version__},
 "symbolic_identity_count":len(checks),"symbolic_residuals":checks,
 "deterministic_examples":numeric,
 "new_random_draws":0,"participant_data_used":False,
 "parameter_recovery_run":False,"model_recovery_run":False,
}
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(report,ensure_ascii=False,indent=2))
