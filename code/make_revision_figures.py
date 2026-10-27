#!/usr/bin/env python3
"""Plot the manuscript's analytical constraints and noise-reference comparison.

Every curve is evaluated from the stated equations and selected parameters.
No fitting, response sampling, or image-generation model is used.
"""
from pathlib import Path
import json
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"build"/"matplotlib"))
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = ROOT/"figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":8.5,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.labelsize":8.3,"axes.titlesize":9.3,
    "legend.fontsize":7.2,"pdf.fonttype":42,"ps.fonttype":42,
    "axes.titlepad":9,"lines.linewidth":1.6,
})
blue,orange,green,purple="#145b78","#ba542f","#387e57","#73558f"

def save(fig,stem):
    for ext in ["pdf","png"]:
        fig.savefig(FIG/f"{stem}.{ext}",dpi=230)
    plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(7.35,2.75),layout="constrained")
ax=axes[0]
vv=np.geomspace(.25,4,180)
for T,color in zip([.5,1,2],[blue,orange,green]):
    w=T/(T+vv)
    ax.plot(np.log(vv),np.log(w/(1-w)),color=color,label=fr"$T/V_0={T:g}$")
ax.set(xlabel=r"$\log(V/V_0)$",ylabel=r"$\operatorname{logit}(w)$",
       xticks=[-1,0,1])
ax.set_title("A   Reliability odds",loc="left",fontweight="bold")
ax.legend(frameon=False,loc="lower left")

ax=axes[1]
weights=np.array([.2,.35,.5,.65,.8,.9])
v,wm=.25,.2
b=(1-weights)/(weights*v)
s2=wm**2/weights
line=np.linspace(0,17,200)
ax.plot(line,wm**2+wm**2*v*line,color=blue)
ax.plot(b,s2,"o",color=orange,mfc="white",ms=4.5)
ax.plot([0],[wm**2],"s",color=blue,ms=4)
ax.set(xlabel=r"$b$ (s/degree)",ylabel=r"$s^2$",xlim=(-.5,17.5),ylim=(.02,.23))
ax.set_title("B   Shared speed and noise",loc="left",fontweight="bold")
ax.text(.04,.94,r"$s^2=A+Db$",transform=ax.transAxes,va="top")
ax.text(.04,.83,r"$A=.04,\quad D=.01$",transform=ax.transAxes,va="top",fontsize=7.5)

ax=axes[2]
ell=np.linspace(0,10,161)
t0,a,ws=1.,.09,.83
times=[.5,.8,1.2,2.]
biases=[.1,.23,-.05,.3]
ax.plot(ell,(1+ell/a)**(1-ws)-1,color="#6c737a",lw=1.2)
for i,(t,B,color,marker) in enumerate(zip(times,biases,[blue,orange,green,purple],["o","s","^","D"])):
    A=t0*(1+t/t0)**ws
    mu=A*(1+ell/a)**(1-ws)+B-t0
    normalized=(mu-mu[0])/A
    idx=np.arange(6+i*7,len(ell),28)
    ax.plot(ell[idx],normalized[idx],marker,ls="none",color=color,mfc="white",
            ms=4,label=fr"$t={t:g}$ s")
ax.set(xlabel="Distance (degrees)",ylabel=r"Distance contrast / $A(t)$",xlim=(0,10))
ax.set_title("C   Bias-free contrast",loc="left",fontweight="bold")
ax.legend(frameon=False,loc="lower right",fontsize=6.8)
save(fig,"figure1_empirical_constraints")

fig,axes=plt.subplots(2,2,figsize=(7.25,5.25),layout="constrained")
distances=np.array([0.,1.,2.,4.,8.,16.,25.])
t,beta,wp=.8,.08,.15
m1=.8*t+beta*distances
m2=.5*t+beta*distances
mu1,mu2=m1+.1,m2+.34
aa,ab,ac,ad=axes.ravel()
aa.plot(distances,mu1,color=blue,label=r"$w=.8,\ B=.1$")
aa.plot(distances,mu2,"o",color=orange,mfc="white",ms=4.8,label=r"$w=.5,\ B=.34$")
aa.set(xlabel="Distance (degrees)",ylabel="Mean reproduction (s)")
aa.legend(frameon=False)
aa.set_title("A   Equal means",loc="left",fontweight="bold")
ab.plot(distances,wp*m1,"-o",color=blue,ms=4,label=r"$w=.8$")
ab.plot(distances,wp*m2,"--s",color=orange,ms=4,label=r"$w=.5$")
ab.set(xlabel="Distance (degrees)",ylabel="Response SD (s)")
ab.legend(frameon=False)
ab.set_title("B   Pre-bias noise separates",loc="left",fontweight="bold")
ac.plot(distances,wp*mu1,color=blue,label=r"$w=.8$")
ac.plot(distances,wp*mu2,"o",color=orange,mfc="white",ms=4.8,label=r"$w=.5$")
ac.set(xlabel="Distance (degrees)",ylabel="Response SD (s)")
ac.legend(frameon=False)
ac.set_title("C   Post-bias noise preserves equality",loc="left",fontweight="bold",fontsize=9.1)
m1p,m2p=.8*t+beta*4,.5*t+beta*4
mup=m1p+.1
y=np.linspace(mup-4*wp*mup,mup+4*wp*mup,360)
ad.plot(y,norm.pdf(y,mup,wp*m1p),color=blue,label=r"Pre, $w=.8$")
ad.plot(y,norm.pdf(y,mup,wp*m2p),"--",color=orange,label=r"Pre, $w=.5$")
ad.plot(y,norm.pdf(y,mup,wp*mup),":",color=green,lw=2,label="Post, both")
ad.set(xlabel="Reproduced duration (s)",ylabel=r"Density (s$^{-1}$)")
ad.legend(frameon=False,fontsize=7)
ad.set_title("D   Complete response distributions",loc="left",fontweight="bold",fontsize=9.1)
save(fig,"figure2_noise_reference")
print(json.dumps({"figures":["figure1_empirical_constraints","figure2_noise_reference"],
                  "source":"Analytical equations and stated parameter values",
                  "new_fit":False,"new_random_draws":0}))
