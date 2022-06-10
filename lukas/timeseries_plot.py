from model import f, run_model, run_parameter_scan
import numpy as np
import scipy as scp
from scipy.constants import N_A
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

rc_ticks = {
            "text.usetex": False,
            "font.size":8,
            "legend.fontsize":6,
            "legend.title_fontsize":10,
            "xtick.labelsize":8,
            "ytick.labelsize": 8,
            "axes.labelsize":10,
            "axes.titlesize":12,
            "lines.linewidth": 1,
            "axes.linewidth": 0.5,
            "lines.markersize": 3,
            "xtick.major.size":4,
            "xtick.major.width":0.5,
            "xtick.major.pad":1,
            "xtick.minor.size": 1.5,
            "xtick.minor.width": 0.5,
            "xtick.minor.pad": 1,
            "ytick.major.size": 4,
            "ytick.major.width": 0.5,
            "ytick.major.pad": 1,
            "ytick.minor.size": 1.5,
            "ytick.minor.width": 0.5,
            "ytick.minor.pad": 1,
            "axes.labelpad":1,
            "axes.titlepad":1
        }
plt.rcParams.update(rc_ticks)

p = dict(
    k_RL = 2e6,
    k_RLm = 1e-2,
    k_Rs = 4,
    k_Rd0 = 4e-4,
    k_Rd1 = 4e-3,
    k_G1 = 1,
    k_Ga = 1e-5,
    L = 1e-6,
    k_Gd = 0.1,
    Rt = 1e4, # total receptor
    Gt = 1e4,  # total heterotrimeric G protein

)

t,y = run_model(np.linspace(0,600,1000),p, initial_fractions = {"fR_0":1,"fG_0":1})

plt.figure(figsize=(3,2))
plt.plot(t,y[0,:]/p["Rt"])
plt.plot(t,y[1,:]/p["Rt"])

plt.plot(t,y[2,:]/p["Gt"])
plt.plot(t,y[3,:]/p["Gt"])

plt.legend(["R", "RL", "G", "Ga"])
plt.tight_layout()
plt.show()
