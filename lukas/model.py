from scipy.integrate import solve_ivp
from scipy.constants import N_A
import pandas as pd
import numpy as np

def f(t,X, p) -> list:
    """

    :param current_state: current state of the ODE system
    :param t: time of simulation
    :param parameter: rate constant for G protein deactivation
    :return: solution of the ODE System
    """


    k_RL = p["k_RL"]
    k_RLm = p["k_RLm"]
    k_Rs = p["k_Rs"]
    k_Rd0 = p["k_Rd0"]
    k_Rd1 = p["k_Rd1"]
    k_G1 = p["k_G1"]
    k_Ga = p["k_Ga"]
    Gt = p["Gt"]

    L = p["L"]
    k_Gd = p["k_Gd"]

    R, RL, G, Ga = X



    # algebraic equations:
    Gd = Gt - G - Ga  # Galpha-GDP
    Gbg = Gt - G  # free Gbetagamma

    # the ODEs ahead:
    dR_dt = -k_RL*L*R + k_RLm*RL - k_Rd0*R + k_Rs
    dRL_dt = k_RL*L*R - k_RLm*RL - k_Rd1*RL
    dG_dt = -k_Ga*RL*G + k_G1*Gd*Gbg
    dGa_dt = k_Ga*RL*G - k_Gd*Ga

    return [dR_dt, dRL_dt, dG_dt, dGa_dt]

def run_model(T,p,initial_fractions = {"fR_0":1,"fG_0":1}):

    p = p.copy()
    p.update(initial_fractions)

    R_0 = p["Rt"] * p["fR_0"]  # free receptor
    RL_0 = p["Rt"] * (1-p["fR_0"]) # receptor bound to ligand
    G_0 = p["Gt"] * p["fG_0"] # inactive heterotrimeric G protein
    Ga_0 = p["Gt"] * (1-p["fG_0"])  # active Galpha-GTP


    S0 = [R_0, RL_0, G_0, Ga_0]

    R = solve_ivp(f, (T[0], T[-1]), S0, args=(p,),
                  method="LSODA", t_eval=T,first_step = 1,rtol = 1e-8)
    return R.t, R.y


def run_parameter_scan(p,scan_over_names = None, t_max = 500, fold_change = 1,n_samples = 10, initial_fractions = {"fR_0":1,"fG_0":1}):
    df = []

    p = p.copy()

    if scan_over_names is None:
        scan_over_names = p.keys()


    for i, k in enumerate(scan_over_names):
        v = p[k]
        s = np.logspace(-1 * fold_change, fold_change, n_samples).astype(float)
        for h in s:
            p_copy = p.copy()
            p_copy[k] = v * h
            p_copy.update(initial_fractions)

            t, y = run_model(np.linspace(0, t_max, 100), p_copy)

            df.append({"parameter": k, "v": p_copy[k],
                       "fold_change": h,
                       "readout": "R",
                       "value": y[0, -1]})
            df.append({"parameter": k, "v": p_copy[k],
                       "fold_change": h,
                       "readout": "RL",
                       "value": y[1, -1]})
            df.append({"parameter": k, "v": p_copy[k],
                       "fold_change": h,
                       "readout": "G",
                       "value": y[2, -1]})
            df.append({"parameter": k, "v": p_copy[k],
                       "fold_change": h,
                       "readout": "Ga",
                       "value": y[3, -1] / p_copy["Gt"]})
            df.append({"parameter": k, "v": p_copy[k],
                       "fold_change": h,
                       "readout": "Ga_max",
                       "value": np.max(y[3, :] / p_copy["Gt"])})
            df.append({"parameter": k, "v": p_copy[k],
                       "fold_change": h,
                       "readout": "Gd",
                       "value": p_copy["Gt"] - y[3, -1]})

    df = pd.DataFrame(df)
    return df