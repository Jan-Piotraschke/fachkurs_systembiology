import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import numpy as np
from sympy.parsing.mathematica import mathematica

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


def J(t,X,p,*args):

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
    Gd = Gt - G - Ga  # Galpha-GDP
    Gbg = Gt - G  # free Gbetagamma

    return [
        [-k_Rd0 - k_RL*L,0,k_RLm,0],
        [k_RL*L,0,-k_Rd1-k_RLm,0],
        [0,-((k_Ga*k_RL*k_Rs*L)/(k_Rd0*k_Rd1 + k_Rd0*k_RLm + k_Rd1*k_RL*L)),-((Gbg*Gd*k_G1*(k_Rd1*k_Rd1 + k_Rd0*k_RLm + k_Rd1*k_RL*L))/(k_RL*k_Rs*L)),0],
        [0,(k_Ga*k_RL*k_Rs*L)/(k_Rd0*k_Rd1 + k_Rd0*k_RLm + k_Rd1*k_RL*L),(Gbg*Gd*k_G1*(k_Rd0*k_Rd1 + k_Rd1*k_RLm + k_Rd1*k_RL*L))/(k_RL*k_Rs*L),-k_Gd]


    ]


def fixpoint(p):

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

    # R, RL, G, Ga = X
    # Gd = Gt - G - Ga  # Galpha-GDP
    # Gbg = Gt - G  # free Gbetagamma

    R_f = (k_Rd1 * k_Rs + k_RLm*k_Rs)/(k_Rd0*k_Rd1 + L*k_Rd1*k_RL + k_Rd0*k_RLm)
    RL_f =(L*k_RL*k_Rs)/(k_Rd0*k_Rd1 + L*k_Rd1*k_RL + k_Rd0*k_RLm)
    G_f = (Gbg*Gd*k_G1*(k_Rd0*k_Rd1 + L*k_Rd1*k_RL +
   k_Rd0*k_RLm))/(L*k_Ga*k_RL*k_Rs)
    Ga_f = (Gbg*Gd*k_G1)/k_Gd

    return [R_f,RL_f,G_f,Ga_f]

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

    R = solve_ivp(f, (T[0], T[-1]), S0, method = "LSODA", args=(p,),
                  t_eval=T,rtol = 1e-8,max_step = 1,first_step = 0.1)
    return R.t, R.y

def run_dose_response(A,p):

    p_norm = p.copy()
    p_norm.update({"L":1e-6})

    t,norm_y = run_model(np.linspace(0, 60, 10), p_norm, initial_fractions={"fR_0": 1, "fG_0": 1})
    Y = []

    for fold_change in A:

        p_l = p.copy()
        p_l["L"] = 1e-9 * fold_change
        t, y = run_model(np.linspace(0, 60, 10), p_l, initial_fractions={"fR_0": 1, "fG_0": 1})
        # plt.plot(
        #     (y[3,:]/(norm_y[3, -1]))
        # )
        # plt.show()
        # plt.close(plt.gcf())
        if len(y) == 0:
            Y.append(np.nan)
        else:
            Y.append(
                np.max(y[3,:])/p["Gt"]/(np.max(norm_y[3, :])/p["Gt"])
            )

    return Y

def run_parameter_scan(p,scan_over_names = None, t_max = 500, fold_change = 1,n_samples = 10, initial_fractions = {"fR_0":1,"fG_0":1}):
    df = []

    p = p.copy()

    if scan_over_names is None:
        scan_over_names = p.keys()


    for i, k in enumerate(scan_over_names):
        v = p[k]
        s = np.logspace(-1 * fold_change, fold_change, n_samples).astype(float)
        for h in s:
            try:
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
            except:
                continue

    df = pd.DataFrame(df)
    return df