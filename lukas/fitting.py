import plotly.io

from model import f, run_model, run_parameter_scan, p, run_dose_response
import numpy as np
import scipy as scp
from scipy.constants import N_A
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit



p_fit = dict(
    # k_RL = 2e6,
    # k_RLm = 1e-2,
    # k_Rs = 4,
    # k_Rd0 = 4e-4,
    # k_Rd1 = 4e-3,
    # k_G1 = 1,
    k_Ga = 1e-5,
    # L = 1e-6,
    k_Gd = 1,
    # Rt = 1e4, # total receptor
    # Gt = 1e4,  # total heterotrimeric G protein

)

def timeseries_fit_function(T , *params):

    p_dict = dict(zip([k for k in p_fit.keys()],params))
    p_merged = p.copy()
    p_merged.update(p_dict)
    t,y = run_model(T,p_merged, initial_fractions = {"fR_0":1,"fG_0":1})

    if y.shape[1] < len(T):
        return np.inf
    else:
        return y[3,:]/p_merged["Gt"]

def dose_response_fit_function(A , *params):

    p_dict = dict(zip([k for k in p_fit.keys()],params))
    p_merged = p.copy()
    p_merged.update(p_dict)

    Y = run_dose_response(A,p_merged)


    return Y


def run_timeseries_fit(p_fit, xdata,ydata,y_std):

    fit = curve_fit(timeseries_fit_function,xdata.copy(),ydata,list(p_fit.values()),sigma=y_std, method="lm")
    p_fitted = {k:fit[0][i] for i,k in enumerate(p_fit)}

    p_optimized = p.copy()
    p_optimized.update(
        p_fitted
    )

    return p_optimized

def run_dose_response_fit(p_fit, p, xdata, ydata, y_std):


    # xdata = np.array( data_df["alphaF"])
    # ydata = np.array(data_df["Gbg_mean"])
    # y_std = np.array(data_df["Gbg_std"])
    p.update({"L":1e-9})
    from scipy.optimize import minimize

    def objective(x,p):

        p.update(dict(zip(p_fit.keys(),x)))

        y = dose_response_fit_function(xdata,*x)

        S = np.sum(np.power(y-ydata,2))

        print(S)
        return S


    fit = minimize(
        objective,list(p_fit.values()),args=(p.copy(),),method="Nelder-Mead",
        tol = 1e-3
    )
    p_fitted = dict(zip(p_fit.keys(),fit.x))
    assert fit.success, "fit failed"
    p_optimized = p.copy()
    p_optimized.update(
        p_fitted
    )

    return p_optimized


#%%
data_df_A = pd.read_csv("../data/Fig-5-A-data.csv")
xdata_A = np.array(data_df_A["time"])
ydata_A = np.array(data_df_A["Gbg_mean"])
y_std_A = np.array(data_df_A["Gbg_std"])

p_optimized = run_timeseries_fit(p_fit, xdata_A,ydata_A,y_std_A)
o_t, o_y = run_model(np.linspace(0, 600, 1000),p_optimized, initial_fractions={"fR_0": 1, "fG_0": 1})

plt.plot(o_t, o_y[3, :] / p_optimized["Gt"])
plt.errorbar(xdata_A, ydata_A, yerr=y_std_A, fmt="o")
dif = np.array(list(p.values())) - np.array(list(p_optimized.values()))
plt.show()

#%%

# data_df_B = pd.read_csv("../data/Fig-5-B-data.csv")
# xdata_B = np.array(data_df_B["alphaF"])
#
# ydata_B = np.array(data_df_B["Gbg_mean"])
# ydata_B = ydata_B/data_df_A.iloc[2]["Gbg_mean"]
#
# y_std_B = np.array(data_df_B["Gbg_std"])
# y_std_B = y_std_B/data_df_A.iloc[2]["Gbg_mean"]
#
# p_optimized = run_dose_response_fit(p_fit, p.copy(), xdata_B,ydata_B,y_std_B)
#
# s = np.logspace(-3,3,100)
# o_y = run_dose_response(s, p_optimized)
# plt.plot(s, o_y)
#
# plt.errorbar(xdata_B, ydata_B, yerr=y_std_B, fmt="o")
# plt.gca().set_xscale('log')
# # dif = np.array(list(p.values())) - np.array(list(p_optimized.values()))
# plt.show()
