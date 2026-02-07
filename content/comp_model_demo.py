import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from sympy import symbols, lambdify, integrate
import argparse
from utils import (
    create_aif_func,
    poly_exp_aif_params,
    micro_params_2tcm,
    resp_funcs_2tcm,
)

# %%
parser = argparse.ArgumentParser(description="Kinetic modeling demo")
parser.add_argument(
    "--aif", type=int, default=1, help="AIF type (1-3)", choices=[1, 2, 3]
)
parser.add_argument(
    "--params",
    type=int,
    default=1,
    help="Micro parameter type (1-2)",
    choices=[1, 2],
)
parser.add_argument("--tmax", type=float, default=90.0, help="Maximum time (min)")
parser.add_argument("--num_frames", type=int, default=35, help="Number of frames")
parser.add_argument(
    "--num_logan", type=int, default=4, help="Number of Logan plot points"
)

args = parser.parse_args()

aif_type = args.aif
micro_param_type = args.params
tmax = args.tmax
num_frames = args.num_frames
n_logan = args.num_logan

# %%
if aif_type == 1:
    aif_params = poly_exp_aif_params(AIF_amps=[-6.0, 5.0, 1.0], AIF_exps=[4, 1, 0.02])
elif aif_type == 2:
    aif_params = poly_exp_aif_params(AIF_amps=[-10, 9.5, 0.5], AIF_exps=[4, 1, 0])
elif aif_type == 3:
    aif_params = poly_exp_aif_params(AIF_amps=[-2, 1.5, 0.5], AIF_exps=[40, 40, 0])
else:
    raise ValueError("Invalid aif_type")

if micro_param_type == 1:
    micro_params = micro_params_2tcm(K1=0.4, k2=0.2, k3=0.14, k4=0.07)
elif micro_param_type == 2:
    micro_params = micro_params_2tcm(K1=0.2, k2=0.2, k3=0.05, k4=0.1)
else:
    raise ValueError("Invalid micro_param_type")

# %%
aif_func = create_aif_func(aif_params)
aif_numeric = lambdify(symbols("t"), aif_func, modules=["numpy"])

resp_func, resp1_func, resp2_func, uir_func, uir1_func, uir2_func = resp_funcs_2tcm(
    micro_params, aif_params
)

resp_numeric = lambdify(symbols("t"), resp_func, modules=["numpy"])
resp1_numeric = lambdify(symbols("t"), resp1_func, modules=["numpy"])
resp2_numeric = lambdify(symbols("t"), resp2_func, modules=["numpy"])

uir_numeric = lambdify(symbols("t"), uir_func, modules=["numpy"])
uir1_numeric = lambdify(symbols("t"), uir1_func, modules=["numpy"])
uir2_numeric = lambdify(symbols("t"), uir2_func, modules=["numpy"])

# %%
# plots of AIF, responses and logan plot
t_values = np.linspace(0, tmax, 1000)
t_frm = np.logspace(-1, np.log10(tmax), num_frames)

# %%
t = symbols("t")
time_integrated_resp_func = integrate(resp_func, (t, 0, t))
time_integrated_aif = integrate(aif_func, (t, 0, t))

logan_y_func = time_integrated_resp_func / resp_func
logan_t_func = time_integrated_aif / resp_func

logan_y_func_numeric = lambdify(t, logan_y_func, modules=["numpy"])
logan_t_func_numeric = lambdify(t, logan_t_func, modules=["numpy"])

# evaluate logan y and t at the frame times
logan_y_frm = logan_y_func_numeric(t_frm)
logan_t_frm = logan_t_func_numeric(t_frm)

# do scipy or numpy linear regression to get the slope and intercept of the logan plot
slope, intercept, r_value, p_value, std_err = linregress(
    logan_t_frm[-n_logan:], logan_y_frm[-n_logan:]
)
print(f"Logan plot slope: {slope:.4f}, true Vt: {micro_params.Vt:.4f}")

# %%
# sample the AIF and response at the frame times
CT_frm = resp_numeric(t_frm)
aif_frm = aif_numeric(t_frm)

# save to single csv file using np.savetxt
data = np.column_stack((t_frm, aif_frm, CT_frm))
np.savetxt(
    f"TACs_sub_{aif_type}_region_{micro_param_type}.csv",
    data,
    delimiter=",",
    header="time[min],arterial input function Ca[kBq/ml], tissue activity concentration C[kBq/ml]",
    comments="",
)

# %%
fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
ax[0].plot(t_values, aif_numeric(t_values), "k-", label="Ca")
ax[0].plot(t_values, resp_numeric(t_values), "r-", label="C")
ax[0].plot(t_values, resp1_numeric(t_values), "b--", label="C1")
ax[0].plot(t_values, resp2_numeric(t_values), "g--", label="C2")
ax[0].plot(t_frm, aif_frm, "ko")
ax[0].plot(t_frm, CT_frm, "ro")
ax[0].set_xlabel("t (min)")
ax[0].set_ylabel("C(t)")
ax[0].grid(ls=":")
ax[0].legend()

ax[1].plot(t_values, uir_numeric(t_values), "k-", label="UIR")
ax[1].plot(t_values, uir1_numeric(t_values), "b--", label="UIR1")
ax[1].plot(t_values, uir2_numeric(t_values), "g--", label="UIR2")
ax[1].set_xlabel("t (min)")
ax[1].set_ylabel("UIR(t)")
ax[1].grid(ls=":")
ax[1].legend()

ax[2].plot(logan_t_func_numeric(t_values[1:]), logan_y_func_numeric(t_values[1:]))
ax[2].plot(logan_t_frm, logan_y_frm, ".")
ax[2].grid(ls=":")
ax[2].set_xlabel("Logan time")
ax[2].set_ylabel("Logan y")
ax[2].set_title("Logan plot")
fig.savefig(f"comp_model_demo_sub_{aif_type}_region_{micro_param_type}.pdf")
fig.show()
