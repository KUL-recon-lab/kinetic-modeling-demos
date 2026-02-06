import numpy as np
import matplotlib.pyplot as plt

from sympy import symbols, lambdify, integrate
from utils import (
    create_aif_func,
    poly_exp_aif_params,
    micro_params_2tcm,
    resp_funcs_2tcm,
)

# %%
micro_params = micro_params_2tcm(K1=0.3, k2=0.3, k3=0.01, k4=0.005)
aif_params = poly_exp_aif_params(AIF_amps=[-3.0, 2.0, 1.0], AIF_exps=[4, 1, 0.01])

# infusion_aif_params = poly_exp_aif_params(AIF_amps=[-2, 1.5, 0.5], AIF_exps=[40, 40, 0])
# bolus_infusion_aif_params = poly_exp_aif_params(
#    AIF_amps=[-25, 24.5, 0.5], AIF_exps=[4, 1, 0]
# )
aif_func = create_aif_func(aif_params)
aif_numeric = lambdify(symbols("t"), aif_func, modules=["numpy"])

tmax = 120.0
num_frames = 35

# %%
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
t_values = np.linspace(0, tmax, 1000)
t_frm = np.logspace(-1, np.log10(tmax), num_frames)

fig1, ax1 = plt.subplots(figsize=(7, 4), layout="constrained")
ax1.plot(t_values, aif_numeric(t_values), "k-")
ax1.plot(t_values, resp_numeric(t_values), "r-")
ax1.plot(t_frm, aif_numeric(t_frm), "ko")
ax1.plot(t_frm, resp_numeric(t_frm), "ro")
ax1.set_xlabel("t (min)")
ax1.set_ylabel("Ca(t)")
ax1.grid(ls=":")
fig1.show()

# %%
# logan plot
t = symbols("t")
time_integrated_resp_func = integrate(resp_func, (t, 0, t))
time_integrated_aif = integrate(aif_func, (t, 0, t))

logan_y_func = time_integrated_resp_func / resp_func
logan_t_func = time_integrated_aif / resp_func

logan_y_func_numeric = lambdify(t, logan_y_func, modules=["numpy"])
logan_t_func_numeric = lambdify(t, logan_t_func, modules=["numpy"])

fig_lin, ax_lin = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)
ax_lin.plot(logan_t_func_numeric(t_values[1:]), logan_y_func_numeric(t_values[1:]))
ax_lin.plot(logan_t_func_numeric(t_frm), logan_y_func_numeric(t_frm), ".")
# ax_lin.set_aspect(1)
ax_lin.grid(ls=":")
ax_lin.set_xlabel("Logan time")
ax_lin.set_ylabel("Logan y")
ax_lin.set_title("Logan plot")
fig_lin.show()
