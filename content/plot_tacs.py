import numpy as np
import matplotlib.pyplot as plt

tacs_file = "TACs_sub_1_region_1.csv"

data = np.loadtxt(tacs_file, delimiter=",", skiprows=1)

t = data[:, 0]
Ca = data[:, 1]
C = data[:, 2]


fig, ax = plt.subplots(layout="constrained")
ax.plot(t, Ca, "o-", label="arterial input Ca(t)")
ax.plot(t, C, "o-", label="tissue concentration C(t)")
ax.set_xlabel("t [min]")
ax.set_ylabel("concentration [kBq/ml]")
ax.legend()
ax.grid(ls=":")
plt.show()

# %%
from scipy.integrate import cumulative_trapezoid
from scipy.stats import linregress

n_logan = 4

logan_t = cumulative_trapezoid(Ca, t, initial=0) / C
logan_y = cumulative_trapezoid(C, t, initial=0) / C

# do scipy or numpy linear regression to get the slope and intercept of the logan plot
slope, intercept, r_value, p_value, std_err = linregress(
    logan_t[-n_logan:], logan_y[-n_logan:]
)
print(f"Logan plot slope: {slope:.4f}")
