import numpy as np
import matplotlib.pyplot as plt

subject_ids = [1, 2, 3]
region_ids = [1, 2]

fig, ax = plt.subplots(
    len(subject_ids),
    len(region_ids),
    figsize=(12, 8),
    layout="constrained",
    sharex=True,
    sharey="row",
)

for i_sub, subject_id in enumerate(subject_ids):
    for i_reg, region_id in enumerate(region_ids):
        tacs_file = f"TACs_sub_{subject_id}_region_{region_id}.csv"
        print(f"Loading {tacs_file}")

        data = np.loadtxt(tacs_file, delimiter=",", skiprows=1)

        t = data[:, 0]
        Ca = data[:, 1]
        C = data[:, 2]

        ax[i_sub, i_reg].plot(t, Ca, "o-", label="arterial input Ca(t)")
        ax[i_sub, i_reg].plot(t, C, "o-", label="tissue concentration C(t)")
        ax[i_sub, i_reg].set_xlabel("t [min]")
        ax[i_sub, i_reg].set_ylabel("concentration [kBq/ml]")
        ax[i_sub, i_reg].grid(ls=":")

        ax[i_sub, i_reg].set_title(f"subject {subject_id}, region {region_id}")

        if i_sub == 0 and i_reg == 0:
            ax[i_sub, i_reg].legend()

fig.savefig("TACs.pdf")
fig.show()
