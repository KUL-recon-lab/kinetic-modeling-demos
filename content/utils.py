import itertools
import numpy as np
from dataclasses import dataclass
from sympy import sympify


@dataclass
class poly_exp_aif_params:
    AIF_amps: list[float]
    AIF_exps: list[float]

    def __post_init__(self):
        if not np.isclose(sum(self.AIF_amps), 0):
            raise ValueError(f"Sum of AIF_amps must equal 0, got {sum(self.AIF_amps)}")


@dataclass
class micro_params_2tcm:
    K1: float
    k2: float
    k3: float
    k4: float

    def __post_init__(self):
        if self.K1 < 0 or self.k2 < 0 or self.k3 < 0 or self.k4 < 0:
            raise ValueError(
                f"All parameters must be >= 0, got K1={self.K1}, k2={self.k2}, k3={self.k3}, k4={self.k4}"
            )

    @property
    def Vt(self) -> float:
        """Calculate volume of distribution at steady state."""
        if self.k4 == 0:
            return float("inf") if self.k3 > 0 else self.K1 / self.k2
        return (self.K1 / self.k2) * (1 + (self.k3 / self.k4))


def resp_funcs_2tcm(mirco_params: micro_params_2tcm, aif_params: poly_exp_aif_params):

    AIF_amps = aif_params.AIF_amps
    AIF_exps = aif_params.AIF_exps

    K1 = mirco_params.K1
    k2 = mirco_params.k2
    k3 = mirco_params.k3
    k4 = mirco_params.k4

    # calculate amplitudes and exponents for the united impulse response function
    a1 = (k2 + k3 + k4 - np.sqrt((k2 + k3 + k4) ** 2 - 4 * k2 * k4)) / 2
    a2 = (k2 + k3 + k4 + np.sqrt((k2 + k3 + k4) ** 2 - 4 * k2 * k4)) / 2

    # calculate the united impulse response function for the 1st and 2nd tissue compartment
    UIR1_exps = [a1, a2]
    UIR1_amps = [
        K1 * (k4 - a1) / (a2 - a1),
        K1 * (a2 - k4) / (a2 - a1),
    ]

    UIR2_exps = [a1, a2]
    UIR2_amps = [
        K1 * k3 / (a2 - a1),
        -K1 * k3 / (a2 - a1),
    ]

    # setup the united impulse response functions for the 1st and 2nd tissue compartment
    UIR1_str = "+".join(
        [
            f"{UIR_amp}*exp(-{UIR_exp}*t)"
            for UIR_amp, UIR_exp in zip(UIR1_amps, UIR1_exps)
        ]
    )

    UIR1_func = sympify(UIR1_str)

    UIR2_str = "+".join(
        [
            f"{UIR_amp}*exp(-{UIR_exp}*t)"
            for UIR_amp, UIR_exp in zip(UIR2_amps, UIR2_exps)
        ]
    )

    UIR2_func = sympify(UIR2_str)

    # calculate the sum of the two impulse response functions
    UIR_func = UIR1_func + UIR2_func

    resp1_str_list = []

    for (UIR_amp, UIR_exp), (AIF_amp, AIF_exp) in itertools.product(
        zip(UIR1_amps, UIR1_exps), zip(AIF_amps, AIF_exps)
    ):
        if UIR_exp == AIF_exp:
            tmp = f"{AIF_amp} * {UIR_amp} * t * exp(-{UIR_exp}*t)"
        else:
            tmp = f"({AIF_amp} * {UIR_amp}) * (exp(-{UIR_exp}*t) - exp(-{AIF_exp}*t)) / ({AIF_exp} - {UIR_exp})"

        resp1_str_list.append(tmp)

    resp1_str = "+".join(resp1_str_list)
    resp1_func = sympify(resp1_str)
    # resp1_func_numeric = lambdify(t, resp1_func, modules=["numpy"])

    print(f"response of the 1st TC C1(t) : {resp1_func}")

    # calculate the response of the 2nd compartment
    if k3 != 0:
        resp2_str_list = []

        for (UIR_amp, UIR_exp), (AIF_amp, AIF_exp) in itertools.product(
            zip(UIR2_amps, UIR2_exps), zip(AIF_amps, AIF_exps)
        ):
            if UIR_exp == AIF_exp:
                tmp = f"{AIF_amp} * {UIR_amp} * t * exp(-{UIR_exp}*t)"
            else:
                tmp = f"({AIF_amp} * {UIR_amp}) * (exp(-{UIR_exp}*t) - exp(-{AIF_exp}*t)) / ({AIF_exp} - {UIR_exp})"

            resp2_str_list.append(tmp)

        resp2_str = "+".join(resp2_str_list)
        resp2_func = sympify(resp2_str)
        # resp2_func_numeric = lambdify(t, resp2_func, modules=["numpy"])

        # calculate the total response of the system
        resp_func = resp1_func + resp2_func
        # resp_func_numeric = lambdify(t, resp_func, modules=["numpy"])

        print(f"response of the 2nd TC C2(t) : {resp2_func}")
        print(f"response of the C1(t) + C2(t): {resp_func}")
    else:
        resp2_func = sympify("0")
        resp_func = resp1_func

    return resp_func, resp1_func, resp2_func, UIR_func, UIR1_func, UIR2_func


def create_aif_func(param: poly_exp_aif_params):
    AIF_amps = param.AIF_amps
    AIF_exps = param.AIF_exps

    AIF_str = "+".join(
        [f"{AIF_amp}*exp(-{AIF_exp}*t)" for AIF_amp, AIF_exp in zip(AIF_amps, AIF_exps)]
    )

    return sympify(AIF_str)
