"""
smith_duller_model.py  -  Smith & Duller (1959) with Chatzidakis et al. (2015) parameters

Source : Smith & Duller, J. Geophys. Res. 64, 2297 (1959)
         Chatzidakis, Chrysikopoulou, Tsoukalas, NIM A 804, 33 (2015)
         (equations and Table 2 as reproduced in Su et al., Front. Energy Res. 9, 750159)
Units  : returns dPhi/dE in m^-2 s^-1 sr^-1 GeV^-1
NOTE   : j_pi (pion decay constant) is not defined in Su et al.; the definition
         j_pi = m_pi c^2 y0 / (rho0 c tau_pi) is used here.
"""
import numpy as np
from muon_kinematics import CM2_TO_M2, scalar_if_scalar

# ---- Table 2 parameters (energies in GeV, depths in g/cm^2, lengths in cm) ----
A_FIT = 0.002382
R = 0.76                    # muon-to-pion energy ratio
A_LOSS = 2.5e-3             # GeV / (g/cm^2)   (2.5 MeV per g/cm^2)
Y0 = 1000.0                 # g/cm^2, atmospheric depth at sea level
GAMMA = 8.0 / 3.0
B_MU = 0.800                # temperature correction factor b_mu
M_MU = 0.105659             # GeV
TAU_MU = 2.2e-6             # s
RHO0 = 0.00123              # g/cm^3
C_LIGHT = 3.0e10            # cm/s
LAMBDA_PI = 120.0           # g/cm^2, pion absorption mean free path
B_ISO = 0.771               # isothermal-atmosphere correction b
TAU_PI = 2.61e-8            # s
M_PI = 0.13958              # GeV

# Derived constants (energies in GeV)
B_MU_E = B_MU * M_MU * Y0 / (RHO0 * C_LIGHT * TAU_MU)   # Eq. (9)  ~ 1.04 GeV
J_PI = M_PI * Y0 / (RHO0 * C_LIGHT * TAU_PI)            # ~ 145 GeV


def smith_duller_chatzidakis(E, theta):
    E = np.asarray(E, dtype=float)
    c = np.cos(theta)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        sec = 1.0 / c
        E_pi = (E + A_LOSS * Y0 * (sec - 0.100)) / R                          # Eq. (7)
        base = 0.100 * c * (1.0 - A_LOSS * (Y0 * sec - 100.0) / (R * E_pi))
        expo = B_MU_E / ((R * E_pi + 100.0 * A_LOSS) * c)
        P_mu = base ** expo                                                    # Eq. (8)
        bj = B_ISO * J_PI
        flux_cm2 = A_FIT * E_pi ** (-GAMMA) * P_mu * LAMBDA_PI * bj / (E_pi * c + bj)  # Eq. (6)
    flux_cm2 = np.nan_to_num(flux_cm2, nan=0.0, posinf=0.0)
    return scalar_if_scalar(flux_cm2 * CM2_TO_M2)
