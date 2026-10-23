"""
tang_model.py  -  Gaisser/Tang modified Gaisser formula

Source : Tang, Horton-Smith, Kudryavtsev, Tonazzo, Phys. Rev. D 74, 053007 (2006),
         arXiv:hep-ph/0604078, Eqs. (3)-(10).
Valid  : intended for the whole range of E and theta (fit degrades above ~85 deg).
Units  : returns dPhi/dE in m^-2 s^-1 sr^-1 GeV^-1

CAUTION - Eq. (7) ambiguity
---------------------------
Eq. (7) is printed with E (ground-level energy) in the exponent 4.5/(E cos*).
Taken literally, the flux comes out 100-1000x too LOW below ~5 GeV (vertical),
and worse at large angles, which contradicts the data fits shown in the paper.
Using E_tilde = E + Delta (energy at the top of the atmosphere) in that exponent
instead agrees with Reyna, Smith-Duller/Chatzidakis and daemonflux to ~15-30%.
EXPONENT_USES_TILDE selects between them (default True). Above ~30 GeV the two
are identical.
"""
import numpy as np
from muon_kinematics import CM2_TO_M2, scalar_if_scalar

# Eq. (10) parameters: effective zenith angle that accounts for Earth's curvature
P1, P2, P3, P4, P5 = 0.102573, -0.068287, 0.958633, 0.0407253, 0.817285

GAMMA = 2.7      # spectral index
SCALE = 0.14     # overall Gaisser scale factor
EPS_PI = 115.0   # GeV, pion critical energy term
EPS_K = 850.0    # GeV, kaon critical energy term (original Tang/Gaisser value)

EXPONENT_USES_TILDE = True   # see CAUTION in the module docstring


def cos_theta_star(theta):
    x = np.cos(theta)
    num = x**2 + P1**2 + P2 * x**P3 + P4 * x**P5
    den = 1.0 + P1**2 + P2 + P4
    return np.sqrt(num / den)


def tang_modified_gaisser(E, theta):
    E = np.asarray(E, dtype=float)
    x = np.cos(theta)
    cs = cos_theta_star(theta)

    # Segment 3 (E <= 1/cos*): substitute E -> (3E + 7 sec*)/10, then use segment 2
    E_use = np.where(E <= 1.0 / cs, (3.0 * E + 7.0 / cs) / 10.0, E)

    # Segment 2: 1/cos* < E <= 100/cos*
    delta = 2.06e-3 * (950.0 / cs - 90.0)                       # Eq. (5)
    E_tilde_mid = E_use + delta                                  # Eq. (6)
    E_exp = E_tilde_mid if EXPONENT_USES_TILDE else E_use
    A_mid = 1.1 * ((90.0 * np.sqrt(x) + 0.001) / 1030.0) ** (4.5 / (E_exp * cs))  # Eq. (7)
    rc_mid = 1.0e-4                                              # Eq. (4)

    # Segment 1: E > 100/cos*  ->  standard Gaisser
    high = E_use > 100.0 / cs
    A = np.where(high, 1.0, A_mid)
    rc = np.where(high, 0.0, rc_mid)
    E_tilde = np.where(high, E_use, E_tilde_mid)

    bracket = (1.0 / (1.0 + 1.1 * E_tilde * cs / EPS_PI)
               + 0.054 / (1.0 + 1.1 * E_tilde * cs / EPS_K)
               + rc)
    flux_cm2 = A * SCALE * E_use ** (-GAMMA) * bracket
    return scalar_if_scalar(flux_cm2 * CM2_TO_M2)
