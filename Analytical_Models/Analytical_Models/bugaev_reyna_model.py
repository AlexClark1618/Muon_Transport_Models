"""
bugaev_reyna_model.py  -  Bugaev (vertical only) and Reyna (all angles)

Sources : Bugaev et al., Phys. Rev. D 58, 054001 (1998)   -> vertical flux, Table 1
          Reyna, arXiv:hep-ph/0604145 (2006)              -> all zenith angles
          (parameters as tabulated in Su et al., Front. Energy Res. 9, 750159)
Both models are written in MOMENTUM p [GeV/c]; E is converted with muon_kinematics.
Reyna is recommended for 1 GeV <= p <= 2000/cos(theta) GeV and 0 <= theta <= 90 deg.
Units   : returns dPhi/dE in m^-2 s^-1 sr^-1 GeV^-1
"""
import numpy as np
from muon_kinematics import momentum, dp_dE, CM2_TO_M2, scalar_if_scalar


def _power_law(p, A, a0, a1, a2, a3):
    """Phi(p) = A * p^-(a3 y^3 + a2 y^2 + a1 y + a0),  y = log10(p)   [cm^-2 s^-1 sr^-1 (GeV/c)^-1]"""
    y = np.log10(p)
    return A * p ** (-(a3 * y**3 + a2 * y**2 + a1 * y + a0))


def bugaev_formula(E, theta=0.0):
    """VERTICAL flux only: theta is ignored. Piecewise parameters from Table 1."""
    p = momentum(E)
    conds = [p < 9.2765e2, p < 1.5878e3, p < 4.1625e5]
    choices = [
        _power_law(p, 2.950e-3, 0.3061, 1.2743, -0.2630, 0.0252),
        _power_law(p, 1.781e-2, 1.7910, 0.3040, 0.0, 0.0),
        _power_law(p, 1.435e1, 3.6720, 0.0, 0.0, 0.0),
    ]
    flux_p = np.select(conds, choices, default=_power_law(p, 1.0e3, 4.0, 0.0, 0.0, 0.0))
    return scalar_if_scalar(flux_p * dp_dE(E) * CM2_TO_M2)


def reyna_formula(E, theta):
    """Reyna: Phi_R(p, theta) = cos^3(theta) * Phi_B(p cos(theta)) with Reyna's refit parameters."""
    p = momentum(E)
    c = np.cos(theta)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        flux_p = c**3 * _power_law(p * c, 0.00253, 0.2455, 1.288, -0.2555, 0.0209)
    flux_p = np.nan_to_num(flux_p, nan=0.0, posinf=0.0)
    return scalar_if_scalar(flux_p * dp_dE(E) * CM2_TO_M2)
