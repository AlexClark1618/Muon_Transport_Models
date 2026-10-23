"""
muon_kinematics.py
Small helpers shared by the momentum-based models (Bugaev/Reyna).

Convention used by every model file in this project
---------------------------------------------------
E     : total muon energy at sea level [GeV]
theta : zenith angle [radians]
return: dPhi/dE  in  m^-2 s^-1 sr^-1 GeV^-1   (same units as gaisser_model.py)
"""
import numpy as np

MUON_MASS = 0.1056583755   # GeV/c^2
CM2_TO_M2 = 1.0e4          # 1 cm^-2 = 1e4 m^-2


def momentum(E):
    """Total energy E [GeV] -> momentum p [GeV/c]."""
    E = np.asarray(E, dtype=float)
    return np.sqrt(np.maximum(E**2 - MUON_MASS**2, 0.0))


def dp_dE(E):
    """Jacobian dp/dE = E/p. Multiply dPhi/dp by this to get dPhi/dE."""
    E = np.asarray(E, dtype=float)
    return E / momentum(E)


def scalar_if_scalar(x):
    """Return a plain float for scalar input (scipy.integrate.quad likes that)."""
    return x.item() if np.ndim(x) == 0 else x
