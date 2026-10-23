"""
muon_spectra.py
===============
Surface muon (mu+ + mu-) spectra from daemonflux.

What this script does
---------------------
1. Computes the DIFFERENTIAL muon energy spectrum dPhi/dE for total energies
   1 GeV to 100 TeV and zenith angles 0 to 90 degrees in 5-degree steps.
2. Saves them to CSV files (spectra + 1-sigma errors + a small info file).
3. Plots (a) the raw vertical differential spectrum and
         (b) the energy-integrated vertical spectrum, I(>E).

Units of everything saved/plotted
---------------------------------
  dPhi/dE : cm^-2 s^-1 sr^-1 GeV^-1
  I(>E)   : cm^-2 s^-1 sr^-1

How the conversion from daemonflux works
----------------------------------------
daemonflux returns muon fluxes as a function of total MOMENTUM p, already
multiplied by p^3. So for each energy E we:
  1. convert to momentum:      p = sqrt(E^2 - m^2)         (natural units, c = 1)
  2. ask daemonflux at p:      scaled = flux(p)            (= p^3 * dPhi/dp)
  3. undo the p^3 scaling:     dPhi/dp = scaled / p^3
  4. change variable to E:     dPhi/dE = dPhi/dp * (dp/dE) = dPhi/dp * (E / p)

Run:  python muon_spectra.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from daemonflux import Flux

# ----------------------------------------------------------------------------
# SETTINGS: change these to suit
# ----------------------------------------------------------------------------
E_MIN = 1.0            # GeV, lowest total muon energy
E_MAX = 1.0e5          # GeV, highest total muon energy (100 TeV)
N_POINTS = 500         # points in the log-spaced energy grid (~100 per decade)
ANGLES = list(range(0, 91, 5))   # zenith angles in degrees: 0, 5, ..., 90
QUANTITY = 'muflux'    # 'muflux' = mu+ + mu-.  Use 'mu+' or 'mu-' for one charge,
                       # or 'total_muflux' to add the (uncalibrated) prompt part
LOCATION = 'generic'
OUTDIR = '.'           # where files are written
MUON_MASS = 0.1056583755   # GeV
# ----------------------------------------------------------------------------

os.makedirs(OUTDIR, exist_ok=True)

# --- Step 1: energy grid and the matching momentum grid -----------------------
E = np.logspace(np.log10(E_MIN), np.log10(E_MAX), N_POINTS)   # GeV
p = np.sqrt(E**2 - MUON_MASS**2)                              # GeV/c
dp_dE = E / p                                                 # Jacobian dp/dE


def dphi_dE(fx, zenith_deg, error=False):
    """Differential flux (or its 1-sigma error) per unit energy, in cm^-2 s^-1 sr^-1 GeV^-1."""
    if error:
        scaled = fx.error(p, str(zenith_deg), QUANTITY)
    else:
        scaled = fx.flux(p, str(zenith_deg), QUANTITY)
    return 10000 * scaled / p**3 * dp_dE


# --- Step 2: compute the spectrum at every angle ------------------------------
fx = Flux(location=LOCATION)

flux_cols, err_cols = [], []
for ang in ANGLES:
    flux_cols.append(dphi_dE(fx, ang))
    err_cols.append(dphi_dE(fx, ang, error=True))
    print(f"computed zenith = {ang:2d} deg")

# --- Step 3: save to CSV -------------------------------------------------------
header = "E_GeV,p_GeVc," + ",".join(f"zen_{a}deg" for a in ANGLES)

spectra = np.column_stack([E, p] + flux_cols)
errors = np.column_stack([E, p] + err_cols)

spectra_file = os.path.join(OUTDIR, "muon_dphidE_spectra.csv")
errors_file = os.path.join(OUTDIR, "muon_dphidE_errors.csv")
np.savetxt(spectra_file, spectra, delimiter=",", header=header, comments="", fmt="%.6e")
np.savetxt(errors_file, errors, delimiter=",", header=header, comments="", fmt="%.6e")

info_file = os.path.join(OUTDIR, "muon_spectra_info.txt")
with open(info_file, "w") as f:
    f.write("Muon spectra from daemonflux\n")
    f.write(f"Quantity : {QUANTITY}  (location = {LOCATION})\n")
    f.write("Columns  : E_GeV = total muon energy, p_GeVc = total momentum,\n")
    f.write("           zen_XXdeg = dPhi/dE at that zenith angle\n")
    f.write("Units    : dPhi/dE in cm^-2 s^-1 sr^-1 GeV^-1 (raw, NOT multiplied by E^3)\n")
    f.write("Errors   : 1-sigma, same units, in muon_dphidE_errors.csv\n")

print(f"\nSaved: {spectra_file}\n       {errors_file}\n       {info_file}")

# --- Step 4: plot the raw vertical differential spectrum ----------------------
i0 = ANGLES.index(0)                    # column index for vertical (0 deg)
vert = flux_cols[i0]
vert_err = err_cols[i0]

fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.loglog(E, vert, color='k', label=r'vertical ($0^\circ$)')
ax1.fill_between(E, np.clip(vert - vert_err, 1e-300, None), vert + vert_err,
                 color='r', alpha=0.3, label=r'1$\sigma$ error')
ax1.set_xlabel('Muon energy E [GeV]')
ax1.set_ylabel(r'$d\Phi/dE$  [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$]')
ax1.set_title('Vertical differential muon spectrum (raw)')
ax1.legend()
ax1.grid(True, which='both', alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(OUTDIR, "vertical_differential_spectrum.png"), dpi=200)

# --- Step 5: energy-integrated vertical spectrum  I(>E) -----------------------
# I(>E) = integral from E to E_MAX of dPhi/dE' dE'.
# Integrating in ln(E) is more accurate on a log grid: dE = E dlnE.
lnE = np.log(E)


def integral_above(spectrum):
    """Cumulative integral from each E up to E_MAX."""
    cum_from_low = cumulative_trapezoid(spectrum * E, lnE, initial=0.0)
    return cum_from_low[-1] - cum_from_low


I_vert = integral_above(vert)
I_lo = integral_above(np.clip(vert - vert_err, 0, None))   # rough band: errors treated
I_hi = integral_above(vert + vert_err)                     # as fully correlated

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.loglog(E[:-1], I_vert[:-1], color='k', label=r'vertical ($0^\circ$)')
ax2.fill_between(E[:-1], I_lo[:-1], I_hi[:-1], color='r', alpha=0.3,
                 label=r'approx. 1$\sigma$ band (errors fully correlated)')
ax2.set_xlabel('Muon energy E [GeV]')
ax2.set_ylabel(r'$I(>E)$  [cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
ax2.set_title(f'Vertical integral muon spectrum  (E up to {E_MAX:.0e} GeV)')
ax2.legend()
ax2.grid(True, which='both', alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTDIR, "vertical_integral_spectrum.png"), dpi=200)

print(f"\nVertical integrated flux above {E_MIN:g} GeV: {I_vert[0]:.3e} cm^-2 s^-1 sr^-1")
print("Saved: vertical_differential_spectrum.png, vertical_integral_spectrum.png")

plt.show()