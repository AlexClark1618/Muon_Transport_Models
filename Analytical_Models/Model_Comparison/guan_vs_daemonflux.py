"""
guan_vs_daemonflux.py
Compare the Guan model and daemonflux against the SAME experimental data.

Needs (same folder as this script, or edit the paths below):
    guan_model.py                    your Guan et al. parametrisation
    all_data_zenith0_energy.csv      BESS-TeV + L3+C + LVD, near-vertical, in ENERGY units
    all_l3c_by_zenith_energy.csv     (optional) L3+C at six zenith angles, for the angular comparison
    mysite.out                       (optional) PARMA AngOut file; its vertical mu+ + mu- spectrum is overlaid
and   pip install daemonflux matplotlib numpy

Conventions (identical for both models, and for the data)
    E      total muon energy [GeV]
    theta  zenith angle [radians]
    flux   dPhi/dE  [m^-2 s^-1 sr^-1 GeV^-1]   (mu+ + mu-)

Things worth knowing before you read the numbers
  * daemonflux works in momentum and returns p^3 * dPhi/dp in cm^-2 units, so it is converted below.
  * daemonflux is only calibrated for roughly 5 GeV - 10 TeV (below ~5 GeV it ignores geomagnetic effects, above ~10 TeV
    no spectrometer data constrain it).  Results are therefore printed for the full range AND for VALID_RANGE.
  * daemonflux was FITTED to BESS-TeV and L3+C (it even applies correction functions to their systematics), so those two
    are not independent tests of it.  LVD was not used in its fit.
  * daemonflux uses location='generic' here, i.e. no site-specific altitude or atmosphere.  Its height_km option needs a
    spline file with a height grid, which the default install does not have (it raises an error), and its exp= option
    is silently ignored.  BESS-TeV (30 m) and L3+C (450 m) are not at the generic surface, which may shift the
    comparison by a few percent.
  * In its own fit, daemonflux applies correction functions to the BESS-TeV and L3+C systematics.  Here the RAW
    published points are used for both models, so daemonflux is compared to data it was not tuned to match exactly.
  * PARMA AngOut tables are in KINETIC energy [MeV] and /(MeV)/cm2/s/sr, one column per cos(theta).  The cos = +1 column
    (downward vertical) is used, converted to total energy E = T + m_mu [GeV] and /GeV/m2 (dE = dT, so only the units
    change).  For the chi^2 tables PARMA is treated as a third model: its table is interpolated to the data energies
    (log-log in energy, linear in cos(theta)).  It only has muons up to ~0.9 TeV, so PARMA is n/a in any row that has
    points above that; the "PARMA range" rows compare all three models on exactly the same points.
    PARMA is a simulation for the site in its header (lat/lon/altitude/date), not the experiments' sites.
"""
import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Analytical_Models"))     # model files

from guan_model import modified_gaisser_formula
from daemonflux import Flux

# ----------------------------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------------------------
DATA = os.path.join(HERE, "Data")
PLOTS = os.path.join(HERE, "Plots")
VERTICAL_FILE = os.path.join(DATA, "all_data_zenith0_energy.csv")
ZENITH_FILE = os.path.join(DATA, "all_l3c_by_zenith_energy.csv")
POINTS_CSV = os.path.join(DATA, "guan_vs_daemonflux_points.csv")      # per-point table (set to None to skip)
SUMMARY_CSV = os.path.join(DATA, "guan_vs_daemonflux_chi2.csv")       # chi^2 table per experiment (set to None to skip)
ZENITH_CSV = os.path.join(DATA, "guan_vs_daemonflux_chi2_by_zenith.csv")   # chi^2 per zenith bin (set to None to skip)
PARMA_FILE = os.path.join(DATA, "mysite.out")                         # PARMA AngOut file (set to None to skip PARMA)
PARMA_CSV = os.path.join(DATA, "parma_vs_guan_daemonflux.csv")        # PARMA spectrum + both models at its energies

LOCATION = "generic"            # daemonflux location
VALID_RANGE = (5.0, 1.0e4)      # GeV: the range daemonflux was calibrated on
LVD_NORM_ERR = 0.10             # extra fractional error on LVD points (its paper quotes ~10%)
EXTRA_REL_ERR = 0.0             # extra fractional error on ALL points, e.g. 0.05 for a 5% model/systematic floor
WEIGHT_POWER = 0                # 0 = raw dPhi/dE; 2.7 flattens the spectrum (easier to see the band)

M_MU = 0.1056583755             # GeV
DATA_STYLE = {                  # marker per experiment (names must match the 'source' column)
    "BESS-TeV": "o",
    "L3+C": "s",
    "LVD": "D",
}

# ----------------------------------------------------------------------------------------------
# THE TWO MODELS, WITH THE SAME (E, theta) -> dPhi/dE  INTERFACE
# ----------------------------------------------------------------------------------------------
_flux_object = None


def _daemonflux():
    """Create the daemonflux object once (loading takes a few seconds)."""
    global _flux_object
    if _flux_object is None:
        with warnings.catch_warnings():          # newer numpy prints a harmless deprecation warning while loading
            warnings.simplefilter("ignore")
            _flux_object = Flux(location=LOCATION)
    return _flux_object


def _angle_string(theta):
    """daemonflux wants the zenith angle in degrees, as a string, between 0 and 90."""
    deg = float(np.clip(np.degrees(theta), 0.0, 90.0))
    return f"{deg:.4f}"


def _convert(E, theta, quantity_function):
    """Shared conversion: daemonflux (momentum, p^3-weighted, cm^-2)  ->  dPhi/dE in m^-2."""
    scalar_input = np.ndim(E) == 0
    E = np.atleast_1d(np.asarray(E, dtype=float))
    p = np.sqrt(E**2 - M_MU**2)                                   # total energy -> momentum
    scaled = np.atleast_1d(quantity_function(p, _angle_string(theta), "muflux"))   # p^3 dPhi/dp  [cm^-2 ...]
    out = scaled / p**3 * (E / p) * 1.0e4                         # undo p^3, dp -> dE (factor E/p), cm^-2 -> m^-2
    return float(out[0]) if scalar_input else out


def guan_formula(E, theta):
    """Guan et al. 2015 modified Gaisser formula (from guan_model.py)."""
    return modified_gaisser_formula(E, theta)


def daemonflux_formula(E, theta):
    """daemonflux central value of the mu+ + mu- spectrum."""
    return _convert(E, theta, lambda p, ang, q: _daemonflux().flux(p, ang, q))


def daemonflux_error(E, theta):
    """daemonflux 1-sigma uncertainty, in the same units as daemonflux_formula."""
    return _convert(E, theta, lambda p, ang, q: _daemonflux().error(p, ang, q))


_parma_cache = None


def _read_parma_table(file):
    """
    PARMA AngOut file -> (E, cosines, flux, run_info) for mu+ + mu-.
    E       : total energy [GeV]  (PARMA gives kinetic energy in MeV)
    cosines : cos(theta) of every column (+1 = downward vertical)
    flux    : dPhi/dE [m^-2 s^-1 sr^-1 GeV^-1], shape (len(E), len(cosines))  (PARMA gives /(MeV)/cm2/s/sr)
    """
    with open(file) as fh:
        lines = fh.read().splitlines()

    run_info, tables, name, cosines = "", {}, None, None
    for line in lines:
        if line.startswith(" Year"):
            run_info = line.strip()
        elif line.startswith(" Particle name ="):
            name = line.split("=")[1].strip()
            tables[name] = []
        elif line.startswith(" E(MeV/n) /cos"):
            cosines = np.array([float(c) for c in line.split()[2:-1]])   # header row: the cos(theta) of every column
        elif name is not None and line.strip() and not line.startswith(" *"):
            tables[name].append([float(v) for v in line.split()])

    mu_plus, mu_minus = np.array(tables["muplus"]), np.array(tables["mumins"])
    E = mu_plus[:, 0] / 1.0e3 + M_MU                                       # kinetic [MeV] -> total [GeV]
    flux = (mu_plus[:, 1:] + mu_minus[:, 1:]) * 1.0e3 * 1.0e4               # /MeV -> /GeV, /cm2 -> /m2
    return E, cosines, flux, run_info


def parma_formula(E, theta):
    """PARMA mu+ + mu- spectrum, interpolated log-log in energy and linearly in cos(theta). NaN outside PARMA's range."""
    global _parma_cache
    if _parma_cache is None:
        _parma_cache = _read_parma_table(PARMA_FILE)
    Eg, cosines, flux, _ = _parma_cache

    scalar_input = np.ndim(E) == 0
    E = np.atleast_1d(np.asarray(E, dtype=float))

    def column(k):
        ok = flux[:, k] > 0
        out = np.full(len(E), np.nan)
        inside = (E >= Eg[ok].min()) & (E <= Eg[ok].max())
        out[inside] = np.exp(np.interp(np.log(E[inside]), np.log(Eg[ok]), np.log(flux[ok, k])))
        return out

    c = float(np.clip(np.cos(theta), cosines[0], cosines[-1]))
    i = int(np.clip(np.searchsorted(cosines, c) - 1, 0, len(cosines) - 2))     # cosines[i] <= c <= cosines[i + 1]
    w = (c - cosines[i]) / (cosines[i + 1] - cosines[i])
    if np.isclose(w, 0):
        out = column(i)
    elif np.isclose(w, 1):
        out = column(i + 1)
    else:
        out = (1 - w) * column(i) + w * column(i + 1)
    return float(out[0]) if scalar_input else out


MODELS = {"Guan": guan_formula, "daemonflux": daemonflux_formula}
if PARMA_FILE and os.path.exists(PARMA_FILE):
    MODELS["PARMA"] = parma_formula
COLORS = {"Guan": "tab:orange", "daemonflux": "tab:blue", "PARMA": "tab:green"}


# ----------------------------------------------------------------------------------------------
# READING THE DATA
# ----------------------------------------------------------------------------------------------
def read_exp_data(file):
    """all_data_zenith0_energy.csv -> {source: (E, flux, err)}  (floats; the text column is kept as text)."""
    d = np.genfromtxt(file, delimiter=",", names=True, dtype=None, encoding="utf-8")
    data = {}
    for source in dict.fromkeys(d["source"]):
        rows = d["source"] == source
        data[str(source)] = (d["E_GeV"][rows], d["dPhidE_per_m2_s_sr_GeV"][rows], d["err_per_m2_s_sr_GeV"][rows])
    return data


def read_zenith_data(file):
    """all_l3c_by_zenith_energy.csv -> {zenith_deg: (E, flux, err)}."""
    d = np.genfromtxt(file, delimiter=",", names=True)
    bins = {}
    for z in dict.fromkeys(d["zenith_deg"]):
        rows = d["zenith_deg"] == z
        bins[float(z)] = (d["E_GeV"][rows], d["dPhidE_per_m2_s_sr_GeV"][rows], d["err_per_m2_s_sr_GeV"][rows])
    return bins


def read_parma_muons(file, cos_column=1.0):
    """
    PARMA AngOut file -> (E, flux, run_info) for mu+ + mu- at one cos(theta) column (default +1 = downward vertical).
    E    : total energy [GeV]  (PARMA gives kinetic energy in MeV)
    flux : dPhi/dE [m^-2 s^-1 sr^-1 GeV^-1]  (PARMA gives /(MeV)/cm2/s/sr)
    Only energies where PARMA's muon flux is non-zero are returned (the tabulated points, no interpolation).
    """
    E, cosines, flux, run_info = _read_parma_table(file)
    col = int(np.argmin(np.abs(cosines - cos_column)))
    keep = flux[:, col] > 0
    return E[keep], flux[keep, col], run_info


def gather(exp_file):
    """Flatten every experiment into arrays (E, flux, err, source), sorted by energy, with the extra errors applied."""
    E, y, err, src = [], [], [], []
    for source, (x, f, e) in read_exp_data(exp_file).items():
        if source == "LVD":
            e = np.hypot(e, LVD_NORM_ERR * f)
        e = np.hypot(e, EXTRA_REL_ERR * f)
        E.append(x); y.append(f); err.append(e); src.append(np.full(len(x), source))
    E, y, err, src = map(np.concatenate, (E, y, err, src))
    order = np.argsort(E)                       # sort by energy once (the file is grouped by experiment)
    return E[order], y[order], err[order], src[order]


# ----------------------------------------------------------------------------------------------
# 1. PLOT: both models and the data on the same axes, with a ratio panel
# ----------------------------------------------------------------------------------------------
def plot_vertical(exp_file=VERTICAL_FILE, theta=0.0):
    E, y, err, src = gather(exp_file)
    grid = np.logspace(np.log10(0.5), 5, 400)
    w = grid**WEIGHT_POWER

    guan_grid = guan_formula(grid, theta)
    dae_grid = daemonflux_formula(grid, theta)
    dae_err = daemonflux_error(grid, theta)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

    # top: spectra
    ax.plot(grid, guan_grid * w, color=COLORS["Guan"], label="Guan")
    ax.plot(grid, dae_grid * w, color=COLORS["daemonflux"], label="daemonflux")
    ax.fill_between(grid, (dae_grid - dae_err) * w, (dae_grid + dae_err) * w, color=COLORS["daemonflux"], alpha=0.25,
                    label="daemonflux 1 sigma")
    parma = None
    if PARMA_FILE and os.path.exists(PARMA_FILE):
        Ep, fp, run_info = read_parma_muons(PARMA_FILE, cos_column=float(np.cos(theta)))
        on_grid = (Ep >= grid[0]) & (Ep <= grid[-1])             # same energy range as the model curves
        Ep, fp = Ep[on_grid], fp[on_grid]
        parma = Ep, fp
        ax.plot(Ep, fp * Ep**WEIGHT_POWER, ".-", color=COLORS["PARMA"], markersize=5, label="PARMA (simulation)")
        fig.text(0.5, 0.005, f"PARMA: {run_info}", ha="center", fontsize=8)
    for source in dict.fromkeys(src):
        m = src == source
        wd = E[m] ** WEIGHT_POWER
        ax.errorbar(E[m], y[m] * wd, yerr=err[m] * wd, fmt=DATA_STYLE.get(source, "o"), color="k", markersize=4,
                    mfc="k" if DATA_STYLE.get(source, "o") in ("o", "D") else "none", elinewidth=0.8, capsize=2,
                    zorder=5, label=f"{source} (data)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel("dN/dE (m^-2 s^-1 sr^-1 GeV^-1)" if WEIGHT_POWER == 0
                  else f"E^{WEIGHT_POWER:g} dN/dE (m^-2 s^-1 sr^-1 GeV^{WEIGHT_POWER - 1:g})")
    ax.set_title(f"Guan vs daemonflux vs data (Zenith = {np.degrees(theta):.0f} deg)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    # bottom: everything divided by daemonflux; band = daemonflux's own 1 sigma
    axr.axhline(1, color="k", lw=0.8)
    axr.fill_between(grid, 1 - dae_err / dae_grid, 1 + dae_err / dae_grid, color=COLORS["daemonflux"], alpha=0.25)
    axr.plot(grid, guan_grid / dae_grid, color=COLORS["Guan"], label="Guan / daemonflux")
    if parma is not None:
        Ep, fp = parma
        axr.plot(Ep, fp / daemonflux_formula(Ep, theta), ".-", color=COLORS["PARMA"], markersize=5,
                 label="PARMA / daemonflux")
    dae_at_data = daemonflux_formula(E, theta)
    for source in dict.fromkeys(src):
        m = src == source
        axr.errorbar(E[m], y[m] / dae_at_data[m], yerr=err[m] / dae_at_data[m], fmt=DATA_STYLE.get(source, "o"),
                     color="k", markersize=4, mfc="k" if DATA_STYLE.get(source, "o") in ("o", "D") else "none",
                     elinewidth=0.8, capsize=2, zorder=5, label=f"{source} / daemonflux")
    axr.axvspan(VALID_RANGE[0], VALID_RANGE[1], color="gray", alpha=0.07, label="daemonflux calibrated range")
    axr.set_xscale("log"); axr.set_ylim(0.5, 1.6)
    axr.set_xlabel("Muon Energy (GeV)"); axr.set_ylabel("ratio to daemonflux")
    axr.grid(alpha=0.3, which="both"); axr.legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout(rect=[0, 0.02, 1, 1])                # leave room for the PARMA run info at the bottom
    fig.savefig(os.path.join(PLOTS, "guan_vs_daemonflux_vertical.png"), dpi=200)
    plt.show()


# ----------------------------------------------------------------------------------------------
# 2. CHI-SQUARED TABLE: per experiment, for the full range and for daemonflux's calibrated range
# ----------------------------------------------------------------------------------------------
def _short(name):
    """Column label for the printed tables."""
    return {"daemonflux": "dmnflx"}.get(name, name)


def chi_squared_table(exp_file=VERTICAL_FILE, theta=0.0):
    E, y, err, src = gather(exp_file)
    names = list(MODELS)
    pred = {n: np.asarray(f(E, theta), dtype=float) for n, f in MODELS.items()}     # NaN where a model is undefined
    pull = {n: (y - pred[n]) / err for n in names}
    chi = {n: pull[n] ** 2 for n in names}

    sets = [("all data", np.ones(len(E), bool)),
            (f"{VALID_RANGE[0]:g} GeV - {VALID_RANGE[1] / 1e3:g} TeV", (E >= VALID_RANGE[0]) & (E <= VALID_RANGE[1]))]
    if "PARMA" in MODELS:
        covered = np.isfinite(pred["PARMA"])
        sets.append((f"PARMA range ({E[covered].min():.2g} - {E[covered].max():.3g} GeV)", covered))
    sources = list(dict.fromkeys(src))
    summary = []                                # rows for SUMMARY_CSV

    for set_name, in_range in sets:
        print(f"\nchi^2 per point, {set_name}   (zenith = {np.degrees(theta):.0f} deg)")
        print(f"{'experiment':<12}{'n':>4}" + "".join(f"{'chi2 ' + _short(m):>13}" for m in names) + f"{'':>4}"
              + "".join(f"{'d/m ' + _short(m):>13}" for m in names) + "   better  (d/m = mean data/model)")
        for label, sel in [(s, (src == s) & in_range) for s in sources] + [("ALL", in_range)]:
            n = int(sel.sum())
            if n == 0:
                print(f"{label:<12}{0:>4}   (no points in this range)")
                continue
            # a model only gets a number if it is defined at EVERY point of the row (PARMA stops at ~0.9 TeV)
            ok = [m for m in names if np.all(np.isfinite(pred[m][sel]))]
            c = {m: chi[m][sel].mean() if m in ok else np.nan for m in names}
            r = {m: np.exp(np.mean(np.log(y[sel] / pred[m][sel]))) if m in ok else np.nan for m in names}  # geometric mean
            better = min(ok, key=c.get)
            summary.append(f"{set_name},{label},{n}," + ",".join(f"{c[m]:.6g}" for m in names) + ","
                           + ",".join(f"{r[m]:.6g}" for m in names) + f",{better}")
            print(f"{label:<12}{n:>4}" + "".join(f"{c[m]:>13.3g}" if m in ok else f"{'n/a':>13}" for m in names)
                  + f"{'':>4}" + "".join(f"{r[m]:>13.3f}" if m in ok else f"{'n/a':>13}" for m in names) + f"   {better}")
    print("\nnote: errors are treated as independent (systematics are really correlated), so use the numbers to RANK models.")
    print("      BESS-TeV and L3+C were used to fit daemonflux, so they are not independent tests of it.")
    if "PARMA" in MODELS:
        print("      PARMA is n/a in rows with points above its highest muon energy; compare all three in the PARMA range rows.")

    if POINTS_CSV:
        with open(POINTS_CSV, "w") as fh:
            fh.write("source,E_GeV,flux,err," + ",".join(f"{m}_model,{m}_chi2" for m in names) + ",in_daemonflux_range\n")
            for i in range(len(E)):
                ok = int(VALID_RANGE[0] <= E[i] <= VALID_RANGE[1])
                fh.write(f"{src[i]},{E[i]:.6e},{y[i]:.6e},{err[i]:.6e},"
                         + ",".join(f"{pred[m][i]:.6e},{chi[m][i]:.6e}" for m in names) + f",{ok}\n")
        print(f"saved {POINTS_CSV}")
    if SUMMARY_CSV:
        with open(SUMMARY_CSV, "w") as fh:
            fh.write("energy_range,experiment,n_points," + ",".join(f"{m}_chi2_per_pt" for m in names) + ","
                     + ",".join(f"{m}_mean_data_over_model" for m in names) + ",better\n")
            fh.write("\n".join(summary) + "\n")
        print(f"saved {SUMMARY_CSV}")
    return E, y, err, src, pred, chi


# ----------------------------------------------------------------------------------------------
# 3. ANGULAR COMPARISON with the six L3+C zenith bins
# ----------------------------------------------------------------------------------------------
def chi_squared_by_zenith(zenith_file=ZENITH_FILE, make_plot=True):
    """
    chi^2 per point and mean data/model in every L3+C zenith bin.
    Every model is evaluated on the same points: those inside the energy range of ALL models (i.e. PARMA's, if loaded),
    so the bins lose their highest-energy points when PARMA is included.
    """
    bins = read_zenith_data(zenith_file)
    names = list(MODELS)
    zen = sorted(bins)
    chi = {n: [] for n in names}
    ratio = {n: [] for n in names}
    n_used = []
    print("\nL3+C at several zenith angles: chi^2 per point and mean data/model (points where every model is defined)")
    print(f"{'zenith':>8}{'n':>4}" + "".join(f"{_short(m) + ' chi2':>13}" for m in names)
          + "".join(f"{_short(m) + ' d/m':>13}" for m in names))
    for z in zen:
        E, y, err = bins[z]
        pred = {n: np.asarray(MODELS[n](E, np.radians(z)), dtype=float) for n in names}
        sel = np.all([np.isfinite(pred[n]) for n in names], axis=0)
        e = np.hypot(err[sel], EXTRA_REL_ERR * y[sel])
        for n in names:
            chi[n].append(np.mean(((y[sel] - pred[n][sel]) / e) ** 2))
            ratio[n].append(np.exp(np.mean(np.log(y[sel] / pred[n][sel]))))
        n_used.append(int(sel.sum()))
        print(f"{z:>8.1f}{n_used[-1]:>4}" + "".join(f"{chi[n][-1]:>13.3g}" for n in names)
              + "".join(f"{ratio[n][-1]:>13.3f}" for n in names))
    print(f"{'average':>8}{'':>4}" + "".join(f"{np.mean(chi[n]):>13.3g}" for n in names)
          + "".join(f"{np.mean(ratio[n]):>13.3f}" for n in names))

    if ZENITH_CSV:
        with open(ZENITH_CSV, "w") as fh:
            fh.write("zenith_deg,n_points," + ",".join(f"{n}_chi2_per_pt" for n in names) + ","
                     + ",".join(f"{n}_mean_data_over_model" for n in names) + "\n")
            for i, z in enumerate(zen):
                fh.write(f"{z:.2f},{n_used[i]}," + ",".join(f"{chi[n][i]:.6g}" for n in names) + ","
                         + ",".join(f"{ratio[n][i]:.6g}" for n in names) + "\n")
            fh.write("average,," + ",".join(f"{np.mean(chi[n]):.6g}" for n in names) + ","
                     + ",".join(f"{np.mean(ratio[n]):.6g}" for n in names) + "\n")
        print(f"saved {ZENITH_CSV}")

    if make_plot:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
        for n in MODELS:
            a1.plot(zen, chi[n], "o-", color=COLORS[n], label=n)
            a2.plot(zen, ratio[n], "o-", color=COLORS[n], label=n)
        a1.axhline(1, color="k", lw=0.8, ls="--"); a1.set_yscale("log")
        a1.set_xlabel("Zenith angle (deg)"); a1.set_ylabel("chi^2 per data point"); a1.set_title("chi^2 vs L3+C")
        a2.axhline(1, color="k", lw=0.8)
        a2.set_xlabel("Zenith angle (deg)"); a2.set_ylabel("mean data / model"); a2.set_title("bias (1 = unbiased)")
        for a in (a1, a2):
            a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, "guan_vs_daemonflux_by_zenith.png"), dpi=200)
        plt.show()
    return zen, chi, ratio


# ----------------------------------------------------------------------------------------------
# 4. PARMA SIMULATION vs the two models: ratios at PARMA's own tabulated energies
# ----------------------------------------------------------------------------------------------
def parma_comparison(parma_file=PARMA_FILE, theta=0.0, E_min=0.5):
    """Ratio PARMA / Guan and PARMA / daemonflux at PARMA's own energies (E >= E_min GeV), printed per decade and saved."""
    E, f, run_info = read_parma_muons(parma_file, cos_column=float(np.cos(theta)))
    sel = E >= E_min
    E, f = E[sel], f[sel]
    guan = guan_formula(E, theta)
    dae = daemonflux_formula(E, theta)

    print(f"\nPARMA vertical mu+ + mu-   ({run_info})")
    print(f"{'decade [GeV]':>16}{'n':>4}{'PARMA/Guan':>12}{'PARMA/dmnflx':>14}   (geometric mean of the ratio)")
    decade = np.floor(np.log10(E)).astype(int)
    for d in np.unique(decade):
        m = decade == d
        print(f"{10.0**d:>7.3g} - {10.0**(d + 1):<7.3g}{m.sum():>4}"
              f"{np.exp(np.mean(np.log(f[m] / guan[m]))):>12.3f}{np.exp(np.mean(np.log(f[m] / dae[m]))):>14.3f}")
    print(f"{'all':>16}{len(E):>4}{np.exp(np.mean(np.log(f / guan))):>12.3f}{np.exp(np.mean(np.log(f / dae))):>14.3f}")

    if PARMA_CSV:
        with open(PARMA_CSV, "w") as fh:
            fh.write(f"# PARMA: {run_info}\n")
            fh.write("E_GeV,PARMA_dPhidE,Guan_dPhidE,daemonflux_dPhidE,PARMA_over_Guan,PARMA_over_daemonflux\n")
            for i in range(len(E)):
                fh.write(f"{E[i]:.6e},{f[i]:.6e},{guan[i]:.6e},{dae[i]:.6e},{f[i] / guan[i]:.6g},{f[i] / dae[i]:.6g}\n")
        print(f"saved {PARMA_CSV}")
    return E, f, guan, dae


def main():
    plot_vertical()
    chi_squared_table()
    if PARMA_FILE and os.path.exists(PARMA_FILE):
        parma_comparison()
    if os.path.exists(ZENITH_FILE):
        chi_squared_by_zenith()
    else:
        print(f"\n(skipping the angular comparison: {ZENITH_FILE} not found)")


if __name__ == "__main__":
    main()
