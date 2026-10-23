"""
compare_to_data.py
Overlay measured sea-level muon fluxes on every analytic model.

Reads the ENERGY tables in ./data_energy/ (made by make_energy_tables.py from the published momentum tables in ./data/).
Every table:  E_GeV (total muon energy), dPhidE (raw differential flux, m^-2 s^-1 sr^-1 GeV^-1), err (1 sigma).

DATA SETS
  bess_tev_zenith0_energy.csv  BESS-TeV, Haino et al. 2004.         0.6-400 GeV/c, sea level, near-vertical, mu+ + mu-
  l3c_zenith0_energy.csv       L3+C, Achard et al. 2004, Table 2.   cos(theta) 0.975-1.0, 20-3000 GeV/c, 450 m altitude
  lvd_zenith0_energy.csv       LVD, Aglietta et al. 1999, Table II. vertical, 1.9-43 TeV, DERIVED from underground data
  l3c_zenith56_energy.csv      L3+C, Achard et al. 2004, Table 9.   cos(theta) 0.525-0.6 (about 56 deg)

CAUTION about LVD: not a direct measurement. LVD scaled a Gaisser-shaped spectrum (fitted gamma=2.78, A=1.95) to match the
muon intensity underground, so the shape is partly built in. The paper quotes an extra ~10% absolute-normalisation error,
added here in quadrature (LVD_NORM_ERR). L3+C near-vertical error bars are systematics only.

Run:  python compare_to_data.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Analytical_Models"))     # model files

from gaisser_model import gaisser_formula
from guan_model import modified_gaisser_formula
from tang_model import tang_modified_gaisser
from bugaev_reyna_model import reyna_formula
from smith_duller_model import smith_duller_chatzidakis

DATA = os.path.join(HERE, "Data")
PLOTS = os.path.join(HERE, "Plots")

MODELS = {
    "Gaisser": gaisser_formula,
    "Guan": modified_gaisser_formula,
    "Gaisser/Tang": tang_modified_gaisser,
    "Bugaev/Reyna": reyna_formula,
    "Smith & Duller/Chatzidakis": smith_duller_chatzidakis,
}

LVD_NORM_ERR = 0.10   # extra absolute-normalisation uncertainty quoted in the LVD paper
WEIGHT_POWER = 2.7    # top panels show E^WEIGHT_POWER * dPhi/dE  (set to 0 for the raw spectrum on a log axis)


def load(filename, label, theta_deg, marker, extra_rel_err=0.0):
    d = np.genfromtxt(os.path.join(DATA, filename), delimiter=",", names=True)
    flux = d["dPhidE_per_m2_s_sr_GeV"]
    err = np.hypot(d["err_per_m2_s_sr_GeV"], extra_rel_err * flux)
    return dict(label=label, E=d["E_GeV"], flux=flux, err=err, theta=np.radians(theta_deg), marker=marker)


def main():
    bess = load("bess_tev_zenith0_energy.csv", "BESS-TeV (sea level)", 0.0, "o")
    l3c = load("l3c_zenith0_energy.csv", "L3+C, cos(theta) 0.975-1.0", 0.0, "s")
    lvd = load("lvd_zenith0_energy.csv", "LVD, derived from depth-intensity (indirect)", 0.0, "D", LVD_NORM_ERR)
    l3c56 = load("l3c_zenith56_energy.csv", "L3+C, cos(theta) 0.525-0.6", np.degrees(np.arccos(0.5625)), "^")

    groups = [("Zenith = 0 (near-vertical)", [bess, l3c, lvd]),
              (f"Inclined (~{np.degrees(l3c56['theta']):.0f} deg)", [l3c56])]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col", gridspec_kw={"height_ratios": [2.2, 1]})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    Egrid = np.logspace(np.log10(0.7), 5, 600)

    for col, (title, datasets) in enumerate(groups):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        theta = datasets[0]["theta"]
        for i, (name, model) in enumerate(MODELS.items()):
            ax_top.plot(Egrid, model(Egrid, theta) * Egrid**WEIGHT_POWER, color=colors[i], label=name)
        for ds in datasets:
            w = ds["E"]**WEIGHT_POWER
            ax_top.errorbar(ds["E"], ds["flux"] * w, yerr=ds["err"] * w, fmt=ds["marker"], ms=4, color="k",
                            mfc="k" if ds["marker"] in ("o", "D") else "none", lw=0.8, label=ds["label"], zorder=5)
            for i, model in enumerate(MODELS.values()):
                ax_bot.plot(ds["E"], model(ds["E"], ds["theta"]) / ds["flux"], ds["marker"] + "-", ms=3, lw=0.8, color=colors[i])
        ax_top.set_xscale("log")
        ax_top.set_yscale("log" if WEIGHT_POWER == 0 else "linear")
        ax_top.set_title(f"{title}  (model theta = {np.degrees(theta):.1f} deg)")
        ax_top.set_ylabel(f"E^{WEIGHT_POWER:g} dPhi/dE  [m^-2 s^-1 sr^-1 GeV^{WEIGHT_POWER - 1:g}]")
        ax_top.legend(fontsize=8)
        ax_top.grid(alpha=0.3, which="both")
        ax_bot.axhline(1, color="k", lw=0.8)
        ax_bot.set_yscale("log")
        ax_bot.set_ylim(0.2, 5)
        ax_bot.set_xlabel("Muon energy E [GeV]")
        ax_bot.set_ylabel("model / data")
        ax_bot.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "models_vs_data.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
