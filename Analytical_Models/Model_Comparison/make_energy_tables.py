"""
make_energy_tables.py
Convert the data tables (as published, in MOMENTUM) into ENERGY tables, and tabulate the analytic models at zenith = 0.

Every output table uses the same convention
    E_GeV                   total muon energy  E = sqrt(p^2 + m_mu^2)   [GeV]
    dPhidE_per_m2_s_sr_GeV  RAW differential flux dPhi/dE               [m^-2 s^-1 sr^-1 GeV^-1]   (no E^n or p^3 weighting)
    err_per_m2_s_sr_GeV     1-sigma uncertainty on that flux (same units)

Conversion of a flux measured per unit momentum:   dPhi/dE = dPhi/dp * (dp/dE) = dPhi/dp * E/p     (c = 1)
L3+C is published as  Phi * <p>^3  [GeV^2 cm^-2 s^-1 sr^-1]  ->  divide by <p>^3, multiply by 1e4 (cm^-2 -> m^-2).

Inputs : ./Data/published/*.csv   (as published)
Outputs: ./Data/*.csv
Run    : python make_energy_tables.py        (model files are read from ../Analytical_Models for models_zenith0_energy.csv)
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Analytical_Models"))     # model files
IN, OUT = os.path.join(HERE, "Data", "published"), os.path.join(HERE, "Data")
os.makedirs(OUT, exist_ok=True)
M_MU = 0.1056583755   # GeV
HDR = "E_GeV,dPhidE_per_m2_s_sr_GeV,err_per_m2_s_sr_GeV"


def to_energy(p, dphi_dp, err_dp):
    E = np.sqrt(p**2 + M_MU**2)
    return E, dphi_dp * E / p, err_dp * E / p


def save(name, E, f, e):
    np.savetxt(os.path.join(OUT, name), np.column_stack([E, f, e]), delimiter=",", fmt="%.6e", header=HDR, comments="")
    return np.column_stack([E, f, e])


# ---- BESS-TeV: add mu+ and mu- (stat in quadrature, sys linearly), then total error in quadrature ----
d = np.genfromtxt(os.path.join(IN, "bess_tev_sea_level.csv"), delimiter=",", names=True)
p = 0.5 * (d["p_plus_GeVc"] + d["p_minus_GeVc"])
flux = d["flux_plus"] + d["flux_minus"]
err = np.hypot(np.hypot(d["sta_plus"], d["sta_minus"]), d["sys_plus"] + d["sys_minus"])
bess = save("bess_tev_zenith0_energy.csv", *to_energy(p, flux, err))

# ---- L3+C (two zenith bins) ----
def l3c(infile, outfile):
    d = np.genfromtxt(os.path.join(IN, infile), delimiter=",", names=True)
    p = d["p_mean_GeV"]
    f = d["flux_x_p3"] / p**3 * 1.0e4
    rel = np.hypot(np.nan_to_num(d["stat_pct"]), d["syst_pct"]) / 100.0     # stat is blank (nan) for the vertical bin
    return save(outfile, *to_energy(p, f, f * rel))

l3c_v = l3c("l3c_vertical_cos0975_1000.csv", "l3c_zenith0_energy.csv")
l3c_56 = l3c("l3c_cos0525_0600.csv", "l3c_zenith56_energy.csv")      # NOT zenith 0: kept separately

# ---- LVD: already a sea-level energy spectrum, only cm^-2 -> m^-2 ----
d = np.genfromtxt(os.path.join(IN, "lvd_sea_level_vertical.csv"), delimiter=",", names=True)
lvd = save("lvd_zenith0_energy.csv", d["E_GeV"], d["flux_cm2_s_sr_GeV"] * 1e4, d["err"] * 1e4)

# ---- one combined table of everything that is (near-)vertical ----
names = ["BESS-TeV", "L3+C", "LVD"]
with open(os.path.join(OUT, "all_data_zenith0_energy.csv"), "w") as fh:
    fh.write("source,E_GeV,dPhidE_per_m2_s_sr_GeV,err_per_m2_s_sr_GeV\n")
    for nm, tab in zip(names, [bess, l3c_v, lvd]):
        for E, f, e in tab:
            fh.write(f"{nm},{E:.6e},{f:.6e},{e:.6e}\n")


# ---- L3+C at several zenith angles: one energy table per bin + one combined multi-angle table ----
# (bins whose tables extracted cleanly; the paper's Tables 3 and 7 are not included)
L3C_BINS = [  # (input file, cos_lo, cos_hi, output file)
    ("l3c_vertical_cos0975_1000.csv", 0.975, 1.000, None),                    # already written above
    ("l3c_cos0900_0938.csv",          0.900, 0.938, "l3c_cos0900_0938_energy.csv"),
    ("l3c_cos0825_0900.csv",          0.825, 0.900, "l3c_cos0825_0900_energy.csv"),
    ("l3c_cos0750_0825.csv",          0.750, 0.825, "l3c_cos0750_0825_energy.csv"),
    ("l3c_cos0600_0675.csv",          0.600, 0.675, "l3c_cos0600_0675_energy.csv"),
    ("l3c_cos0525_0600.csv",          0.525, 0.600, None),                    # already written above (l3c_zenith56_energy.csv)
]
with open(os.path.join(OUT, "all_l3c_by_zenith_energy.csv"), "w") as fh:
    fh.write("cos_lo,cos_hi,zenith_deg,E_GeV,dPhidE_per_m2_s_sr_GeV,err_per_m2_s_sr_GeV\n")
    for infile, lo, hi, outfile in L3C_BINS:
        d = np.genfromtxt(os.path.join(IN, infile), delimiter=",", names=True)
        p = d["p_mean_GeV"]
        f = d["flux_x_p3"] / p**3 * 1.0e4
        rel = np.hypot(np.nan_to_num(d["stat_pct"]), d["syst_pct"]) / 100.0
        E, F, Err = to_energy(p, f, f * rel)
        if outfile:
            save(outfile, E, F, Err)
        zen = np.degrees(np.arccos(0.5 * (lo + hi)))                 # bin-centre zenith angle
        for e_, f_, r_ in zip(E, F, Err):
            fh.write(f"{lo},{hi},{zen:.2f},{e_:.6e},{f_:.6e},{r_:.6e}\n")

# ---- analytic models at zenith = 0 on a common log grid ----
try:
    from gaisser_model import gaisser_formula
    from guan_model import modified_gaisser_formula
    from tang_model import tang_modified_gaisser
    from bugaev_reyna_model import bugaev_formula, reyna_formula
    from smith_duller_model import smith_duller_chatzidakis
    E = np.logspace(0, 5, 201)                     # 1 GeV ... 100 TeV
    cols = {"Gaisser": gaisser_formula, "Guan": modified_gaisser_formula, "Gaisser_Tang": tang_modified_gaisser,
            "Bugaev": bugaev_formula, "Reyna": reyna_formula, "SmithDuller_Chatzidakis": smith_duller_chatzidakis}
    table = np.column_stack([E] + [np.atleast_1d(f(E, 0.0)) for f in cols.values()])
    np.savetxt(os.path.join(OUT, "models_zenith0_energy.csv"), table, delimiter=",", fmt="%.6e",
               header="E_GeV," + ",".join(f"{k}_dPhidE_per_m2_s_sr_GeV" for k in cols), comments="")
except ImportError as exc:
    print("skipped models table (model files not found):", exc)

print("written to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
