import os, sys
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "Data")
PLOTS_DIR = os.path.join(HERE, "Plots")
sys.path.insert(0, os.path.join(HERE, "..", "Analytical_Models"))     # model files

from gaisser_model import gaisser_formula
from guan_model import modified_gaisser_formula
from tang_model import tang_modified_gaisser
from bugaev_reyna_model import reyna_formula          # bugaev_formula is vertical-only
from smith_duller_model import smith_duller_chatzidakis

# One marker/colour per experiment (names must match the 'source' column of the CSV)
DATA_STYLE = {
    "BESS-TeV": dict(fmt="o", color="black"),
    "L3+C":     dict(fmt="s", color="dimgray"),
    "LVD":      dict(fmt="D", color="firebrick"),
}


def read_exp_data(file):
    """
    Read all_data_zenith0_energy.csv and return a dict  {source: (E, flux, err)}.

    Columns: source, E_GeV, dPhidE_per_m2_s_sr_GeV, err_per_m2_s_sr_GeV
    np.genfromtxt keeps the text column as text and the other columns as real floats.
    (Putting a text column and numbers into one np.array turns EVERYTHING into strings,
    and matplotlib then plots strings as evenly spaced categories instead of numbers.)
    """
    d = np.genfromtxt(file, delimiter=",", names=True, dtype=None, encoding="utf-8")

    data = {}
    for source in dict.fromkeys(d["source"]):          # unique sources, in file order
        rows = d["source"] == source
        data[str(source)] = (d["E_GeV"][rows],
                             d["dPhidE_per_m2_s_sr_GeV"][rows],
                             d["err_per_m2_s_sr_GeV"][rows])
    return data


def _gather_data(exp_file, lvd_norm_err=0.10, extra_rel_err=0.0):
    """All experiments -> four flat arrays (E, flux, err, source), with optional extra errors added in quadrature."""
    E, y, err, src = [], [], [], []
    for source, (x, f, e) in read_exp_data(exp_file).items():
        if source == "LVD":
            e = np.hypot(e, lvd_norm_err * f)     # LVD paper quotes ~10% extra absolute-normalisation error
        e = np.hypot(e, extra_rel_err * f)        # optional model/systematic floor for ALL points
        E.append(x); y.append(f); err.append(e); src.append(np.full(len(x), source))
    return tuple(map(np.concatenate, (E, y, err, src)))


def differential_spectra(models, energy_range, theta, exp_file=None, weight_power=0):
    """
    Plot every model's differential spectrum and, if exp_file is given, the measured
    data with error bars on the SAME axes.

    weight_power = 0   -> raw dPhi/dE
    weight_power = 2.7 -> E^2.7 * dPhi/dE  (flattens the spectrum so differences are easier to see)
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    E = np.asarray(energy_range, dtype=float)

    # models
    for label, model in models.items():
        diff_spectrum = model(E, theta)               # all the model files accept arrays
        ax.plot(E, diff_spectrum * E**weight_power, label=label)

    # experimental data (same axes, so it must be drawn BEFORE plt.show())
    if exp_file is not None:
        for source, (x, y, err) in read_exp_data(exp_file).items():
            style = DATA_STYLE.get(source, dict(fmt="o", color="gray"))
            w = x**weight_power
            ax.errorbar(x, y * w, yerr=err * w, fmt=style["fmt"], color=style["color"],
                        markersize=4, elinewidth=0.8, capsize=2, zorder=5,
                        label=f"{source} (data)")

    ax.set_xlabel("Muon Energy (GeV)")
    if weight_power == 0:
        ax.set_ylabel("dN/dE (m^-2 s^-1 sr^-1 GeV^-1)")
    else:
        ax.set_ylabel(f"E^{weight_power:g} dN/dE (m^-2 s^-1 sr^-1 GeV^{weight_power - 1:g})")
    ax.set_title(f"Differential Muon Spectrum (Zenith = {theta})")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(PLOTS_DIR, f"differential_spectra_zenith{theta}.png"), dpi=200)
    plt.show()


def chi_squared_by_energy(models, exp_file, theta=0,
                          ref_energies=(1, 1e1, 1e2, 1e3, 1e4, 1e5), half_width=np.sqrt(10),
                          lvd_norm_err=0.10, extra_rel_err=0.0, min_points=1, save_csv=None, make_plot=True):
    """
    Chi-squared of every model against the experimental data, in log-energy windows.

    For each reference energy E_ref the window is [E_ref/half_width, E_ref*half_width), i.e. +/- half a decade
    by default, so the windows touch and every data point is used exactly once.
    Inside a window:   chi2 = sum_i ((data_i - model(E_i)) / err_i)^2      (models are evaluated at the data energies,
    so the data are never interpolated).  chi2/n is printed, plus the mean model/data ratio (is the model high or low?).

    lvd_norm_err : extra fractional error added (in quadrature) to the LVD points (its paper quotes ~10%)
    extra_rel_err: extra fractional error added to ALL points, e.g. 0.05 to allow a 5% model/systematic uncertainty
    min_points   : windows with fewer points than this are reported as n/a
    """
    # 1. put every data point of every experiment into flat arrays
    E, y, err, src = _gather_data(exp_file, lvd_norm_err, extra_rel_err)

    # 2. loop over windows and models
    names = list(models)
    rows = []
    for E_ref in ref_energies:
        lo, hi = E_ref / half_width, E_ref * half_width
        sel = (E >= lo) & (E < hi)
        row = dict(E_ref=E_ref, lo=lo, hi=hi, n=int(sel.sum()),
                   sources="+".join(dict.fromkeys(src[sel])) if sel.any() else "-",
                   data_range=(E[sel].min(), E[sel].max()) if sel.any() else None)
        for name, model in models.items():
            if sel.sum() >= min_points:
                pred = model(E[sel], theta)
                row[name] = (np.sum(((y[sel] - pred) / err[sel])**2) / sel.sum(),   # chi2 per point
                             np.exp(np.mean(np.log(pred / y[sel]))))               # mean model/data
            else:
                row[name] = (np.nan, np.nan)
        rows.append(row)

    # an "all points" row
    allrow = dict(E_ref=None, lo=E.min(), hi=E.max(), n=len(E), sources="all", data_range=(E.min(), E.max()))
    for name, model in models.items():
        pred = model(E, theta)
        allrow[name] = (np.sum(((y - pred) / err)**2) / len(E), np.exp(np.mean(np.log(pred / y))))

    # 3. print the tables
    w = 15
    short = [n.split(" (")[0].split(" &")[0][:w] for n in names]
    for title, k, fmt in (("chi^2 per data point  (smaller is better)", 0, "{:>{w}.3g}"),
                          ("mean model/data ratio  (1 = unbiased)", 1, "{:>{w}.3f}")):
        print(f"\n{title}")
        print(f"{'E_ref [GeV]':>12} {'n':>3}  {'data used':<22}" + "".join(f"{s:>{w}}" for s in short))
        for r in rows + [allrow]:
            label = "all" if r["E_ref"] is None else f"{r['E_ref']:.0e}"
            used = "no data" if r["data_range"] is None else f"{r['data_range'][0]:.2g}-{r['data_range'][1]:.2g} GeV"
            print(f"{label:>12} {r['n']:>3}  {used:<22}" + "".join(fmt.format(r[n][k], w=w) for n in names))
    print("\nnote: chi^2 treats every error as independent. Systematic errors are really correlated between points,"
          "\n      so use these numbers to RANK models, not as absolute goodness-of-fit probabilities.")

    # 4. optional csv + plot
    if save_csv:
        with open(save_csv, "w") as fh:
            fh.write("E_ref_GeV,window_lo,window_hi,n_points,sources," + ",".join(f"{n}_chi2_per_pt,{n}_mean_ratio" for n in names) + "\n")
            for r in rows + [allrow]:
                ref = "all" if r["E_ref"] is None else r["E_ref"]
                fh.write(f"{ref},{r['lo']:.4g},{r['hi']:.4g},{r['n']},{r['sources']}," +
                         ",".join(f"{r[n][0]:.6g},{r[n][1]:.6g}" for n in names) + "\n")
        print(f"saved {save_csv}")
    if make_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        for n in names:
            ax.plot([r["E_ref"] for r in rows], [r[n][0] for r in rows], "o-", label=n)
        ax.axhline(1, color="k", lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Reference energy (GeV)  [window = +/- half a decade]")
        ax.set_ylabel("chi^2 per data point")
        ax.set_title(f"Model vs data (Zenith = {theta})")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        fig.savefig(os.path.join(PLOTS_DIR, f"chi2_by_energy_zenith{theta}.png"), dpi=200)
        plt.show()
    return rows, allrow


def chi_squared_all_points(models, exp_file, theta=0, lvd_norm_err=0.10, extra_rel_err=0.0,
                           smooth=4, show_residuals=True, save_csv=None, summary_csv=None, make_plot=True):
    """
    Chi-squared contribution of EVERY data point, for every model, plotted against energy.

        pull_i = (data_i - model(E_i)) / err_i          (signed: + means the data are ABOVE the model)
        chi2_i = pull_i ** 2                             (the point's contribution; about 1 if the model is right)

    Top panel   : chi2_i for every point (marker shape = experiment, colour = model) with a running median per model
                  (smooth = half-width in points; smooth=0 turns the line off).
    Bottom panel: the signed pulls, so you can see whether a model is too high or too low.
    lvd_norm_err / extra_rel_err: extra errors, same meaning as in chi_squared_by_energy.
    save_csv   : per-point table (one row per data point)
    summary_csv: the printed summary table (one row per model)
    """
    E, y, err, src = _gather_data(exp_file, lvd_norm_err, extra_rel_err)

    # sort by energy ONCE: the file is ordered by experiment and the experiments overlap in energy,
    # so a line through the points in file order would zigzag back and forth
    order = np.argsort(E)
    E, y, err, src = E[order], y[order], err[order], src[order]

    names = list(models)
    pred = {n: np.asarray(models[n](E, theta), dtype=float) for n in names}
    pull = {n: (y - pred[n]) / err for n in names}
    chi = {n: pull[n] ** 2 for n in names}
    sources = list(dict.fromkeys(src))

    # ---- summary table: total chi2, chi2 per point, and chi2 per point for each experiment ----
    w = 15
    print(f"\nchi^2 over all {len(E)} data points (E = {E.min():.2g} - {E.max():.2g} GeV)")
    print(f"{'model':<28}{'total chi2':>12}{'chi2/point':>12}" + "".join(f"{s + ' /pt':>{w}}" for s in sources))
    for n in names:
        per_src = "".join(f"{np.mean(chi[n][src == s]):>{w}.3g}" for s in sources)
        print(f"{n:<28}{chi[n].sum():>12.4g}{chi[n].mean():>12.3g}" + per_src)
    print("note: errors are treated as independent; use the numbers to rank models (see the earlier caveats).")

    if save_csv:
        header = "source,E_GeV,flux,err," + ",".join(f"{n}_model,{n}_chi2" for n in names)
        cols = [src, E, y, err] + [a for n in names for a in (pred[n], chi[n])]
        with open(save_csv, "w") as fh:
            fh.write(header + "\n")
            for i in range(len(E)):
                fh.write(f"{src[i]},{E[i]:.6e},{y[i]:.6e},{err[i]:.6e}," +
                         ",".join(f"{a[i]:.6e}" for n in names for a in (pred[n], chi[n])) + "\n")
        print(f"saved {save_csv}")
    if summary_csv:
        with open(summary_csv, "w") as fh:
            fh.write("model,n_points,total_chi2,chi2_per_pt," + ",".join(f"{s}_chi2_per_pt" for s in sources) + "\n")
            for n in names:
                fh.write(f"{n},{len(E)},{chi[n].sum():.6g},{chi[n].mean():.6g}," +
                         ",".join(f"{np.mean(chi[n][src == s]):.6g}" for s in sources) + "\n")
        print(f"saved {summary_csv}")

    if make_plot:
        from matplotlib.lines import Line2D
        nrows = 2 if show_residuals else 1
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 7.5 if show_residuals else 5.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]} if show_residuals else None)
        ax = np.atleast_1d(axes)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        def running_median(v, half):
            return np.array([np.median(v[max(0, i - half): i + half + 1]) for i in range(len(v))])

        for k, n in enumerate(names):
            for s in sources:
                m = src == s
                mk = DATA_STYLE.get(s, dict(fmt="o"))["fmt"]
                ax[0].plot(E[m], chi[n][m], mk, color=colors[k], alpha=0.45, markersize=4, linestyle="none")
                if show_residuals:
                    ax[1].plot(E[m], pull[n][m], mk, color=colors[k], alpha=0.45, markersize=4, linestyle="none")
            if smooth > 0:
                ax[0].plot(E, running_median(chi[n], smooth), "-", color=colors[k], lw=2)

        ax[0].axhline(1, color="k", lw=0.8, ls="--")
        ax[0].set_yscale("log"); ax[0].set_xscale("log")
        ax[0].set_ylabel("chi^2 contribution per point\n(data - model)^2 / err^2")
        ax[0].set_title(f"Every data point vs every model (Zenith = {theta})")
        ax[0].grid(alpha=0.3, which="both")
        handles = [Line2D([], [], color=colors[k], lw=2, label=n) for k, n in enumerate(names)]
        handles += [Line2D([], [], color="gray", marker=DATA_STYLE.get(s, dict(fmt="o"))["fmt"], ls="", label=s) for s in sources]
        ax[0].legend(handles=handles, fontsize=8, ncol=2)
        if show_residuals:
            ax[1].axhline(0, color="k", lw=0.8)
            ax[1].set_yscale("symlog", linthresh=1)
            biggest = max(np.abs(pull[n]).max() for n in names)
            ticks = sorted({0} | {sign * 10**k for sign in (-1, 1) for k in range(int(np.ceil(np.log10(biggest))) + 1)})
            ax[1].set_yticks(ticks)
            ax[1].set_yticklabels([f"{t:g}" for t in ticks])
            ax[1].set_ylabel("pull = (data - model)/err\n(+ : data above model)")
            ax[1].grid(alpha=0.3, which="both")
        ax[-1].set_xlabel("Muon Energy (GeV)")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"chi2_all_points_zenith{theta}.png"), dpi=200)
        plt.show()
    return dict(E=E, flux=y, err=err, source=src, pred=pred, chi2=chi, pull=pull)


def rank_models(models, exp_file, theta=0, inclined_file=None, inclined_theta_deg=55.8, save_csv=None):
    """
    Rank the models under several different, reasonable definitions of "best" and report the average rank.
    If one model wins in most scenarios, it is the robust choice; if the winner changes, the answer depends on
    what you care about (see the printed per-scenario table).

    Scenarios (lower is better in every one):
      chi2/pt, published errors         every point counts equally (BESS has 65 of the 90 points, so low energy dominates)
      chi2/pt, +5% / +10% error floor   allows for model uncertainty; stops tiny error bars from dominating
      chi2/pt, direct data only         drops LVD (its points are model-dependent)
      chi2/pt, equal weight per decade  each decade of energy counts equally, so the TeV region matters as much as GeV
      typical % deviation               rms of ln(model/data); does not depend on the error bars at all
      chi2/pt at ~56 deg (optional)     an angular check, if inclined_file is given
    save_csv: the printed table (one row per scenario: value and rank of every model, then average rank and wins)
    """
    names = list(models)
    E, y, err, src = _gather_data(exp_file, lvd_norm_err=0.10, extra_rel_err=0.0)

    def chi2_pt(E, y, err, mask=None, floor=0.0):
        m = np.ones(len(E), bool) if mask is None else mask
        e = np.hypot(err[m], floor * y[m])
        return {n: np.mean(((y[m] - models[n](E[m], theta)) / e) ** 2) for n in names}

    scen = {}
    scen["chi2/pt, published errors"] = chi2_pt(E, y, err)
    scen["chi2/pt, +5% error floor"] = chi2_pt(E, y, err, floor=0.05)
    scen["chi2/pt, +10% error floor"] = chi2_pt(E, y, err, floor=0.10)
    scen["chi2/pt, no LVD"] = chi2_pt(E, y, err, mask=(src != "LVD"))

    decade = np.floor(np.log10(E)).astype(int)
    per_decade = {n: [] for n in names}
    for d in np.unique(decade):
        sel = decade == d
        for n, v in chi2_pt(E, y, err, mask=sel).items():
            per_decade[n].append(v)
    scen["chi2/pt, equal weight/decade"] = {n: float(np.mean(v)) for n, v in per_decade.items()}

    scen["typical % deviation"] = {n: 100 * (np.exp(np.sqrt(np.mean(np.log(models[n](E, theta) / y) ** 2))) - 1) for n in names}

    if inclined_file:
        d = np.genfromtxt(inclined_file, delimiter=",", names=True)
        Ei, yi, ei = d["E_GeV"], d["dPhidE_per_m2_s_sr_GeV"], d["err_per_m2_s_sr_GeV"]
        th = np.radians(inclined_theta_deg)
        scen[f"chi2/pt at {inclined_theta_deg:.0f} deg"] = {n: np.mean(((yi - models[n](Ei, th)) / ei) ** 2) for n in names}

    # ranks (1 = best) in every scenario
    ranks = {}
    for sname, vals in scen.items():
        order = sorted(names, key=lambda n: vals[n])
        ranks[sname] = {n: order.index(n) + 1 for n in names}
    mean_rank = {n: np.mean([ranks[s][n] for s in scen]) for n in names}

    short = [n.split(" (")[0].split(" &")[0][:12] for n in names]
    print(f"\n{'scenario (value; rank in brackets)':<34}" + "".join(f"{s:>16}" for s in short))
    for sname, vals in scen.items():
        print(f"{sname:<34}" + "".join(f"{vals[n]:>11.3g} ({ranks[sname][n]})" for n in names))
    print(f"{'AVERAGE RANK (lower = better)':<34}" + "".join(f"{mean_rank[n]:>16.2f}" for n in names))
    wins = {n: sum(ranks[s][n] == 1 for s in scen) for n in names}
    print(f"{'scenarios won':<34}" + "".join(f"{wins[n]:>16d}" for n in names))
    best = min(names, key=lambda n: mean_rank[n])
    print(f"\nbest on average: {best}")

    if save_csv:
        with open(save_csv, "w") as fh:
            fh.write("scenario," + ",".join(f"{n}_value,{n}_rank" for n in names) + "\n")
            for sname, vals in scen.items():
                fh.write(f"{sname}," + ",".join(f"{vals[n]:.6g},{ranks[sname][n]}" for n in names) + "\n")
            fh.write("average rank," + ",".join(f",{mean_rank[n]:.3g}" for n in names) + "\n")
            fh.write("scenarios won," + ",".join(f",{wins[n]}" for n in names) + "\n")
        print(f"saved {save_csv}")
    return scen, ranks, mean_rank


def read_zenith_data(file):
    """all_l3c_by_zenith_energy.csv -> {zenith_deg: (E, flux, err)}, one entry per L3+C zenith bin."""
    d = np.genfromtxt(file, delimiter=",", names=True)
    bins = {}
    for z in dict.fromkeys(d["zenith_deg"]):
        rows = d["zenith_deg"] == z
        bins[float(z)] = (d["E_GeV"][rows], d["dPhidE_per_m2_s_sr_GeV"][rows], d["err_per_m2_s_sr_GeV"][rows])
    return bins


def chi_squared_by_zenith(models, zenith_file, extra_rel_err=0.0, save_csv=None, make_plot=True):
    """
    How well does each model follow the ANGULAR dependence?  Uses the L3+C measurements at several zenith angles.
    Each model is evaluated at the centre zenith angle of each bin; prints chi^2 per point and the mean model/data
    ratio for every angle, then plots chi^2/pt against zenith angle.
    extra_rel_err: extra fractional error added to every point (e.g. 0.05 for a 5% model floor).
    """
    bins = read_zenith_data(zenith_file)
    names = list(models)
    zen = sorted(bins)
    chi, ratio = {n: [] for n in names}, {n: [] for n in names}
    for z in zen:
        E, y, err = bins[z]
        e = np.hypot(err, extra_rel_err * y)
        for n in names:
            pred = models[n](E, np.radians(z))
            chi[n].append(np.mean(((y - pred) / e) ** 2))
            ratio[n].append(np.exp(np.mean(np.log(pred / y))))

    w = 15
    short = [n.split(" (")[0].split(" &")[0][:12] for n in names]
    for title, store, fmt in (("chi^2 per point (L3+C, 20 GeV - 3 TeV)", chi, "{:>{w}.3g}"),
                              ("mean model/data ratio", ratio, "{:>{w}.3f}")):
        print(f"\n{title}")
        print(f"{'zenith [deg]':>12} {'n':>3}" + "".join(f"{s:>{w}}" for s in short))
        for i, z in enumerate(zen):
            print(f"{z:>12.1f} {len(bins[z][0]):>3}" + "".join(fmt.format(store[n][i], w=w) for n in names))
        allchi = {n: np.mean(store[n]) for n in names}
        print(f"{'average':>12} {'':>3}" + "".join(fmt.format(allchi[n], w=w) for n in names))

    if save_csv:
        with open(save_csv, "w") as fh:
            fh.write("zenith_deg,n_points," + ",".join(f"{n}_chi2_per_pt,{n}_mean_ratio" for n in names) + "\n")
            for i, z in enumerate(zen):
                fh.write(f"{z:.2f},{len(bins[z][0])}," + ",".join(f"{chi[n][i]:.6g},{ratio[n][i]:.6g}" for n in names) + "\n")
            fh.write("average,," + ",".join(f"{np.mean(chi[n]):.6g},{np.mean(ratio[n]):.6g}" for n in names) + "\n")
        print(f"saved {save_csv}")

    if make_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        for n in names:
            ax.plot(zen, chi[n], "o-", label=n)
        ax.axhline(1, color="k", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("Zenith angle (deg)  [bin centre]")
        ax.set_ylabel("chi^2 per data point")
        ax.set_title("Angular dependence: models vs L3+C")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        fig.savefig(os.path.join(PLOTS_DIR, "chi2_by_zenith.png"), dpi=200)
        plt.show()
    return zen, chi, ratio

def main():

    # 0.5 GeV to 100 TeV, so the curves cover the whole range of the data (0.6 GeV - 43 TeV)
    ENERGY_RANGE = np.logspace(np.log10(0.5), 5, 400)
    THETA = 0
    EXP_DATA = os.path.join(DATA_DIR, "all_data_zenith0_energy.csv")
    WEIGHT_POWER = 2.7        # 0 = raw spectrum; try 2.7 to flatten it and compare models more easily

    MODELS = {
        "Gaisser": gaisser_formula,
        "Guan (modified Gaisser)": modified_gaisser_formula,
        "Gaisser/Tang": tang_modified_gaisser,
        "Bugaev/Reyna": reyna_formula,
        "Smith & Duller/Chatzidakis": smith_duller_chatzidakis,
    }
    differential_spectra(MODELS, ENERGY_RANGE, THETA, exp_file=EXP_DATA, weight_power=WEIGHT_POWER)
    chi_squared_all_points(MODELS, EXP_DATA, THETA, save_csv=os.path.join(DATA_DIR, "chi2_all_points.csv"),
                           summary_csv=os.path.join(DATA_DIR, "chi2_all_points_summary.csv"))
    chi_squared_by_energy(MODELS, EXP_DATA, THETA, save_csv=os.path.join(DATA_DIR, "chi2_by_energy.csv"))
    chi_squared_by_zenith(MODELS, os.path.join(DATA_DIR, "all_l3c_by_zenith_energy.csv"), save_csv=os.path.join(DATA_DIR, "chi2_by_zenith.csv"))
    rank_models(MODELS, EXP_DATA, THETA, save_csv=os.path.join(DATA_DIR, "model_ranking.csv"))   # add inclined_file=os.path.join(DATA_DIR, "l3c_zenith56_energy.csv") for the 56 deg check


if __name__ == "__main__":
    main()
