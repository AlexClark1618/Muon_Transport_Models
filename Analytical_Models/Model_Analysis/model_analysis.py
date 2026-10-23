import os, sys
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt

# model files live in ../Analytical_Models
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Analytical_Models"))

# plots are saved in ./Plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Plots")

from gaisser_model import gaisser_formula
from guan_model import modified_gaisser_formula
from tang_model import tang_modified_gaisser
from bugaev_reyna_model import reyna_formula          # bugaev_formula is vertical-only
from smith_duller_model import smith_duller_chatzidakis

def differential_spectra(models, energy_range, theta, weight_power=0):
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
    plt.savefig(os.path.join(PLOTS_DIR, f"differential_spectra_zenith{theta}.png"), dpi=200)
    plt.show()
    plt.clf()

def integrated_vertical_flux(models, energy_range, theta):

    for label, model in models.items():
        int_list = [integrate.quad(model, E, np.inf, args=(theta,), epsabs=0)[0] for E in energy_range]

        plt.plot(energy_range, int_list, label = label)

    plt.xlabel("Minimum Muon Energy (GeV)")
    plt.ylabel("Vertical Muon Flux (m^-2 s^-1 sr^-1)")
    plt.title(f"Vertical Muon Flux at Surface (Zenith = {theta})")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, f"integrated_vertical_flux_zenith{theta}.png"), dpi=200)
    plt.show()
    plt.clf()

def angular_spectra(models, energy_range):

    zenith_points = np.linspace(0, np.pi/2, 90)
    energy_lower_bound = energy_range[0]

    for label, model in models.items():
        if label != "Gaisser":
            int_list = [integrate.quad(model, energy_lower_bound, np.inf, args=(theta,), epsabs=0)[0] for theta in zenith_points]

            plt.plot(zenith_points*180/np.pi, int_list, label = label)

    # cos^2 reference, normalised to the vertical flux of the last model
    cosine_list = [int_list[0]*np.cos(theta)**2 for theta in zenith_points]
    plt.scatter(zenith_points*180/np.pi, cosine_list, c = 'red', label = 'cos^2(x)')

    plt.xlabel("Zenith")
    plt.ylabel("Vertical Muon Flux (m^-2 s^-1 sr^-1)")
    plt.title(f"Muon Angular Distribution at Surface")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "angular_spectra.png"), dpi=200)
    plt.show()
    plt.clf()
    
def main():

    # 0.5 GeV to 100 TeV, so the curves cover the whole range of the data (0.6 GeV - 43 TeV)
    ENERGY_RANGE = np.logspace(np.log10(0.5), 5, 400)
    THETA = 0
    WEIGHT_POWER = 2.7        # 0 = raw spectrum; try 2.7 to flatten it and compare models more easily

    MODELS = {
        "Gaisser": gaisser_formula,
        "Guan (modified Gaisser)": modified_gaisser_formula,
        "Gaisser/Tang": tang_modified_gaisser,
        "Bugaev/Reyna": reyna_formula,
        "Smith & Duller/Chatzidakis": smith_duller_chatzidakis,
    }
    differential_spectra(MODELS, ENERGY_RANGE, THETA, weight_power=WEIGHT_POWER)
    integrated_vertical_flux(MODELS, ENERGY_RANGE, THETA)
    angular_spectra(MODELS, ENERGY_RANGE)

if __name__ == "__main__":
    main()
