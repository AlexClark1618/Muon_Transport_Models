# daemonflux: user guide

A practical guide to using **daemonflux** (DAta-drivEn and MuOn-calibrated Neutrino flux) from Python.
daemonflux is a tabulated and splined version of an atmospheric lepton flux model that has been
calibrated on muon spectrometer data. It returns atmospheric **muon** and **neutrino** fluxes
together with their uncertainties.

- Package: https://pypi.org/project/daemonflux/
- Source: https://github.com/mceq-project/daemonflux
- Paper: Yañez & Fedynitch, arXiv:2303.00022

> **Scope note.** This guide documents what is stated in the official PyPI documentation.
> Items the official page does not spell out (other `location` values, the model's parameter
> names, which zenith angles are allowed) are marked **"discover it yourself"**, and
> [section 8](#8-discovering-features-this-guide-does-not-list) shows how to look them up
> in your own install.

---

## Table of contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Quickstart](#3-quickstart)
4. [API reference](#4-api-reference)
5. [Quantities you can request](#5-quantities-you-can-request)
6. [Units (read this!)](#6-units-read-this)
7. [Worked examples](#7-worked-examples)
8. [Discovering features this guide does not list](#8-discovering-features-this-guide-does-not-list)
9. [Troubleshooting](#9-troubleshooting)
10. [Citation](#10-citation)

---

## 1. Requirements

| Requirement | Notes |
|---|---|
| Python | 3.10 or newer for the latest release (0.9.0). Older releases allowed older Pythons. |
| numpy, scipy | Installed automatically with the package. |
| matplotlib | Needed for plotting and the examples. Install it yourself. |

Check your Python version:

```bash
python --version
```

## 2. Installation

### Step 1: Create and activate a virtual environment (recommended)

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 2: Install from PyPI

```bash
pip install daemonflux matplotlib
```

### Alternative: install from source (editable)

Use this if you want the package to update whenever you `git pull`:

```bash
git clone https://github.com/mceq-project/daemonflux
cd daemonflux
python3 -m pip install -e .
```

### Step 3: Verify

```bash
python -c "import daemonflux; print('daemonflux imported OK')"
```

## 3. Quickstart

```python
import numpy as np
import matplotlib.pyplot as plt
from daemonflux import Flux

fx = Flux(location='generic')          # 1. create the flux object
egrid = np.logspace(0, 5)              # 2. energies in GeV (1 GeV to 100 TeV)

fl  = fx.flux(egrid, '15', 'numuflux')   # 3. flux at 15 degrees zenith
err = fx.error(egrid, '15', 'numuflux')  # 4. 1-sigma uncertainty

plt.loglog(egrid, fl, color='k')
plt.fill_between(egrid, fl - err, fl + err, color='r', alpha=0.3, label=r'1$\sigma$ error')
plt.xlabel('Energy [GeV]')
plt.ylabel(r'$E^3 \times$ flux')
plt.legend()
plt.show()
```

## 4. API reference

### `Flux(location=...)`

Creates the flux calculator. Everything else is a method on this object.

| Argument | Type | Description |
|---|---|---|
| `location` | `str` | The site the flux is computed for. `'generic'` is the documented default used in the official quickstart. **Discover it yourself:** other site names may be supported, see [section 8](#8-discovering-features-this-guide-does-not-list). |

```python
fx = Flux(location='generic')
```

### `Flux.flux(grid, angle, quantity)`

Returns the calibrated central-value flux.

| Argument | Type | Description |
|---|---|---|
| `grid` | `numpy.ndarray` | Energies in **GeV** for neutrinos, or total momenta in **GeV/c** for muons. A log-spaced grid from `np.logspace(...)` is typical. |
| `angle` | `str` | Zenith angle in degrees, passed as a **string** (for example `'15'`). Which values are allowed: **discover it yourself**. |
| `quantity` | `str` | What to compute, e.g. `'muflux'` or `'numuflux'`. Full list in [section 5](#5-quantities-you-can-request). |

**Returns:** a numpy array, one value per grid point, already multiplied by E³ (neutrinos)
or p³ (muons). See [section 6](#6-units-read-this).

### `Flux.error(grid, angle, quantity)`

Returns the 1σ uncertainty on the same quantity, with the same arguments and units as
`Flux.flux`. The model parameters are correlated; per the official docs, the errors **already
account for the covariance matrix** when you use `error`.

### `Flux.chi2({...})`

Returns the chi-squared penalty for a given combination of model parameters.
Pass a **dictionary of the parameters you modified**.

Use this when you include daemonflux in your own fit with free-floating parameters: adding the
chi2 as a penalty term keeps the parameter correlations (determined from the calibration data)
in your fit.

```python
# schematic only: the parameter names are NOT listed here,
# find them in your install (section 8)
penalty = fx.chi2({'some_parameter': 0.5})
```

## 5. Quantities you can request

Pass any of these as the `quantity` string.

| Group | Quantity names |
|---|---|
| Muons | `muflux`, `muratio`, `mu+`, `mu-` |
| Muon neutrinos | `numuflux`, `numuratio`, `numu`, `antinumu`, `flavorratio` |
| Electron neutrinos | `nueflux`, `nueratio`, `nue`, `antinue`, `flavorratio` |
| Total (conventional + prompt) | `total_` prefix, e.g. `total_muflux` |

### Definitions

| Quantity | Meaning |
|---|---|
| `XXXflux` | Particle plus antiparticle, e.g. `numuflux = numu + antinumu`, `muflux = mu+ + mu-` |
| `XXXratio` | Particle divided by antiparticle, e.g. `numuratio = numu / antinumu` |
| `flavorratio` | `(numu + antinumu) / (nue + antinue)` |
| `total_XXX` | Conventional **plus** prompt atmospheric flux |

### About the `total_` quantities

Only the **conventional** flux is calibrated by the daemonflux method, because surface muons
are not sensitive to prompt fluxes. In a `total_` quantity, the prompt part is added
**uncalibrated**, with no correction parameters and no errors. The conventional part stays calibrated.

## 6. Units (read this!)

The outputs are **pre-multiplied by a power of the energy or momentum**, which flattens the
steep spectra. Natural units (ħ = c = 1) are used throughout.

| Particle type | x-axis | Output units |
|---|---|---|
| Neutrinos | Energy E in GeV | (E/GeV)³ / (GeV s sr cm²) |
| Muons | **Total momentum** p in GeV/c (not energy) | (p/(GeV/c))³ / ((GeV/c) s sr cm²) |

To recover the raw differential flux:

```python
raw_nu = fx.flux(egrid, '15', 'numuflux') / egrid**3   # neutrinos
raw_mu = fx.flux(pgrid, '15', 'muflux')   / pgrid**3   # muons
```

## 7. Worked examples

### 7.1 Muon spectrum at the surface (μ⁺ + μ⁻, and each charge)

```python
import numpy as np
import matplotlib.pyplot as plt
from daemonflux import Flux

fx = Flux(location='generic')
pgrid = np.logspace(0, 4, 60)      # momentum in GeV/c
angle = '15'

mu       = fx.flux(pgrid, angle, 'muflux')
mu_err   = fx.error(pgrid, angle, 'muflux')
mu_plus  = fx.flux(pgrid, angle, 'mu+')
mu_minus = fx.flux(pgrid, angle, 'mu-')

fig, ax = plt.subplots()
ax.loglog(pgrid, mu, 'k', label=r'$\mu^+ + \mu^-$')
ax.fill_between(pgrid, mu - mu_err, mu + mu_err, color='r', alpha=0.3, label=r'1$\sigma$')
ax.loglog(pgrid, mu_plus,  '--', label=r'$\mu^+$')
ax.loglog(pgrid, mu_minus, ':',  label=r'$\mu^-$')
ax.set_xlabel('Muon momentum p [GeV/c]')
ax.set_ylabel(r'$p^3\,d\Phi/dp$')
ax.legend()
plt.show()
```

### 7.2 Muon charge ratio

```python
ratio = fx.flux(pgrid, angle, 'muratio')
plt.semilogx(pgrid, ratio)
plt.xlabel('Muon momentum p [GeV/c]')
plt.ylabel(r'$\mu^+/\mu^-$')
plt.show()
```

### 7.3 Include prompt flux

```python
total = fx.flux(egrid, angle, 'total_numuflux')
conv  = fx.flux(egrid, angle, 'numuflux')
prompt_like = total - conv     # the uncalibrated prompt contribution
```

### 7.4 Compare several zenith angles

```python
for ang in ['0', '30', '60']:      # check these values are accepted in your install
    plt.loglog(egrid, fx.flux(egrid, ang, 'numuflux'), label=f'{ang}°')
plt.legend()
plt.show()
```

### 7.5 Neutrino flavor ratio

```python
ratio = fx.flux(egrid, angle, 'flavorratio')
```

## 8. Discovering features this guide does not list

Python can tell you what is inside any package. These are the quickest ways to find parameter
names, other locations and any extra methods.

### Step 1: See everything on the object

```python
from daemonflux import Flux
fx = Flux(location='generic')
print([name for name in dir(fx) if not name.startswith('_')])
```

### Step 2: Read the built-in help

```python
help(Flux)             # constructor and class documentation
help(fx.flux)          # arguments of flux()
help(fx.error)         # arguments of error()
help(fx.chi2)          # arguments of chi2()
```

### Step 3: Inspect stored attributes and values

```python
for key, value in vars(fx).items():
    print(key, type(value))
```

This usually reveals the model's adjustable parameters, available angles and so on.

### Step 4: Read the example notebook and source

The project's GitHub repository (`mceq-project/daemonflux`) contains a detailed example
notebook and the source code. When something here is marked "discover it yourself", that is
the authoritative place to check.

## 9. Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'daemonflux'` | Wrong interpreter. In VS Code press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and pick the one in `.venv`. |
| `pip install` fails with a Python-version error | The latest release needs Python 3.10+. Upgrade Python, or install an older daemonflux. |
| Muon plot looks wrong on the x-axis | Muon grids are **momentum in GeV/c**, not energy. |
| Flux values look "too flat" or too large | Outputs are multiplied by E³ or p³. Divide to get the raw flux ([section 6](#6-units-read-this)). |
| Error when passing a number as the angle | The angle must be a **string**, e.g. `'15'` not `15`. |
| Angle or location raises an error | That value may not be supported. Use [section 8](#8-discovering-features-this-guide-does-not-list) to find valid ones. |
| No error band on prompt flux | Expected: the prompt component is not calibrated and has no errors. |
| PowerShell blocks venv activation (Windows) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry. |

## 10. Citation

If you use daemonflux in published work, cite:

```bibtex
@article{Yanez:2023lsy,
    author = "Ya\~nez, Juan Pablo and Fedynitch, Anatoli",
    title = "{daemonflux: DAta-drivEn MuOn-calibrated Neutrino Flux}",
    eprint = "2303.00022",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "2",
    year = "2023"
}
```

daemonflux is released under the BSD 3-Clause License.
