# Reproducibility Package
**Operational Quantum Foundations: Relational Quantum Mechanics and the Principle of Finite Informational Alignment for Subjective Physics**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17139825.svg)](https://doi.org/10.5281/zenodo.17139825)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
---

## Overview
This package provides the Python scripts used in the publication:

*Vladimir Khomyakov (2025). Operational Quantum Foundations: Relational Quantum Mechanics and the Principle of Finite Informational Alignment for Subjective Physics.*

- **Version-specific DOI:** [10.5281/zenodo.17139825](https://doi.org/10.5281/zenodo.17139825)  
- **Concept DOI (latest version):** [10.5281/zenodo.17139824](https://doi.org/10.5281/zenodo.17139824) 

---

## Description (for Zenodo)

This work develops the operational foundations of quantum theory within **Relational Quantum Mechanics**, introducing the principle of **Finite Informational Capacity** and **R-events** as fundamental units of interaction. The framework explains the **emergence of classicality** and addresses the **Wigner’s Friend paradox** through Monte Carlo simulations and bootstrap inference, providing a reproducible model of the **quantum measurement problem** grounded in subjective physics.

---

## Repository
- **Source repository:** [https://github.com/Khomyakov-Vladimir/operational-quantum-foundations](https://github.com/Khomyakov-Vladimir/operational-quantum-foundations)

---

## Package Structure

```
operational-quantum-foundations/
│
├── README.md
│
├── scripts/
│   │ 
│   ├── monte_carlo_simulation_particle_filter.py  # Monte Carlo simulation of the two-observer quantum 
│   │                                              # state tracking task using a Particle Filter
│   │                                              # (Bayesian particle approximation) for state estimation.
│   │ 
│   └── fit_alpha_bootstrap_replicates.py          # Estimation of the constant α in the model:
│                                                  # F_max(m) ≈ 1 - α / 2^m
│                                                  # using bootstrap resampling at the trial level from
│                                                  # plateau_estimates_raw.csv.
│
└── figures/ 
    │
    ├── alpha_fit_plot.pdf
    ├── alpha_bootstrap_summary.json
    ├── alpha_bootstrap_replicates_samples.csv
    ├── alpha_bootstrap_replicates_hist.pdf
    ├── plateau_estimates_raw.csv
    ├── plateau_estimates.csv
    ├── fidelity_results.csv
    └── fidelity_convergence.pdf
```

---

## Dependencies
The code requires the following Python packages (used in the scripts):

- Python 3.9+
- NumPy
- SciPy
- Pandas
- Matplotlib
- scikit-learn

The exact versions used for verification and reproducibility are pinned in `requirements.txt` (see below).

Install the pinned versions with:

```bash
pip install -r requirements.txt
```

If you prefer conda, see the `environment.yml` included in this archive and use `conda env create -f environment.yml`.

---

## Installation

To install the Python dependencies pinned for this release, run:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file included in this archive pins the package versions used to reproduce the results reported in this repository. If you prefer a conda-managed environment, use the provided `environment.yml` file.

---

### requirements.txt (pinned versions used for verification and reproducibility)

```
numpy==2.0.2
scipy==1.13.1
scikit-learn==1.6.1
matplotlib==3.9.4
pandas==2.3.1
```

---

## Python Environment (conda)

File: `environment.yml`

```yaml
name: RQM-SF
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy=2.0.2
  - scipy=1.13.1
  - scikit-learn=1.6.1
  - matplotlib=3.9.4
  - pandas=2.3.1
```

You can activate this environment with:

```bash
conda env create -f environment.yml
conda activate RQM-SF
```

---

## Usage (from repository root)

**Important:** the scripts are located in the `scripts/` directory. All example commands below assume you are running them from the repository root (the `operational-quantum-foundations/` folder). Output files are written into `./figures` by default.

### 1. Monte Carlo simulation (particle filter)

Run the simulation script (particle-filter mode):

```bash
python scripts/monte_carlo_simulation_particle_filter.py --m_list 1 2 3 4 5 6 7 --n_max 3000 --trials 1000 --update_mode particle --particles 2048 --measure_protocol same
```

This script produces the following files in `./figures`:

- `./figures/fidelity_results.csv`: Average fidelity vs. measurement number for each m (one row per measurement number; columns for m values).
- `./figures/plateau_estimates.csv`: Mean plateau fidelity for each m (averaged over trials).
- `./figures/plateau_estimates_raw.csv`: Per-trial plateau fidelity values (trial-level) used for bootstrap analysis.
- `./figures/fidelity_convergence.pdf`: Plot of fidelity convergence dynamics (average fidelity vs measurement number for different m).

### 2. Estimate α via bootstrap (trial-level resampling)

Once `plateau_estimates_raw.csv` is available in `./figures`, run:

```bash
python scripts/fit_alpha_bootstrap_replicates.py --boots 5000
```

This script produces the following files in `./figures`:

- `./figures/alpha_bootstrap_replicates_hist.pdf`: Histogram of bootstrap replicates (trial-level resampling) with visual summary statistics.
- `./figures/alpha_bootstrap_replicates_samples.csv`: CSV with raw bootstrap replicates (column `alpha`). The script appends a `# Summary:` block with key statistics (alpha_hat, CI, R², n_boot, seed).
- `./figures/alpha_bootstrap_summary.json`: JSON file containing structured summary (alpha_hat, ci_low, ci_high, r2_linear, r2_log, n_boot, seed).
- `./figures/alpha_fit_plot.pdf`: Fit plot of F_max(m) with the model curve and statistical summary (α̂, 95% CI, R²).

---

## Citation

Khomyakov, V. (2025). *Operational Quantum Foundations: Relational Quantum Mechanics and the Principle of Finite Informational Alignment for Subjective Physics (1.0)*. Zenodo. [https://doi.org/10.5281/zenodo.17139825]

📄 BibTeX:

```
@misc{khomyakov_2025_17139825,
  author       = {Khomyakov, Vladimir},
  title        = {Operational Quantum Foundations: Relational
                  Quantum Mechanics and the Principle of Finite
                  Informational Alignment for Subjective Physics
                  },
  month        = sep,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.17139825},
  url          = {https://doi.org/10.5281/zenodo.17139825},
}
```
