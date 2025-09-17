"""
fit_alpha_bootstrap_replicates.py

Estimation of the constant α in the model:
    F_max(m) ≈ 1 - α / 2^m
using bootstrap resampling at the trial level from plateau_estimates_raw.csv.

Outputs:
--------
- Console: point estimate α, 95% confidence interval, R² (linear and logarithmic scales)
- PDF: histogram of bootstrap replicates with summary statistics
- CSV: raw bootstrap replicates with summary statistics
- JSON: structured summary (α̂, CI, R², number of bootstrap replicates, random seed)
- PDF: fit plot of F_max(m) with model curve and statistical summary

Usage example:
# From the repository root
python scripts/fit_alpha_bootstrap_replicates.py --boots 5000
python fit_alpha_bootstrap_replicates.py --boots 5000
 
# From the scripts directory
python fit_alpha_bootstrap_replicates.py --boots 5000

Author: Vladimir Khomyakov
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/operational-quantum-foundations
Citation: DOI:10.5281/zenodo.17139825
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def get_repo_paths():
    """Determine repository paths for consistent data and figure storage"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Always find repo root by looking for specific markers
    # (like .git directory or setup.py) or go up one level if in scripts
    repo_root = script_dir
    if os.path.basename(script_dir) == 'scripts':
        repo_root = os.path.dirname(script_dir)
    else:
        # Try to find repo root by looking for common markers
        current = script_dir
        while current != os.path.dirname(current):  # while not at root
            if any(os.path.exists(os.path.join(current, marker)) for marker in ['.git', 'setup.py', 'README.md']):
                repo_root = current
                break
            current = os.path.dirname(current)
    
    figures_dir = os.path.join(repo_root, "figures")
    data_dir = os.path.join(repo_root, "data")
    return {
        'repo_root': repo_root,
        'figures_dir': figures_dir,
        'data_dir': data_dir,
        'script_dir': script_dir
    }


def fit_alpha(ms, F):
    """Estimate α from the linear model y = α x, with y = 1 - F, x = 2^{-m}."""
    x = (2.0 ** (-ms)).reshape(-1, 1)
    y = (1.0 - F).reshape(-1, 1)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(x, y)
    return float(lr.coef_[0][0])


def bootstrap_alpha(df, n_boot=5000, seed=42):
    """
    Bootstrap procedure for α:
    - Within each m, resample trial-level replicates with replacement
    - Compute mean plateau value per resample
    - Re-estimate α from the resampled dataset
    """
    rng = np.random.default_rng(seed)
    alphas = []

    grouped = {m: g["plateau_mean"].values for m, g in df.groupby("m")}
    ms_unique = np.array(sorted(grouped.keys()))

    for _ in range(n_boot):
        sampled_means = []
        for m in ms_unique:
            values = grouped[m]
            idx = rng.integers(0, len(values), len(values))
            sampled = values[idx].mean()
            sampled_means.append(sampled)
        alpha = fit_alpha(ms_unique, np.array(sampled_means))
        alphas.append(alpha)

    return np.array(alphas)


def compute_r2(ms, F, alpha_hat):
    """
    Compute coefficients of determination (R²):
    - Linear scale: R² between observed F_max(m) and fitted curve
    - Logarithmic scale: R² for regression log(1 - F_max(m)) ~ m
    """
    # Linear scale
    F_pred = 1 - alpha_hat / (2 ** ms)
    r2_linear = r2_score(F, F_pred)

    # Logarithmic scale
    mask = (1 - F) > 0
    if np.sum(mask) > 1:
        y_log = np.log(1 - F[mask])
        x_log = ms[mask]
        slope, intercept = np.polyfit(x_log, y_log, 1)
        y_log_pred = slope * x_log + intercept
        r2_log = r2_score(y_log, y_log_pred)
    else:
        r2_log = np.nan

    return r2_linear, r2_log


def plot_fit(df, alpha_hat, ci_low, ci_high, r2_linear, r2_log, out_path):
    """Generate plot of F_max(m) with fitted curve, α̂, confidence interval, and R² values."""
    mean_df = df.groupby("m")["plateau_mean"].mean()
    sem_df = df.groupby("m")["plateau_mean"].sem()
    ms = mean_df.index.values
    means = mean_df.values
    errors = sem_df.values

    m_grid = np.linspace(ms.min(), ms.max(), 200)
    model = 1 - alpha_hat / (2 ** m_grid)

    plt.figure(figsize=(7, 5))
    plt.errorbar(ms, means, yerr=errors, fmt="o", capsize=4, label="Empirical mean ± SE")
    plt.plot(
        m_grid,
        model,
        "r--",
        label=(f"Model: 1 - α / 2^m\n"
               f"α̂ = {alpha_hat:.4f}\n"
               f"95% CI = [{ci_low:.3f}, {ci_high:.3f}]\n"
               f"R² (linear) = {r2_linear:.3f}\n"
               f"R² (log) = {r2_log:.3f}")
    )
    plt.xlabel("m (system size)")
    plt.ylabel("F_max(m)")
    plt.title("Fit of F_max(m) with bootstrap-estimated α")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved fit plot: {out_path}")


def plot_histogram(alphas, alpha_hat, ci_low, ci_high, r2_linear, r2_log, out_path):
    """Generate histogram of bootstrap replicates with α̂, confidence interval, and R² values."""
    plt.figure(figsize=(8, 5))
    plt.hist(alphas, bins=50, density=True, alpha=0.7, color="steelblue")
    plt.axvline(alpha_hat, color="red", linestyle="--")
    plt.axvline(ci_low, color="black", linestyle=":")
    plt.axvline(ci_high, color="black", linestyle=":")

    plt.xlabel("α (bootstrap replicates)")
    plt.ylabel("Probability density")
    plt.title("Bootstrap distribution of α (trial-level resampling)")

    plt.legend(
        [f"α̂ = {alpha_hat:.4f}\n"
         f"95% CI = [{ci_low:.3f}, {ci_high:.3f}]\n"
         f"R² (linear) = {r2_linear:.3f}\n"
         f"R² (log) = {r2_log:.3f}"],
        loc="upper right"
    )

    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved histogram: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap estimation of α from plateau_estimates_raw.csv")
    parser.add_argument("--boots", type=int, default=5000, help="Number of bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Get repository paths
    paths = get_repo_paths()
    
    # Set default paths
    args.csv = os.path.join(paths['figures_dir'], "plateau_estimates_raw.csv")
    args.outdir = paths['figures_dir']

    df = pd.read_csv(args.csv)
    required_columns = ["m", "trial", "plateau_mean"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain the following columns: {required_columns}")

    # Rest of the code remains the same...
    # Point estimate
    mean_df = df.groupby("m", as_index=False)["plateau_mean"].mean()
    ms = mean_df["m"].values
    F = mean_df["plateau_mean"].values
    alpha_hat = fit_alpha(ms, F)
    print(f"Point estimate α = {alpha_hat:.6f}")

    # Goodness of fit
    r2_linear, r2_log = compute_r2(ms, F, alpha_hat)
    print(f"R² (linear scale) = {r2_linear:.4f}")
    print(f"R² (log scale)    = {r2_log:.4f}")

    # Bootstrap replicates
    alphas = bootstrap_alpha(df, n_boot=args.boots, seed=args.seed)
    ci_low, ci_high = np.percentile(alphas, [2.5, 97.5])
    print(f"95% CI for α: [{ci_low:.6f}, {ci_high:.6f}]")

    # Summary
    summary = {
        "alpha_hat": float(alpha_hat),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "r2_linear": float(r2_linear),
        "r2_log": float(r2_log),
        "n_boot": int(args.boots),
        "seed": int(args.seed)
    }

    # Histogram
    hist_path = f"{args.outdir}/alpha_bootstrap_replicates_hist.pdf"
    plot_histogram(alphas, alpha_hat, ci_low, ci_high, r2_linear, r2_log, hist_path)

    # CSV with bootstrap replicates + summary
    out_csv = f"{args.outdir}/alpha_bootstrap_replicates_samples.csv"
    df_out = pd.DataFrame({"alpha": alphas})
    df_out.to_csv(out_csv, index=False)
    with open(out_csv, "a") as f:
        f.write("\n# Summary:\n")
        for k, v in summary.items():
            f.write(f"# {k} = {v}\n")
    print(f"Saved bootstrap samples with summary: {out_csv}")

    # JSON summary
    out_json = f"{args.outdir}/alpha_bootstrap_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {out_json}")

    # Fit plot
    fit_path = f"{args.outdir}/alpha_fit_plot.pdf"
    plot_fit(df, alpha_hat, ci_low, ci_high, r2_linear, r2_log, fit_path)


if __name__ == "__main__":
    main()