"""
Generate the two Chapter 6 figures from saved correct-model estimates:
  figures/fig_ch6_nu_identification.png   — estimated vs true ν under SV-t
  figures/fig_ch6_rho_identification.png  — estimated vs true ρ under ASV

Run:  uv run python scripts/make_chapter6_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 12})
Path("figures").mkdir(exist_ok=True)


def nu_figure():
    nn = np.load("results/misspec/svt_tcn_correct.npz")
    mc = np.load("results/misspec/svt_stochvol_correct.npz")
    true = nn["true"][:, 3]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    for ax, est, title in [(axes[0], nn["est"][:, 3], "Neural network (correct SV-t)"),
                           (axes[1], mc["est"][:, 3], "MCMC / stochvol (correct SV-t)")]:
        ax.scatter(true, est, alpha=0.5, s=20, color="#4878D0", edgecolors="none")
        ax.plot([2, 42], [2, 42], "k--", lw=1, label="perfect estimation")
        ax.axvline(10, color="#EE854A", lw=1, ls=":", label="low/high ν split (ν=10)")
        ax.set_xlim(2, 42); ax.set_ylim(0, 60)
        ax.set_xlabel("True ν"); ax.set_title(title); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Estimated ν"); axes[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Degrees-of-freedom estimation under SV-t: accurate at low ν, "
                 "weakly identified at high ν", y=1.01)
    fig.tight_layout()
    fig.savefig("figures/fig_ch6_nu_identification.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def rho_figure():
    nn = np.load("results/misspec/asv_tcn_correct.npz")
    mc = np.load("results/misspec/asv_stochvol_correct.npz")
    true = nn["true"][:, 3]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    for ax, est, title, rmse in [(axes[0], nn["est"][:, 3], "Neural network (correct ASV)", 0.407),
                                 (axes[1], mc["est"][:, 3], "MCMC / stochvol (correct ASV)", 0.150)]:
        ax.scatter(true, est, alpha=0.5, s=20, color="#4878D0", edgecolors="none")
        ax.plot([-1, 0.6], [-1, 0.6], "k--", lw=1, label="perfect estimation")
        ax.set_xlim(-1.0, 0.6); ax.set_ylim(-1.0, 0.6)
        ax.set_xlabel("True ρ"); ax.set_title(f"{title}\n(RMSE {rmse})"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Estimated ρ"); axes[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Leverage parameter ρ under ASV: recovered by MCMC, poorly by the network", y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig_ch6_rho_identification.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    nu_figure()
    rho_figure()
    print("Wrote figures/fig_ch6_nu_identification.png and fig_ch6_rho_identification.png")
