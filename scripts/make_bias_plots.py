"""
Bias plots (estimation error vs true value) for TCN and MCMC at T=500/1000/2000.

Supervisor request: extend the T=1000 TCN bias plot (Chapter 5 Figure 6) to
T=500 and T=2000, and add MCMC versions for comparison (appendix). Tests whether
the shrinkage toward the centre of the simulation range is strongest at T=500 and
weakest at T=2000 (i.e. the range acts like a prior when data is scarce).

Output: figures/fig_bias_{tcn,mcmc}_T{500,1000,2000}.png

Run:  uv run python scripts/make_bias_plots.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 12})
Path("figures").mkdir(exist_ok=True)

PARAMS = [(r"$\mu$ (long-run mean)", "True $\\mu$"),
          (r"$\varphi$ (persistence)", "True $\\varphi$"),
          (r"$\sigma_\eta$ (vol of vol)", "True $\\sigma_\\eta$")]


def bias_plot(true, est, method, T):
    err = est - true
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for i, (ax, (title, xlabel)) in enumerate(zip(axes, PARAMS)):
        ax.scatter(true[:, i], err[:, i], alpha=0.45, s=18, color="#4878D0", edgecolors="none")
        ax.axhline(0, color="black", lw=1.0, ls="--", label="zero bias")
        order = np.argsort(true[:, i])
        xs, es = true[order, i], err[order, i]
        w = 30
        rm = np.convolve(es, np.ones(w) / w, mode="valid")
        ax.plot(xs[w // 2: w // 2 + len(rm)], rm, color="#EE854A", lw=1.8, label="running mean")
        # slope of est~true (shrinkage indicator)
        slope = np.polyfit(true[:, i], est[:, i], 1)[0]
        ax.set_title(f"{title}\nslope(est~true) = {slope:.2f}")
        ax.set_xlabel(xlabel); ax.set_ylabel("Estimated − True")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"{method} estimation error vs. true parameter — T = {T} (N = 200)", y=1.02)
    fig.tight_layout()
    out = f"figures/fig_bias_{method.lower()}_T{T}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {p[0].split()[0]: float(np.polyfit(true[:, i], est[:, i], 1)[0]) for i, p in enumerate(PARAMS)}


def main():
    print(f"{'method':>6} {'T':>6} | slope mu / phi / sigma (->1 = no shrinkage)")
    for T in (500, 1000, 2000):
        tcn = np.load(f"results/tcn_best_T{T}/predictions.npz")
        s_tcn = bias_plot(tcn["true"], tcn["preds"], "TCN", T)
        sv = np.load(f"results/stochvol_T{T}/results.npz")
        s_mc = bias_plot(sv["true_params"], sv["means"], "MCMC", T)
        print(f"  TCN  {T:>6} | {s_tcn['$\\mu$']:.2f} / {s_tcn['$\\varphi$']:.2f} / {s_tcn['$\\sigma_\\eta$']:.2f}")
        print(f"  MCMC {T:>6} | {s_mc['$\\mu$']:.2f} / {s_mc['$\\varphi$']:.2f} / {s_mc['$\\sigma_\\eta$']:.2f}")
    print("\nWrote figures/fig_bias_{tcn,mcmc}_T{500,1000,2000}.png")


if __name__ == "__main__":
    main()
