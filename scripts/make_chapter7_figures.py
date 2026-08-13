"""
Generate the Chapter 7 (empirical application) figures from results/real/*.json:

  figures/fig_ch7_oos_gap.png    — TCN minus MCMC out-of-sample predictive LL, per
                                    asset, SV and ASV-t panels (F2/F3: MCMC generally
                                    better; gap widens for ASV-t and concentrates in FX).
  figures/fig_ch7_rho.png        — leverage rho, MCMC vs TCN, coloured by asset class
                                    (F4: equities agree and are strongly negative; FX
                                    near zero; commodities disagree).
  figures/fig_ch7_asvt_gain.png  — ASV-t minus base-SV OOS predictive LL under MCMC
                                    (F1: the richer model helps most for equities).

Run:  uv run python scripts/make_chapter7_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.real_data import ASSET_GROUPS, LABELS

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 12})
Path("figures").mkdir(exist_ok=True)

GROUP_COLOR = {"FX": "#4878D0", "Equity": "#EE854A", "Commodity": "#6ACC64"}
ORDER = [(g, a) for g, m in ASSET_GROUPS.items() for a in m]           # grouped order
R = {a: json.load(open(f"results/real/{a}.json")) for _, a in ORDER}


def _yaxis_by_group(ax):
    labels = [LABELS[a] for _, a in ORDER]
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    for i, (g, _) in enumerate(ORDER):                                  # tick colour = group
        ax.get_yticklabels()[i].set_color(GROUP_COLOR[g])


def oos_gap_figure():
    colors = [GROUP_COLOR[g] for g, _ in ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, model, title in [(axes[0], "sv", "Base SV model"),
                             (axes[1], "asvt", "ASV-t model")]:
        diff = [R[a][model]["tcn"]["oos_ll"] - R[a][model]["mcmc"]["oos_ll"] for _, a in ORDER]
        ax.barh(range(len(ORDER)), diff, color=colors, alpha=0.85)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(title)
        ax.set_xlabel("TCN − MCMC  out-of-sample log-likelihood")
        ax.grid(axis="x", alpha=0.25)
    _yaxis_by_group(axes[0])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOR.values()]
    axes[1].legend(handles, GROUP_COLOR.keys(), fontsize=9, loc="lower left", title="Asset class")
    fig.suptitle("Out-of-sample predictive likelihood: TCN relative to MCMC "
                 "(left of 0 = MCMC better)", y=1.00)
    fig.tight_layout()
    fig.savefig("figures/fig_ch7_oos_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def rho_figure():
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for g, m in ASSET_GROUPS.items():
        xm = [R[a]["asvt"]["mcmc"]["params"][3] for a in m]
        yt = [R[a]["asvt"]["tcn"]["params"][3] for a in m]
        ax.scatter(xm, yt, s=70, color=GROUP_COLOR[g], label=g, edgecolors="k", linewidths=0.4, alpha=0.9)
        for a, x, y in zip(m, xm, yt):
            ax.annotate(LABELS[a], (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    lim = [-0.95, 0.35]
    ax.plot(lim, lim, "k--", lw=1, label="agreement (y = x)")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("ρ  (MCMC posterior mean)"); ax.set_ylabel("ρ  (TCN estimate)")
    ax.set_title("Leverage ρ: MCMC vs TCN\nequities agree (strong −ρ); commodities disagree")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("figures/fig_ch7_rho.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def params_figure():
    """MCMC vs TCN scatter for mu, phi, sigma_eta, nu (ASV-t), by asset class."""
    specs = [(0, "mu", "Log-volatility level  μ", (-10.8, -7.8)),
             (1, "phi", "Persistence  φ", (0.84, 1.0)),
             (2, "sigma", "Volatility of volatility  ση", (0.0, 0.4)),
             (4, "nu", "Degrees of freedom  ν", (3, 30))]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.8))
    axes = axes.ravel()
    for ax, (idx, key, title, lim) in zip(axes, specs):
        for g, m in ASSET_GROUPS.items():
            xm = [R[a]["asvt"]["mcmc"]["params"][idx] for a in m]
            yt = [R[a]["asvt"]["tcn"]["params"][idx] for a in m]
            ax.scatter(xm, yt, s=55, color=GROUP_COLOR[g], label=g,
                       edgecolors="k", linewidths=0.4, alpha=0.9)
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("MCMC"); ax.set_ylabel("TCN"); ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Parameter estimates, MCMC vs TCN (ASV-t model), by asset class", y=1.00)
    fig.tight_layout()
    fig.savefig("figures/fig_ch7_params.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def asvt_gain_figure():
    colors = [GROUP_COLOR[g] for g, _ in ORDER]
    gain = [R[a]["asvt"]["mcmc"]["oos_ll"] - R[a]["sv"]["mcmc"]["oos_ll"] for _, a in ORDER]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.barh(range(len(ORDER)), gain, color=colors, alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    _yaxis_by_group(ax)
    ax.set_xlabel("ASV-t − base SV  out-of-sample log-likelihood (MCMC)")
    ax.set_title("Gain from the ASV-t model over base SV\n(right of 0 = ASV-t better)")
    ax.grid(axis="x", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOR.values()]
    ax.legend(handles, GROUP_COLOR.keys(), fontsize=9, loc="lower right", title="Asset class")
    fig.tight_layout()
    fig.savefig("figures/fig_ch7_asvt_gain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    oos_gap_figure()
    rho_figure()
    params_figure()
    asvt_gain_figure()
    print("Wrote figures/fig_ch7_oos_gap.png, fig_ch7_rho.png, fig_ch7_params.png, fig_ch7_asvt_gain.png")
