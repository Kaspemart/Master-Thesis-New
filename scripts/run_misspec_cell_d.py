"""
Cell (d): correctly-specified stochvol on the misspecified data.

For each scenario, runs the correct-model stochvol (leverage / t / both) on the
200 estimation-window series in parallel, via stochvol_runner_correct.R. Two
chains per series for R-hat. Saves estimates and reports RMSE/bias per parameter
(nu split by low/high true nu). Checkpoints per series (crash-safe).

Output: results/misspec/{scenario}_stochvol_correct.npz  (est, true, rhat)

Run:  uv run python scripts/run_misspec_cell_d.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

R_SCRIPT = "src/estimation/stochvol_runner_correct.R"
T_ESTIM = 1000
DRAWS, BURNIN = 1000, 1000
N_JOBS = 4
OUT = Path("results/misspec")

SPEC = {
    "asv":  ("data/test_misspec_asv.npz",  "asv",  ["mu", "phi", "sigma_eta", "rho"]),
    "svt":  ("data/test_misspec_svt.npz",  "svt",  ["mu", "phi", "sigma_eta", "nu"]),
    "asvt": ("data/test_misspec_asvt.npz", "asvt", ["mu", "phi", "sigma_eta", "rho", "nu"]),
}


def run_one(y, model, seed):
    """Run correct-model stochvol on one series; return (post_mean, rhat)."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        np.savetxt(f.name, y); csv = f.name
    out = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False).name
    try:
        r = subprocess.run(
            ["Rscript", R_SCRIPT, csv, out, model, str(DRAWS), str(BURNIN),
             str(seed), str(seed + 1)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-400:])
        res = json.load(open(out))
        return np.array(res["post_mean"]), np.array(res["rhat"])
    finally:
        Path(csv).unlink(missing_ok=True)
        Path(out).unlink(missing_ok=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for scen, (path, model, names) in SPEC.items():
        data = np.load(path)
        R = data["returns"][:, :T_ESTIM].astype(np.float64)
        true = data["params"].astype(np.float32)
        N = R.shape[0]
        print(f"[{scen}] running correct-model stochvol on {N} series ({model})...")

        results = Parallel(n_jobs=N_JOBS)(
            delayed(run_one)(R[i], model, 100 + 2 * i) for i in range(N)
        )
        est  = np.array([r[0] for r in results], dtype=np.float32)
        rhat = np.array([r[1] for r in results], dtype=np.float32)
        np.savez_compressed(OUT / f"{scen}_stochvol_correct.npz",
                            est=est, true=true, rhat=rhat)

        err = est - true
        rmse = np.sqrt((err ** 2).mean(axis=0))
        bias = err.mean(axis=0)
        print(f"  {scen}  (correct stochvol):")
        for i, nm in enumerate(names):
            print(f"    {nm:>10}: RMSE={rmse[i]:.4f}  bias={bias[i]:+.4f}  "
                  f"frac_rhat>1.1={(rhat[:, i] > 1.1).mean():.2f}")
        if "nu" in names:
            j = names.index("nu"); tn = true[:, j]
            for lo, hi in [(0, 10), (10, 100)]:
                m = (tn >= lo) & (tn < hi)
                if m.sum():
                    rm = np.sqrt(((est[m, j] - tn[m]) ** 2).mean())
                    print(f"      nu in [{lo},{hi}) (n={m.sum():3d}): RMSE={rm:.3f}")


if __name__ == "__main__":
    main()
