"""
Generate the misspecification-analysis test data (Chapter 6).

Three scenarios — ASV, SV-t, ASV-t — each N=200 series. Each series is a
continuous path of length 2000: the first 1000 observations are the estimation
window (T=1000), the next 1000 are the out-of-sample continuation under the same
parameters and latent path, used for the out-of-sample predictive log-likelihood.

Seed 999 is used for all three scenarios so the base parameters (mu, phi,
sigma_eta) are the SAME draws across scenarios — a controlled property so that
cross-scenario differences reflect the added feature (leverage / fat tails)
rather than a different underlying SV process.

Output: data/test_misspec_{asv,svt,asvt}.npz, each with
    returns   (200, 2000) float32
    params    (200, k)     float32  — k = 4 (asv, svt) or 5 (asvt)
    latent_h  (200, 2000) float32
    T_estim   scalar 1000
    T_oos     scalar 1000

Run:  uv run python scripts/generate_misspec_test_data.py
"""

from __future__ import annotations

import numpy as np

from src.simulation.simulator import (
    simulate_sv_leverage,
    simulate_sv_t,
    simulate_asv_t,
)

N = 200
T_TOTAL = 2000          # 1000 estimation + 1000 out-of-sample
T_ESTIM = 1000
T_OOS = 1000
SEED = 999


def save(path, result):
    np.savez_compressed(
        path,
        returns=result.returns,
        params=result.params,
        latent_h=result.latent_h,
        T_estim=np.int64(T_ESTIM),
        T_oos=np.int64(T_OOS),
    )
    print(f"  saved {path}  returns={result.returns.shape}  params={result.params.shape}")


def main():
    print("Generating misspecification test data (N=200, path length 2000, seed 999)")
    print("-" * 66)

    print("ASV (leverage):")
    save("data/test_misspec_asv.npz",
         simulate_sv_leverage(N=N, T=T_TOTAL, seed=SEED))

    print("SV-t (fat tails):")
    save("data/test_misspec_svt.npz",
         simulate_sv_t(N=N, T=T_TOTAL, seed=SEED))

    print("ASV-t (leverage + fat tails):")
    save("data/test_misspec_asvt.npz",
         simulate_asv_t(N=N, T=T_TOTAL, seed=SEED))

    # Confirm shared base parameters across scenarios (controlled property).
    a = np.load("data/test_misspec_asv.npz")["params"][:, :3]
    s = np.load("data/test_misspec_svt.npz")["params"][:, :3]
    t = np.load("data/test_misspec_asvt.npz")["params"][:, :3]
    print("-" * 66)
    print(f"base params identical across scenarios (asv==svt): {np.allclose(a, s)}")
    print(f"base params identical across scenarios (asv==asvt): {np.allclose(a, t)}")


if __name__ == "__main__":
    main()
