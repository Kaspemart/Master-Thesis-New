"""
Cell (c): correctly-specified NN estimation on the misspecified data.

Each correct-model TCN estimates the FULL parameter set of its DGP on the
estimation window (first 1000 obs). RMSE/bias reported per parameter; for the
t-variants, nu accuracy is additionally split by low vs high true nu (nu is
weakly identified at the high end).

Output: results/misspec/{scenario}_tcn_correct.npz  (est, true)

Run:  uv run python scripts/run_misspec_cell_c.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.tcn import SVTCNNet
from src.simulation.sv_params import SVLeverageParams, SVtParams, ASVtParams

T_ESTIM = 1000
OUT = Path("results/misspec")

# scenario -> (data, config, n_outputs, param names, checkpoint)
SPEC = {
    "asv":  ("data/test_misspec_asv.npz",  SVLeverageParams(), 4,
             ["mu", "phi", "sigma_eta", "rho"],       "asv_correct_T1000"),
    "svt":  ("data/test_misspec_svt.npz",  SVtParams(),        4,
             ["mu", "phi", "sigma_eta", "nu"],        "svt_correct_T1000"),
    "asvt": ("data/test_misspec_asvt.npz", ASVtParams(),       5,
             ["mu", "phi", "sigma_eta", "rho", "nu"], "asvt_correct_T1000"),
}


def main():
    print("=" * 60)
    print("CELL (c) — correctly-specified TCN on misspecified data")
    print("=" * 60)
    for scen, (path, cfg, nout, names, ckpt) in SPEC.items():
        data = np.load(path)
        r = data["returns"][:, :T_ESTIM].astype(np.float32)
        true = data["params"].astype(np.float32)              # (N, nout)

        model = SVTCNNet(n_channels=32, kernel_size=7, n_blocks=6,
                         dropout=0.0, n_outputs=nout)
        model.load_state_dict(torch.load(f"checkpoints/{ckpt}/best.pt",
                                         map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            preds_t = model(torch.from_numpy(r)).cpu().numpy()
        est = cfg.inverse_transform(preds_t.astype(np.float64)).astype(np.float32)
        np.savez_compressed(OUT / f"{scen}_tcn_correct.npz", est=est, true=true)

        err = est - true
        rmse = np.sqrt((err ** 2).mean(axis=0))
        bias = err.mean(axis=0)
        print(f"\n  {scen}  (correct TCN):")
        for i, nm in enumerate(names):
            print(f"    {nm:>10}: RMSE={rmse[i]:.4f}  bias={bias[i]:+.4f}")

        # nu split (weak identification at high nu)
        if "nu" in names:
            j = names.index("nu")
            tn = true[:, j]
            for lo, hi in [(0, 10), (10, 100)]:
                m = (tn >= lo) & (tn < hi)
                if m.sum():
                    rm = np.sqrt(((est[m, j] - tn[m]) ** 2).mean())
                    bi = (est[m, j] - tn[m]).mean()
                    print(f"      nu in [{lo},{hi}) (n={m.sum():3d}): "
                          f"RMSE={rm:.3f} bias={bi:+.3f}")


if __name__ == "__main__":
    main()
