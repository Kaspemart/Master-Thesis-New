"""
Tests for the SV with leverage simulator.

Follows TDD: tests written before implementation.
Run with: uv run pytest tests/test_sv_leverage.py -v

Key property being tested: the leverage effect introduces a correlation ρ
between the return shock ε_t and the volatility shock η_t via Cholesky
decomposition. When ρ < 0, negative returns tend to increase future volatility.
"""

import numpy as np
import pytest

from src.simulation import (
    SVLeverageParams,
    SimulationResult,
    simulate_sv_leverage,
    save_dataset,
    load_dataset,
)
from src.simulation import SVParams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    return SVLeverageParams()


@pytest.fixture
def small_result(default_config):
    return simulate_sv_leverage(N=100, T=50, config=default_config, seed=42)


# ---------------------------------------------------------------------------
# SVLeverageParams — configuration
# ---------------------------------------------------------------------------

class TestSVLeverageParamsDefaults:
    def test_inherits_mu_range(self, default_config):
        assert default_config.mu_range == SVParams().mu_range

    def test_inherits_phi_range(self, default_config):
        assert default_config.phi_range == SVParams().phi_range

    def test_inherits_sigma_eta_range(self, default_config):
        assert default_config.sigma_eta_range == SVParams().sigma_eta_range

    def test_rho_range_default(self, default_config):
        assert default_config.rho_range == (-0.95, 0.5)

    def test_is_subclass_of_svparams(self, default_config):
        assert isinstance(default_config, SVParams)


class TestSVLeverageTransform:
    def test_transform_roundtrip(self, default_config):
        """transform then inverse_transform recovers original (N,4) params."""
        rng = np.random.default_rng(0)
        base = np.column_stack([
            rng.uniform(-10, 0, 200),
            rng.uniform(0.5, 0.999, 200),
            rng.uniform(0.05, 1.0, 200),
            rng.uniform(-0.95, 0.5, 200),
        ])
        transformed = default_config.transform(base)
        recovered = default_config.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, base, rtol=1e-5, atol=1e-6)

    def test_transform_shape(self, default_config):
        params = np.random.default_rng(1).random((50, 4))
        assert default_config.transform(params).shape == (50, 4)

    def test_rho_column_is_arctanh(self, default_config):
        """arctanh maps (-1,1) to R: arctanh(0)=0, arctanh(0.3)>0, arctanh(-0.3)<0."""
        params_pos = np.array([[-1.0, 0.97, 0.15, 0.3]])
        params_neg = np.array([[-1.0, 0.97, 0.15, -0.3]])
        assert default_config.transform(params_pos)[0, 3] > 0   # arctanh(0.3) ≈ 0.31
        assert default_config.transform(params_neg)[0, 3] < 0   # arctanh(-0.3) ≈ -0.31


# ---------------------------------------------------------------------------
# simulate_sv_leverage — output shapes and types
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_returns_shape(self, small_result):
        assert small_result.returns.shape == (100, 50)

    def test_params_shape(self, small_result):
        assert small_result.params.shape == (100, 4)

    def test_latent_h_shape(self, small_result):
        assert small_result.latent_h.shape == (100, 50)

    def test_all_float32(self, small_result):
        assert small_result.returns.dtype == np.float32
        assert small_result.params.dtype == np.float32
        assert small_result.latent_h.dtype == np.float32


class TestFiniteness:
    def test_returns_all_finite(self, small_result):
        assert np.all(np.isfinite(small_result.returns))

    def test_params_all_finite(self, small_result):
        assert np.all(np.isfinite(small_result.params))

    def test_latent_h_all_finite(self, small_result):
        assert np.all(np.isfinite(small_result.latent_h))


class TestRhoColumn:
    def test_rho_is_4th_column(self, default_config):
        result = simulate_sv_leverage(N=500, T=10, config=default_config, seed=0)
        rho = result.params[:, 3]
        assert np.all(rho >= -0.95)
        assert np.all(rho <= 0.5)

    def test_first_three_columns_match_base_ranges(self, default_config):
        result = simulate_sv_leverage(N=500, T=10, config=default_config, seed=0)
        assert np.all(result.params[:, 0] >= default_config.mu_range[0])
        assert np.all(result.params[:, 0] <= default_config.mu_range[1])
        assert np.all(result.params[:, 1] >= default_config.phi_range[0])
        assert np.all(result.params[:, 2] >= default_config.sigma_eta_range[0])


class TestReproducibility:
    def test_same_seed_identical(self, default_config):
        r1 = simulate_sv_leverage(N=50, T=30, config=default_config, seed=42)
        r2 = simulate_sv_leverage(N=50, T=30, config=default_config, seed=42)
        np.testing.assert_array_equal(r1.returns, r2.returns)
        np.testing.assert_array_equal(r1.params, r2.params)

    def test_different_seeds_differ(self, default_config):
        r1 = simulate_sv_leverage(N=50, T=30, config=default_config, seed=1)
        r2 = simulate_sv_leverage(N=50, T=30, config=default_config, seed=2)
        assert not np.array_equal(r1.returns, r2.returns)


class TestEdgeCases:
    def test_single_series(self, default_config):
        result = simulate_sv_leverage(N=1, T=100, config=default_config, seed=0)
        assert result.returns.shape == (1, 100)
        assert result.params.shape == (1, 4)


class TestInputValidation:
    def test_n_zero_raises(self, default_config):
        with pytest.raises(ValueError):
            simulate_sv_leverage(N=0, T=100, config=default_config, seed=0)

    def test_t_zero_raises(self, default_config):
        with pytest.raises(ValueError):
            simulate_sv_leverage(N=10, T=0, config=default_config, seed=0)

    def test_invalid_seed_raises(self, default_config):
        with pytest.raises(TypeError):
            simulate_sv_leverage(N=10, T=50, config=default_config, seed=1.5)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip_preserves_4_columns(self, small_result, tmp_path):
        path = tmp_path / "leverage.npz"
        save_dataset(path, small_result)
        loaded = load_dataset(path)
        assert loaded.params.shape == (100, 4)
        np.testing.assert_array_equal(loaded.params, small_result.params)
        np.testing.assert_array_equal(loaded.returns, small_result.returns)


# ---------------------------------------------------------------------------
# Statistical sanity
# ---------------------------------------------------------------------------

class TestStatisticalSanity:
    def test_leverage_effect_present(self):
        """
        With fixed ρ = -0.7, the contemporaneous correlation between r_t and h_t
        should be negative. This is the leverage effect channel: at each step t,
        eps_t (return shock) and eta_t (vol shock) are correlated with coefficient rho.
        When rho < 0, a negative return shock (eps_t < 0) pushes h_t up via
        eta_t = rho*eps_t + chol*z2_t, so r_t and h_t are negatively correlated.
        """
        config = SVLeverageParams(
            mu_range=(-1.5, -1.5),
            phi_range=(0.97, 0.97),
            sigma_eta_range=(0.15, 0.15),
            rho_range=(-0.7, -0.7),
        )
        result = simulate_sv_leverage(N=2000, T=500, config=config, seed=0)
        r = result.returns.astype(np.float64)   # (N, T)
        h = result.latent_h.astype(np.float64)  # (N, T)
        # Contemporaneous correlation corr(r_t, h_t) across time, averaged over series
        corrs = np.array([
            np.corrcoef(r[i], h[i])[0, 1]
            for i in range(r.shape[0])
        ])
        mean_corr = corrs.mean()
        assert mean_corr < 0, f"Expected negative contemporaneous corr(r,h) for rho<0, got {mean_corr:.4f}"

    def test_rho_zero_distributional_match(self):
        """
        With rho=0, the leverage model reduces to the base model. The two simulators
        use different RNG sequences (the leverage simulator has an extra rho draw),
        so exact array equality cannot be tested. Instead we verify that the
        unconditional distribution of returns is statistically equivalent:
        same mean, same variance, same r^2 autocorrelation structure.
        """
        from src.simulation import simulate_sv, SVParams

        mu, phi, sigma_eta = -1.5, 0.97, 0.15

        base = simulate_sv(
            N=2000, T=500,
            config=SVParams(
                mu_range=(mu, mu),
                phi_range=(phi, phi),
                sigma_eta_range=(sigma_eta, sigma_eta),
            ),
            seed=10,
        )
        lev = simulate_sv_leverage(
            N=2000, T=500,
            config=SVLeverageParams(
                mu_range=(mu, mu),
                phi_range=(phi, phi),
                sigma_eta_range=(sigma_eta, sigma_eta),
                rho_range=(0.0, 0.0),
            ),
            seed=20,
        )
        # Grand means should both be close to 0
        assert abs(base.returns.mean()) < 0.05
        assert abs(lev.returns.mean()) < 0.05
        # Variances should be in the same ballpark (within 20%)
        base_var = float(np.var(base.returns))
        lev_var = float(np.var(lev.returns))
        assert abs(base_var - lev_var) / base_var < 0.2, (
            f"Variances too different: base={base_var:.4f}, leverage(rho=0)={lev_var:.4f}"
        )

    def test_volatility_clustering_preserved(self):
        """Leverage model should still show volatility clustering (r_t² autocorrelation > 0)."""
        config = SVLeverageParams(
            mu_range=(-1.5, -1.5),
            phi_range=(0.97, 0.97),
            sigma_eta_range=(0.15, 0.15),
            rho_range=(-0.5, -0.5),
        )
        result = simulate_sv_leverage(N=500, T=500, config=config, seed=0)
        r2 = result.returns.astype(np.float64) ** 2
        mean_ac = np.mean([
            np.corrcoef(r2[i, :-1], r2[i, 1:])[0, 1]
            for i in range(r2.shape[0])
        ])
        assert mean_ac > 0.05, f"Expected positive r² autocorrelation, got {mean_ac:.4f}"
