"""
Tests for the base SV model simulator.

Follows TDD: tests are written before implementation.
Run with: uv run pytest tests/test_simulator.py -v
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from src.simulation import (
    SVParams,
    SimulationResult,
    draw_parameters,
    load_dataset,
    save_dataset,
    simulate_sv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    return SVParams()


@pytest.fixture
def small_result(default_config):
    return simulate_sv(N=100, T=50, config=default_config, seed=42)


# ---------------------------------------------------------------------------
# Phase 2 — SVParams and draw_parameters
# ---------------------------------------------------------------------------

class TestSVParamsDefaults:
    def test_mu_range(self, default_config):
        assert default_config.mu_range == (-10.0, 0.0)

    def test_phi_range(self, default_config):
        assert default_config.phi_range == (0.5, 0.999)

    def test_sigma_eta_range(self, default_config):
        assert default_config.sigma_eta_range == (0.05, 1.0)


class TestDrawParameters:
    def test_shape(self, default_config):
        rng = np.random.default_rng(0)
        params = draw_parameters(500, default_config, rng)
        assert params.shape == (500, 3)

    def test_dtype_is_float32(self, default_config):
        rng = np.random.default_rng(0)
        params = draw_parameters(100, default_config, rng)
        assert params.dtype == np.float32

    def test_mu_within_bounds(self, default_config):
        rng = np.random.default_rng(1)
        params = draw_parameters(10_000, default_config, rng)
        assert np.all(params[:, 0] >= default_config.mu_range[0])
        assert np.all(params[:, 0] <= default_config.mu_range[1])

    def test_phi_within_bounds(self, default_config):
        rng = np.random.default_rng(1)
        params = draw_parameters(10_000, default_config, rng)
        assert np.all(params[:, 1] >= default_config.phi_range[0])
        assert np.all(params[:, 1] <= default_config.phi_range[1])

    def test_sigma_eta_within_bounds(self, default_config):
        rng = np.random.default_rng(1)
        params = draw_parameters(10_000, default_config, rng)
        assert np.all(params[:, 2] >= default_config.sigma_eta_range[0])
        assert np.all(params[:, 2] <= default_config.sigma_eta_range[1])

    def test_reproducible(self, default_config):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        p1 = draw_parameters(200, default_config, rng1)
        p2 = draw_parameters(200, default_config, rng2)
        np.testing.assert_array_equal(p1, p2)


class TestTransformRoundtrip:
    def test_roundtrip_recovers_original(self, default_config):
        rng = np.random.default_rng(7)
        params = draw_parameters(500, default_config, rng).astype(np.float64)
        transformed = default_config.transform(params)
        recovered = default_config.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, params, rtol=1e-5, atol=1e-6)

    def test_transform_output_is_unconstrained(self, default_config):
        """Transformed φ and σ_η should not be bounded to their original domains."""
        rng = np.random.default_rng(8)
        params = draw_parameters(1000, default_config, rng).astype(np.float64)
        transformed = default_config.transform(params)
        # logit(phi) for phi in (0.5, 0.999): logit(0.5)=0, logit(0.999)≈6.9
        # Values should be well outside the original [0.5, 0.999] domain
        assert transformed[:, 1].max() > 5.0   # logit(0.999) ≈ 6.9
        assert transformed[:, 1].min() >= 0.0  # logit(0.5) = 0
        # log(sigma_eta) for sigma_eta in (0.05, 1.0): log(0.05)≈-3, log(1.0)=0
        assert transformed[:, 2].min() < -2.0  # well below the original domain


# ---------------------------------------------------------------------------
# Phase 3 — simulate_sv
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_returns_shape(self, small_result):
        assert small_result.returns.shape == (100, 50)

    def test_params_shape(self, small_result):
        assert small_result.params.shape == (100, 3)

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


class TestReproducibility:
    def test_same_seed_identical(self, default_config):
        r1 = simulate_sv(N=50, T=30, config=default_config, seed=42)
        r2 = simulate_sv(N=50, T=30, config=default_config, seed=42)
        np.testing.assert_array_equal(r1.returns, r2.returns)
        np.testing.assert_array_equal(r1.params, r2.params)
        np.testing.assert_array_equal(r1.latent_h, r2.latent_h)

    def test_different_seeds_differ(self, default_config):
        r1 = simulate_sv(N=50, T=30, config=default_config, seed=1)
        r2 = simulate_sv(N=50, T=30, config=default_config, seed=2)
        assert not np.array_equal(r1.returns, r2.returns)


class TestEdgeCases:
    def test_single_series(self, default_config):
        result = simulate_sv(N=1, T=100, config=default_config, seed=0)
        assert result.returns.shape == (1, 100)
        assert result.latent_h.shape == (1, 100)
        assert result.params.shape == (1, 3)

    def test_short_series(self, default_config):
        result = simulate_sv(N=10, T=10, config=default_config, seed=0)
        assert result.returns.shape == (10, 10)
        assert np.all(np.isfinite(result.returns))

    def test_high_persistence_clips_and_warns(self):
        """With near-unit-root phi and large sigma_eta, h_t must be clipped.

        phi=0.9999, sigma_eta=1.0 → stationary std ≈ 70.7, so h_0 will
        routinely exceed H_CLIP=50, guaranteeing the clip path is exercised.
        """
        config = SVParams(phi_range=(0.9999, 0.9999), sigma_eta_range=(1.0, 1.0), mu_range=(-1.0, -1.0))
        with pytest.warns(UserWarning, match="[Cc]lip"):
            result = simulate_sv(N=50, T=200, config=config, seed=0)
        assert np.all(np.isfinite(result.returns))


class TestInputValidation:
    def test_n_zero_raises(self, default_config):
        with pytest.raises(ValueError):
            simulate_sv(N=0, T=100, config=default_config, seed=0)

    def test_t_zero_raises(self, default_config):
        with pytest.raises(ValueError):
            simulate_sv(N=10, T=0, config=default_config, seed=0)

    def test_invalid_seed_type_raises(self, default_config):
        with pytest.raises(TypeError):
            simulate_sv(N=10, T=50, config=default_config, seed=1.5)


# ---------------------------------------------------------------------------
# Phase 3 — save_dataset / load_dataset
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, small_result, tmp_path):
        path = tmp_path / "test.npz"
        save_dataset(path, small_result)
        loaded = load_dataset(path)
        np.testing.assert_array_equal(loaded.returns, small_result.returns)
        np.testing.assert_array_equal(loaded.params, small_result.params)
        np.testing.assert_array_equal(loaded.latent_h, small_result.latent_h)

    def test_creates_parent_dirs(self, small_result, tmp_path):
        path = tmp_path / "nested" / "dir" / "data.npz"
        save_dataset(path, small_result)
        assert path.exists()

    def test_file_is_npz(self, small_result, tmp_path):
        path = tmp_path / "data.npz"
        save_dataset(path, small_result)
        data = np.load(path)
        assert set(data.files) == {"returns", "params", "latent_h"}


# ---------------------------------------------------------------------------
# Phase 3 — statistical sanity
# ---------------------------------------------------------------------------

class TestStatisticalSanity:
    def test_volatility_clustering(self):
        """
        Squared returns r_t^2 should show positive autocorrelation at lag 1.
        This is a necessary (not sufficient) condition for the SV model to be
        simulating correctly — volatility clustering is the key empirical feature.
        """
        config = SVParams(
            mu_range=(-1.5, -1.5),
            phi_range=(0.97, 0.97),
            sigma_eta_range=(0.15, 0.15),
        )
        result = simulate_sv(N=2000, T=500, config=config, seed=0)
        r2 = result.returns ** 2  # (2000, 500)
        # Lag-1 autocorrelation of r^2 for each series, then average
        mean_ac = np.mean([
            np.corrcoef(r2[i, :-1], r2[i, 1:])[0, 1]
            for i in range(r2.shape[0])
        ])
        assert mean_ac > 0.05, f"Expected positive autocorrelation, got {mean_ac:.4f}"

    def test_returns_approximately_zero_mean(self):
        """Returns should have mean close to 0 (SV model has no drift)."""
        result = simulate_sv(N=500, T=1000, seed=0)
        grand_mean = result.returns.mean()
        assert abs(grand_mean) < 0.05, f"Grand mean too large: {grand_mean:.4f}"
