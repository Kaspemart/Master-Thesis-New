# Chapter 7 — Empirical Application: Consolidated Results

**Data.** 15 assets (5 FX, 5 equity index, 5 commodity ETF/index), daily decimal log-returns.
In-sample 2010–2017 (~2000 obs), out-of-sample 2018–2025 (~2000 obs). Returns demeaned by
the in-sample mean. FX standardised to USD-per-foreign (CHF/USD, JPY/USD inverted).

**Estimators.** Base SV and ASV-t (leverage + Student-t), each by stochvol MCMC and by the
TCN. base SV → T=2000 network; ASV-t → T=1000 sign-input network (see methodology note 1).

**Metric.** Out-of-sample one-step predictive log-likelihood via bootstrap particle filter
(10,000 particles, mean over 3 seeds), state initialised on the in-sample window (no look-ahead).

## Table 1 — Out-of-sample predictive log-likelihood (higher = better)

| Asset | SV MCMC | SV TCN | ASV-t MCMC | ASV-t TCN | ASV-t−SV (MCMC) |
|---|--:|--:|--:|--:|--:|
| **FX** | | | | | |
| EUR/USD | 8084.2 | 8015.1 | 8097.6 | 7973.1 | 13.3 |
| GBP/USD | 7707.0 | 7687.1 | 7714.2 | 7666.8 | 7.2 |
| CHF/USD | 8000.1 | 7965.8 | 8025.2 | 7948.2 | 25.1 |
| JPY/USD | 7747.7 | 7737.5 | 7781.1 | 7746.4 | 33.4 |
| AUD/USD | 7397.6 | 7386.0 | 7407.5 | 7363.9 | 9.8 |
| **Equity** | | | | | |
| S&P 500 | 6510.5 | 6485.1 | 6565.3 | 6564.6 | 54.8 |
| Nikkei 225 | 5918.8 | 5919.3 | 5950.7 | 5940.9 | 31.9 |
| ASX 200 | 6912.3 | 6916.7 | 6945.5 | 6936.6 | 33.2 |
| EuroStoxx 50 | 6403.0 | 6405.4 | 6466.1 | 6457.4 | 63.1 |
| FTSE 100 | 6892.2 | 6890.1 | 6930.6 | 6925.2 | 38.3 |
| **Commodity** | | | | | |
| Gold (GLD) | 6655.1 | 6648.1 | 6668.0 | 6631.5 | 12.9 |
| Silver (SLV) | 5419.5 | 5393.1 | 5459.4 | 5400.4 | 39.9 |
| Oil (USO) | 4950.7 | 4958.0 | 4956.6 | 4966.8 | 5.9 |
| Commodities (DBC) | 6320.5 | 6308.7 | 6319.9 | 6314.2 | -0.7 |
| Agriculture (DBA) | 6814.9 | 6799.3 | 6815.3 | 6794.1 | 0.4 |

## Table 2 — ASV-t parameter estimates (posterior mean / TCN point estimate)

| Asset | μ (MCMC/TCN) | φ (MCMC/TCN) | σ_η (MCMC/TCN) | ρ (MCMC/TCN) | ν (MCMC/TCN) | max R̂ |
|---|---|---|---|---|---|--:|
| **FX** | | | | | | |
| EUR/USD | -10.43/-10.22 | 0.995/0.939 | 0.06/0.09 | -0.25/-0.08 | 10.2/7.2 | 1.006 |
| GBP/USD | -10.35/-10.37 | 0.992/0.922 | 0.08/0.09 | -0.34/-0.28 | 11.0/10.2 | 1.076 |
| CHF/USD | -10.32/-10.05 | 0.991/0.952 | 0.08/0.12 | 0.09/0.09 | 6.6/7.7 | 1.050 |
| JPY/USD | -10.31/-10.18 | 0.984/0.924 | 0.11/0.22 | -0.09/-0.03 | 6.2/6.3 | 1.099 |
| AUD/USD | -10.07/-9.89 | 0.994/0.951 | 0.07/0.09 | -0.36/-0.26 | 15.4/12.3 | 1.080 |
| **Equity** | | | | | | |
| S&P 500 | -9.85/-9.71 | 0.951/0.956 | 0.33/0.27 | -0.73/-0.68 | 11.3/9.0 | 1.063 |
| Nikkei 225 | -8.92/-8.79 | 0.937/0.939 | 0.29/0.21 | -0.56/-0.73 | 16.6/11.3 | 1.026 |
| ASX 200 | -9.67/-9.48 | 0.977/0.968 | 0.16/0.13 | -0.67/-0.65 | 28.3/16.5 | 1.084 |
| EuroStoxx 50 | -9.00/-8.83 | 0.969/0.964 | 0.23/0.19 | -0.75/-0.76 | 12.5/9.7 | 1.050 |
| FTSE 100 | -9.70/-9.46 | 0.960/0.953 | 0.25/0.22 | -0.69/-0.75 | 25.4/12.9 | 1.025 |
| **Commodity** | | | | | | |
| Gold (GLD) | -9.29/-9.15 | 0.984/0.951 | 0.11/0.12 | 0.05/-0.41 | 5.8/5.3 | 1.005 |
| Silver (SLV) | -8.21/-8.10 | 0.991/0.865 | 0.10/0.20 | 0.12/-0.16 | 4.6/4.6 | 1.021 |
| Oil (USO) | -8.13/-8.07 | 0.990/0.976 | 0.10/0.12 | -0.54/-0.36 | 15.0/13.9 | 1.020 |
| Commodities (DBC) | -9.34/-9.25 | 0.994/0.948 | 0.08/0.12 | -0.44/-0.45 | 10.1/12.3 | 1.058 |
| Agriculture (DBA) | -9.79/-9.64 | 0.988/0.887 | 0.10/0.16 | -0.02/-0.27 | 21.1/17.2 | 1.063 |

## Aggregate statistics

Differences are judged against the particle-filter noise floor: per-seed SD ≈ 0.2–0.45 LL
(SE ≈ 0.13–0.26 for the reported 3-seed means), so |Δ| ≤ 0.75 LL is treated as a tie.
Cross-asset paired tests use the Wilcoxon signed-rank test over the 15 assets.

- **SV model** — TCN−MCMC OOS LL: mean −14.6, median −11.7. MCMC better on 11, TCN better on
  3 (ASX, EuroStoxx, Oil), 1 tie (Nikkei). Wilcoxon p = 0.007 → MCMC significantly better.
    - FX: mean −29.0 · Equity: mean −4.0 · Commodity: mean −10.7
- **ASV-t model** — TCN−MCMC OOS LL: mean −31.5, median −21.2. MCMC better on 13, TCN better
  on 1 (Oil), 1 tie (S&P 500). Wilcoxon p = 0.001 → MCMC significantly better.
    - FX: mean −65.4 · Equity: mean −6.7 · Commodity: mean −22.4
- **ASV-t over SV (MCMC)**: mean +24.6 LL. 13 clear improvements, 2 ties (DBC, DBA), none
  significantly worse. Wilcoxon p = 0.0002 → the richer model significantly improves fit.

## Findings

**F1 — The richer ASV-t model improves out-of-sample fit almost everywhere (MCMC: 13/15
clear improvements, 2 ties, none worse; mean +24.6 LL, Wilcoxon p = 0.0002).** Gains are
largest for equities (+32 to +63), moderate for FX (+7 to +33, driven by fat tails rather
than leverage), and split within commodities: precious metals benefit (Silver +40, Gold +13)
while broad indices are a wash (Oil +6, DBA and DBC within noise). So the leverage-plus-fat-
tails extension pays off precisely where the stylised facts are strongest.

**F2 — MCMC significantly outperforms the TCN on out-of-sample predictive likelihood, and the
gap widens for the richer model.** SV: MCMC better on 11/15, TCN on 3 (ASX, EuroStoxx, Oil),
one tie; mean −14.6 LL, Wilcoxon p = 0.007. ASV-t: MCMC better on 13/15, TCN on 1 (Oil), S&P
500 a tie; mean −31.5 LL, p = 0.001. The amortised network is competitive but does not
overtake a correctly-specified MCMC fit on real data. Differences above ~0.75 LL are real
(particle-filter noise SD ≈ 0.2–0.45).

**F3 — The TCN's disadvantage concentrates where the extra parameters are weakly identified.**
By group, the ASV-t TCN−MCMC gap is smallest for equities (−6.7) and largest for FX (−65.4).
On equities the ASV-t TCN essentially ties MCMC (S&P 500 −0.7). On FX the ASV-t TCN performs
*worse than its own SV TCN* (e.g. EUR/USD 7973 vs 8015): when leverage is near zero, adding ρ
and ν injects estimation noise the amortised estimator cannot regularise away, whereas MCMC's
priors shrink weakly-identified parameters gracefully. This is the empirical counterpart to the
misspecification finding — amortised inference matches the Bayesian benchmark when the model is
well-identified and loses ground when it is not.

**F4 — Leverage recovery is asset-class dependent, and the two methods diverge exactly where ρ
is weak.** Equities: both methods find strong negative ρ (−0.56 to −0.75) and agree closely —
the textbook leverage effect. FX: both near zero, agree. Commodities: the methods *disagree in
sign* — Gold (MCMC +0.05 vs TCN −0.41), Silver (+0.12 vs −0.16), Agriculture (−0.02 vs −0.27).
The TCN systematically imposes more-negative ρ than MCMC on weakly-leveraged assets, consistent
with its training distribution (ρ range (−0.95, 0.5), negative-skewed): with little signal the
amortised estimator shrinks toward its negative training mean, while MCMC shrinks toward zero.
The TCN carries an equity-like leverage bias.

**F5 — The TCN shrinks persistence φ toward the interior of its training range.** Real FX and
commodity vols are near-unit-root (MCMC φ ≈ 0.98–0.995); the TCN pulls these down (EUR/USD
0.995→0.939, Silver 0.991→0.865, DBA 0.988→0.887). Where true persistence sits at the edge of
(or beyond) the training range, the amortised estimator underestimates it — a real limitation
that also contributes to its predictive-likelihood shortfall on the most persistent series.

**F6 — Fat tails are universal.** All ν are finite; heaviest for precious metals (Gold/Silver
ν ≈ 4.6–5.8), lighter for equities and broad indices (ν ≈ 10–28). MCMC and TCN agree on the
ordering, though the TCN compresses the range (rarely exceeds ~17).

**F7 — Jump episodes stress all four estimators (discussion).** The out-of-sample window
(2018–2025) spans the March-2020 COVID crash and the 2020–2022 energy/commodity shocks — the
2020 negative-oil episode is extreme for USO. None of the models includes jumps, so all take a
predictive-likelihood hit on those days. Oil is the sole asset where the TCN beats MCMC on both
models and the only equity/commodity where ASV-t barely helps, reflecting its idiosyncratic
roll costs and 2020 dislocation rather than a genuine volatility-model advantage.

## Methodology notes (for the Ch7 methodology section)

**Note 1 — Per-model network selection via length-invariance.** The TCN uses global average
pooling and is therefore sequence-length invariant; we select, for each model, the network with
the best held-out parameter recovery on 2000-observation simulated series (matching the real
window). base SV uses the T=2000 network (best σ_η, validated in Ch5). ASV-t uses the **T=1000**
sign-input network: training the ASV-t network directly at T=2000 triggered a multi-task
optimisation pathology in which the strong-signal parameters (μ, φ, σ_η, ν) dominated the loss
and the leverage parameter collapsed to a shrinkage solution (held-out ρ correlation 0.47,
estimates ~half their true magnitude). The T=1000 network recovers ρ at correlation 0.92 on
2000-obs series and matches the T=2000 network on all other parameters, so it is the better
estimator for the leverage specification despite the shorter training length.

**Note 2 — MCMC sampling.** stochvol with simulation-consistent priors (the same parameter
ranges the networks were trained on, so both methods are confined to identical support — a fair
comparison). Base SV: 1000 draws after 1000 burn-in, 2 chains (R-hat < 1.04). ASV-t: 8000 draws
after 6000 burn-in; the two most persistent FX vols (EUR/USD, CHF/USD; φ ≈ 0.99, near unit root)
were raised to 20000/10000 to bring φ below R-hat 1.1. Final: every asset has max R-hat < 1.1.

**Note 3 — Predictive-likelihood evaluation.** Out-of-sample one-step predictive log-likelihood
via bootstrap particle filter (10,000 particles, systematic resampling, mean over 3 seeds). The
filter is run over the full series with the state initialised on the in-sample window and the
likelihood accumulated only over the out-of-sample portion (t_split), so there is no look-ahead;
the in-sample mean is used to demean the entire series. The filter's Monte Carlo error is
negligible relative to the reported differences: per-seed SD ≈ 0.2–0.45 LL, and doubling to
20,000 particles leaves the mean unchanged — so |Δ| ≤ 0.75 LL is treated as a tie.

**Note 4 — MCMC uses the posterior mean as a point estimate**, plugged into the particle filter,
for an apples-to-apples comparison with the TCN's point estimate. This is *not* the full Bayesian
posterior-predictive (which would integrate the filter over posterior draws). That choice is
conservative for the reported conclusion: marginalising over parameter uncertainty would, if
anything, raise MCMC's predictive likelihood further, widening — not narrowing — its lead over
the TCN. The MCMC-vs-TCN gap is therefore a lower bound on the Bayesian benchmark's advantage.

**Note 5 — Boundary caveat on FX μ.** The estimated FX log-volatility means (μ ≈ −10.1 to −10.4)
sit at or below the −10 lower edge of both the network training range and the simulation-
consistent prior. MCMC (Gaussian prior on μ) can move below −10; the TCN, bounded by its training
range, cannot and shrinks μ upward (e.g. EUR/USD MCMC −10.43 vs TCN −10.22). This partly explains
the TCN's larger shortfall on FX and is consistent with F5.
