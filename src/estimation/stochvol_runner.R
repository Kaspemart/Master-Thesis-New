#!/usr/bin/env Rscript
# stochvol_runner.R — Single-series stochvol MCMC for the base SV model.
#
# Implements the ASIS (ancillarity-sufficiency interweaving) sampler from
# Kastner & Frühwirth-Schnatter (2014), via the stochvol R package.
#
# Called by stochvol_runner.py via subprocess. Not intended to be run directly.
#
# Usage:
#   Rscript stochvol_runner.R <input_csv> <output_json> <draws> <burnin> <seed1> <seed2>
#
# Arguments:
#   input_csv   — path to CSV file with one column of log-returns (no header)
#   output_json — path to write JSON output
#   draws       — number of posterior draws per chain
#   burnin      — number of burn-in iterations per chain
#   seed1       — RNG seed for chain 1
#   seed2       — RNG seed for chain 2 (used only for R-hat computation)
#
# Output JSON fields:
#   post_mean  — posterior means [mu, phi, sigma_eta]
#   post_sd    — posterior standard deviations [mu, phi, sigma_eta]
#   rhat       — Gelman-Rubin R-hat [mu, phi, sigma_eta], computed across 2 chains
#   samples    — posterior draws from chain 1, shape (draws, 3): [[mu,phi,sigma_eta],...]
#
# Priors (weakly informative, consistent with simulation ranges):
#   mu        ~ Normal(-5, 3)              covers mu in (-10, 0)       [P ~ 0.90]
#   phi       ~ Beta(5, 1.5) on (phi+1)/2  concentrates mass near 1    [P(phi>0.5) ~ 0.61]
#   sigma_eta ~ sqrt(Gamma(0.5, 0.5))      covers sigma_eta in (0.05,1) [P ~ 0.64]
# These cannot be made exactly Uniform (stochvol does not support Uniform priors), but
# are wide enough that the prior has minimal influence on the posterior for T >= 500.

suppressPackageStartupMessages({
  library(stochvol)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) {
  stop("Expected 6 arguments: input_csv output_json draws burnin seed1 seed2")
}

input_csv   <- args[1]
output_json <- args[2]
draws_n     <- as.integer(args[3])
burnin_n    <- as.integer(args[4])
seed1       <- as.integer(args[5])
seed2       <- as.integer(args[6])

# ── Load returns ──────────────────────────────────────────────────────────────
y <- read.csv(input_csv, header = FALSE)[[1]]
if (any(!is.finite(y))) stop("Non-finite values found in input returns.")

# ── Prior specification ───────────────────────────────────────────────────────
priors <- specify_priors(
  mu     = sv_normal(mean = -5.0, sd = 3.0),
  phi    = sv_beta(shape1 = 5, shape2 = 1.5),
  sigma2 = sv_gamma(shape = 0.5, rate = 0.5)
)

# ── Gelman-Rubin R-hat for two chains ────────────────────────────────────────
compute_rhat <- function(chain1, chain2) {
  n           <- length(chain1)
  chain_means <- c(mean(chain1), mean(chain2))
  chain_vars  <- c(var(chain1),  var(chain2))
  grand_mean  <- mean(chain_means)
  B           <- n * var(chain_means)           # between-chain variance * n
  W           <- mean(chain_vars)               # within-chain variance
  var_hat     <- (n - 1L) / n * W + B / n
  sqrt(var_hat / W)
}

# ── Run two chains ────────────────────────────────────────────────────────────
set.seed(seed1)
res1 <- tryCatch(
  svsample(y, draws = draws_n, burnin = burnin_n, priorspec = priors, quiet = TRUE),
  error = function(e) stop(paste("stochvol chain 1 failed:", conditionMessage(e)))
)

set.seed(seed2)
res2 <- tryCatch(
  svsample(y, draws = draws_n, burnin = burnin_n, priorspec = priors, quiet = TRUE),
  error = function(e) stop(paste("stochvol chain 2 failed:", conditionMessage(e)))
)

# ── Extract draws: columns mu, phi, sigma (sigma = sigma_eta, not sigma^2) ───
# para() returns an mcmc object — convert to plain numeric matrix for JSON serialisation
d1 <- matrix(para(res1)[, c("mu", "phi", "sigma")], ncol = 3L,
             dimnames = list(NULL, c("mu", "phi", "sigma")))
d2 <- matrix(para(res2)[, c("mu", "phi", "sigma")], ncol = 3L,
             dimnames = list(NULL, c("mu", "phi", "sigma")))

# ── R-hat per parameter ───────────────────────────────────────────────────────
rhat_vals <- c(
  compute_rhat(d1[, "mu"],    d2[, "mu"]),
  compute_rhat(d1[, "phi"],   d2[, "phi"]),
  compute_rhat(d1[, "sigma"], d2[, "sigma"])
)

# ── Posterior summaries (chain 1) ─────────────────────────────────────────────
post_mean <- colMeans(d1)
post_sd   <- apply(d1, 2L, sd)

# ── Write JSON ────────────────────────────────────────────────────────────────
# samples: (draws, 3) matrix — toJSON serialises row-major: [[mu,phi,sig],...]
out <- list(
  post_mean = unname(as.numeric(post_mean)),
  post_sd   = unname(as.numeric(post_sd)),
  rhat      = unname(as.numeric(rhat_vals)),
  samples   = unname(d1)          # matrix (draws x 3), serialised as array of rows
)

write(toJSON(out, digits = 10L, matrix = "rowmajor"), output_json)
