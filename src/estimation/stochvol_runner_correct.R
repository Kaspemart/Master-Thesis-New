#!/usr/bin/env Rscript
# Correct-model stochvol runner for the misspecification analysis (cell d).
#
# Estimates the CORRECT model for each misspecified DGP:
#   asv  — leverage:        params [mu, phi, sigma, rho]
#   svt  — Student-t:       params [mu, phi, sigma, nu]
#   asvt — leverage + t:    params [mu, phi, sigma, rho, nu]
#
# Priors: base as in stochvol_runner.R; plus
#   rho ~ Beta(4,4) on (rho+1)/2  — symmetric, weakly informative, centred at 0
#                                    (avoids the mis-centred-prior problem seen with phi;
#                                     checked post-hoc for prior-induced bias)
#   nu  ~ 2 + Exponential(0.1)     — stochvol's standard weakly-informative t prior
#
# Usage:
#   Rscript stochvol_runner_correct.R <in_csv> <out_json> <model> <draws> <burnin> <seed1> <seed2>

suppressPackageStartupMessages({
  library(stochvol)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7L) stop("Expected 7 args: in_csv out_json model draws burnin seed1 seed2")
input_csv <- args[1]; output_json <- args[2]; model <- args[3]
draws_n <- as.integer(args[4]); burnin_n <- as.integer(args[5])
seed1 <- as.integer(args[6]); seed2 <- as.integer(args[7])

y <- read.csv(input_csv, header = FALSE)[[1]]
if (any(!is.finite(y))) stop("Non-finite values in input returns.")

mu_p  <- sv_normal(mean = -5.0, sd = 3.0)
phi_p <- sv_beta(shape1 = 7, shape2 = 1)
sig_p <- sv_gamma(shape = 0.5, rate = 0.5)
rho_p <- sv_constant(0)      # default: no leverage
nu_p  <- sv_infinity()       # default: Gaussian

if (model == "asv") {
  rho_p <- sv_beta(shape1 = 4, shape2 = 4)
  cols <- c("mu", "phi", "sigma", "rho")
} else if (model == "svt") {
  nu_p <- sv_exponential(rate = 0.1)
  cols <- c("mu", "phi", "sigma", "nu")
} else if (model == "asvt") {
  rho_p <- sv_beta(shape1 = 4, shape2 = 4)
  nu_p  <- sv_exponential(rate = 0.1)
  cols <- c("mu", "phi", "sigma", "rho", "nu")
} else {
  stop(paste("unknown model:", model))
}

priors <- specify_priors(mu = mu_p, phi = phi_p, sigma2 = sig_p, rho = rho_p, nu = nu_p)

compute_rhat <- function(c1, c2) {
  n <- length(c1)
  cm <- c(mean(c1), mean(c2))
  B <- n * var(cm)
  W <- mean(c(var(c1), var(c2)))
  sqrt(((n - 1L) / n * W + B / n) / W)
}

set.seed(seed1)
res1 <- svsample(y, draws = draws_n, burnin = burnin_n, priorspec = priors, quiet = TRUE)
set.seed(seed2)
res2 <- svsample(y, draws = draws_n, burnin = burnin_n, priorspec = priors, quiet = TRUE)

p1 <- para(res1); p2 <- para(res2)
d1 <- matrix(p1[, cols], ncol = length(cols), dimnames = list(NULL, cols))
d2 <- matrix(p2[, cols], ncol = length(cols), dimnames = list(NULL, cols))

rhat_vals <- sapply(cols, function(cc) compute_rhat(d1[, cc], d2[, cc]))
post_mean <- colMeans(d1)
post_sd   <- apply(d1, 2L, sd)

out <- list(
  cols      = cols,
  post_mean = unname(as.numeric(post_mean)),
  post_sd   = unname(as.numeric(post_sd)),
  rhat      = unname(as.numeric(rhat_vals))
)
write(toJSON(out, digits = 10L, auto_unbox = TRUE), output_json)
