"""Cross-entropy randomized quasi-Monte Carlo；仅 run() 启动计算。"""
import time
import numpy as np
from scipy.stats import norm, beta, qmc


def halton_normals(n, seed, clip):
    uniforms = qmc.Halton(d=2, scramble=True, seed=seed).random(n)
    return norm.ppf(uniforms.clip(clip, 1 - clip))


def simulate_losses(z, config, x_d, rng):
    shape = (len(z), config.n_obligors)
    eta_d = rng.standard_normal(shape)
    eta_l = rng.standard_normal(shape)
    residual = np.sqrt(1 - config.rho ** 2)
    x = config.rho * z[:, 1, None] + residual * eta_d
    y = config.rho * z[:, 0, None] + residual * eta_l
    uniforms = norm.cdf(y).clip(config.probability_clip, 1 - config.probability_clip)
    epsilon = beta.ppf(uniforms, config.lgd_alpha, config.lgd_beta)
    individual = config.exposure * epsilon * (x > x_d)
    return individual.sum(axis=1), individual


def ce_pilot_mu(config, threshold, chol, x_d, sigma_inv):
    rng = np.random.default_rng(config.ce_seed)
    mu = np.zeros(2)
    for iteration in range(config.ce_max_iterations):
        normals = halton_normals(config.ce_pilot_samples, config.ce_seed + iteration,
                                 config.probability_clip)
        z = mu + normals @ chol.T
        losses, _ = simulate_losses(z, config, x_d, rng)
        if np.quantile(losses, 1 - config.ce_tail_fraction) > threshold:
            break
        indicators = losses > threshold
        if not indicators.any():
            mu += config.ce_shift_step
            continue
        log_weights = -(z @ (mu @ sigma_inv)) + 0.5 * (mu @ sigma_inv @ mu)
        weights = indicators * np.exp(log_weights)
        if weights.sum() <= 0:
            mu += config.ce_shift_step
            continue
        mu = (weights[:, None] * z).sum(axis=0) / weights.sum()
    return mu


def ce_varc_once(config, a, mu, seed, chol, x_d, sigma_inv):
    rng = np.random.default_rng(seed)
    z = mu + halton_normals(config.ce_samples, seed, config.probability_clip) @ chol.T
    losses, individual = simulate_losses(z, config, x_d, rng)
    mask = np.abs(losses - a) <= config.ce_bandwidth
    if not mask.any():
        return np.full(config.n_obligors, np.nan)
    weights = np.exp(-(z @ (mu @ sigma_inv)) + 0.5 * (mu @ sigma_inv @ mu))[mask]
    return (weights[:, None] * individual[mask]).sum(axis=0) / weights.sum()


def run(config):
    config.validate("ce_rqmc")
    if __package__:
        from .pmc import estimate_loss_threshold
    else:
        from pmc import estimate_loss_threshold
    start = time.perf_counter()
    threshold = (estimate_loss_threshold(config) if config.loss_threshold is None
                 else config.loss_threshold)
    covariance = np.array([[1, config.tau], [config.tau, 1]])
    chol = np.linalg.cholesky(covariance)
    sigma_inv = np.linalg.inv(covariance)
    x_d = norm.ppf(1 - np.linspace(config.p_low, config.p_high, config.n_obligors))
    mu = ce_pilot_mu(config, threshold, chol, x_d, sigma_inv)
    pilot_time = time.perf_counter() - start
    results = []
    for alpha, a in zip(config.confidence_levels, config.var_values):
        start = time.perf_counter()
        values = np.asarray([
            ce_varc_once(config, a, mu, config.ce_seed + config.ce_replication_seed_offset + r,
                         chol, x_d, sigma_inv)
            for r in range(config.repetitions)
        ])
        se = (values.std(axis=0, ddof=1) / np.sqrt(config.repetitions)
              if config.repetitions > 1 else np.zeros(config.n_obligors))
        results.append({"alpha": alpha, "a": a, "mean_VaRC": values.mean(axis=0),
                        "se_VaRC": se, "mu_star": mu.copy(), "loss_threshold": threshold,
                        "pilot_time": pilot_time, "total_time": time.perf_counter() - start})
    return results
