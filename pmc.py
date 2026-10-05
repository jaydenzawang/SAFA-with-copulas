"""Plain Monte Carlo；导入本模块不会启动模拟。"""
import time
import numpy as np
from scipy.stats import norm, beta


def generate_samples(config, rng, n_samples):
    z = rng.multivariate_normal([0, 0], [[1, config.tau], [config.tau, 1]], n_samples)
    eta_l = rng.standard_normal((n_samples, config.n_obligors))
    eta_d = rng.standard_normal((n_samples, config.n_obligors))
    residual = np.sqrt(1 - config.rho ** 2)
    y = config.rho * z[:, 0, None] + residual * eta_l
    x = config.rho * z[:, 1, None] + residual * eta_d
    epsilon = beta.ppf(norm.cdf(y), config.lgd_alpha, config.lgd_beta)
    pd = np.linspace(config.p_low, config.p_high, config.n_obligors)
    losses = config.exposure * epsilon * (x > norm.ppf(1 - pd))
    return losses.sum(axis=1), losses


def estimate_loss_threshold(config):
    """原 Loss_Threshold.py 的 3 × E[L]，分批累计以减少内存使用。"""
    config.validate("pmc")
    rng = np.random.default_rng(config.seed)
    total = 0.0
    count = 0
    for _ in range(config.threshold_repetitions):
        for start in range(0, config.threshold_samples, config.threshold_batch_size):
            size = min(config.threshold_batch_size, config.threshold_samples - start)
            losses, _ = generate_samples(config, rng, size)
            total += losses.sum()
            count += size
    return config.threshold_multiplier * total / count


def run(config):
    config.validate("pmc")
    rng = np.random.default_rng(config.seed)
    results = []
    for alpha in config.confidence_levels:
        start = time.perf_counter()
        vars_, contributions, counts = [], [], []
        for _ in range(config.repetitions):
            losses, individual = generate_samples(config, rng, config.pmc_samples)
            var = np.quantile(losses, alpha)
            mask = np.abs(losses - var) <= config.pmc_bandwidth
            vars_.append(var)
            counts.append(mask.sum())
            contributions.append(individual[mask].mean(axis=0) if mask.any()
                                 else np.full(config.n_obligors, np.nan))
        results.append({
            "alpha": alpha, "a": np.mean(vars_),
            "se_VaR": np.std(vars_) / np.sqrt(config.repetitions),
            "mean_VaRC": np.mean(contributions, axis=0),
            "se_VaRC": np.std(contributions, axis=0) / np.sqrt(config.repetitions),
            "samples": np.mean(counts), "total_time": time.perf_counter() - start,
        })
    return results
