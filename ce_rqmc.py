"""CE-RQMC：单因子 copula 的正态辅助坐标重要抽样。"""
import time
import warnings
import numpy as np
from scipy.stats import norm, qmc

if __package__:
    from .pmc import PortfolioModel, mean_se, resolve_var_values
else:
    from pmc import PortfolioModel, mean_se, resolve_var_values


def log_likelihood_ratio(base, shift):
    """p(W)/q_mu(W)，p=N(0,I)，q_mu=N(mu,I)。"""
    return -np.asarray(base) @ shift + 0.5 * np.dot(shift, shift)


def proposal_batches(config, model, shift, samples, seed):
    qmc_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    engine = qmc.Halton(d=model.base_dimension, scramble=True, seed=np.random.default_rng(qmc_seed))
    rng = np.random.default_rng(noise_seed)
    for start in range(0, samples, config.batch_size):
        size = min(config.batch_size, samples - start)
        uniforms = engine.random(size).clip(config.probability_clip, 1 - config.probability_clip)
        base = norm.ppf(uniforms) + shift
        z, scale = model.factors_from_base(base)
        losses, individual = model.losses_given_factors(z, scale, rng)
        yield base, losses, individual


def ce_pilot_mu(config, model, target):
    """逐步提高精英阈值，以带原分布权重的均值更新实现 CE 投影。"""
    shift = np.zeros(model.base_dimension)
    history = []
    for iteration in range(config.ce_max_iterations):
        base = np.empty((config.ce_pilot_samples, model.base_dimension))
        losses = np.empty(config.ce_pilot_samples)
        start = 0
        for w, loss, _ in proposal_batches(config, model, shift, config.ce_pilot_samples,
                                           config.ce_seed + iteration):
            stop = start + len(loss)
            base[start:stop], losses[start:stop] = w, loss
            start = stop
        level = min(target, np.quantile(losses, 1 - config.ce_tail_fraction))
        # Avoid the mass at L=0 making every observation an elite observation.
        elite = (losses >= level) if level > 0 else (losses > 0)
        history.append(float(level))
        if not elite.any():
            warnings.warn("CE pilot saw no positive losses; using the current valid IS proposal", RuntimeWarning)
            break
        log_weights = log_likelihood_ratio(base[elite], shift)
        weights = np.exp(log_weights - log_weights.max())
        fitted = np.sum(weights[:, None] * base[elite], axis=0) / weights.sum()
        shift = np.clip((1 - config.ce_smoothing) * shift + config.ce_smoothing * fitted,
                        -config.ce_max_shift, config.ce_max_shift)
        if level >= target:
            break
    return shift, history


def ce_varc_once(config, model, a, shift, seed):
    numerator = np.zeros(config.n_obligors)
    denominator, weight_squares = 0.0, 0.0
    reference = -np.inf
    hits = 0
    for base, losses, individual in proposal_batches(config, model, shift, config.ce_samples, seed):
        mask = np.abs(losses - a) <= config.ce_bandwidth
        if not mask.any():
            continue
        logs = log_likelihood_ratio(base[mask], shift)
        # Rescale every block to the largest log weight observed so far.
        new_reference = max(reference, float(logs.max()))
        factor = np.exp(reference - new_reference)
        numerator *= factor
        denominator *= factor
        weight_squares *= factor * factor
        weights = np.exp(logs - new_reference)
        numerator += np.sum(weights[:, None] * individual[mask], axis=0)
        denominator += weights.sum()
        weight_squares += np.dot(weights, weights)
        reference = new_reference
        hits += int(mask.sum())
    if hits == 0:
        return np.full(config.n_obligors, np.nan), 0, 0.0
    return numerator / denominator, hits, denominator * denominator / weight_squares


def run(config, copula):
    config.validate("ce_rqmc", copula)
    model = PortfolioModel(config, copula)
    values, source, calibration_time = resolve_var_values(config, copula, model)
    results = []
    for alpha, a in zip(config.confidence_levels, values):
        start = time.perf_counter()
        target = max(0.0, a - config.ce_bandwidth)
        shift, history = ce_pilot_mu(config, model, target)
        pilot_time = time.perf_counter() - start
        estimates, hits, ess = [], [], []
        for r in range(config.repetitions):
            estimate, count, effective = ce_varc_once(
                config, model, a, shift, config.ce_seed + config.ce_replication_seed_offset + r)
            estimates.append(estimate)
            hits.append(count)
            ess.append(effective)
        if min(hits) == 0:
            warnings.warn("Some CE replications have no samples in the VaR band; returning NaN", RuntimeWarning)
        mean, se = mean_se(estimates)
        results.append({"alpha": alpha, "a": a, "mean_VaRC": mean, "se_VaRC": se,
                        "copula": copula, "mu_star": shift, "loss_threshold": target,
                        "pilot_thresholds": history, "pilot_time": pilot_time,
                        "samples": np.mean(hits), "band_ess": np.mean(ess),
                        "var_source": source, "var_calibration_time": calibration_time,
                        "total_time": time.perf_counter() - start, "time_scope": "this_level"})
    return results
