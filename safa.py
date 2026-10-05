"""SAFA：将经验 CDF 的分部积分精确化简后，对自身 LGD 解析积分。

A_i(a) = E[p_i(Z,S) e_i* f(e_i*|Z,S) 1{0<e_i*<1}],
e_i*=(a-L_{-i})/u_i。VaRC_i = a A_i / sum_j A_j。
这包含 L_{-i}=0 的概率质量，不再减去该质量对应的贡献。
"""
import time
import warnings
import numpy as np

if __package__:
    from .pmc import PortfolioModel, mean_se, resolve_var_values
else:
    from pmc import PortfolioModel, mean_se, resolve_var_values


def conditional_numerators(model, a, losses, z, scale):
    """每条条件路径的 A_i 样本；支持非单位 EAD，并处理零损失原子。"""
    remaining = losses.sum(axis=1, keepdims=True) - losses
    # All losses are nonnegative; eliminate tiny subtraction roundoff.
    remaining = np.maximum(remaining, 0.0)
    epsilon = (a - remaining) / model.config.exposure
    valid = (epsilon > 0) & (epsilon < 1)
    log_density = model.conditional_lgd_logpdf(epsilon, z, scale)
    value = np.zeros_like(epsilon)
    value[valid] = np.exp(np.log(epsilon[valid]) + log_density[valid])
    return value * model.conditional_default_probabilities(z, scale)


def run(config, copula):
    config.validate("safa", copula)
    model = PortfolioModel(config, copula)
    values, source, calibration_time = resolve_var_values(config, copula, model)
    rng = np.random.default_rng(config.seed)
    results = []
    for alpha, a in zip(config.confidence_levels, values):
        start = time.perf_counter()
        contributions, densities = [], []
        for _ in range(config.repetitions):
            accumulated = np.zeros(config.n_obligors)
            zs, scales = model.sample_factors(rng, config.safa_outer)
            for z, scale in zip(zs, scales):
                inner_sum = np.zeros(config.n_obligors)
                for offset in range(0, config.safa_inner, config.batch_size):
                    size = min(config.batch_size, config.safa_inner - offset)
                    _, losses = model.losses_given_factors(np.full(size, z), np.full(size, scale), rng)
                    inner_sum += conditional_numerators(model, a, losses, z, scale).sum(axis=0)
                accumulated += inner_sum / config.safa_inner
            accumulated /= config.safa_outer
            density = accumulated.sum() / a
            densities.append(density)
            if not np.isfinite(density) or density <= 0:
                warnings.warn("SAFA estimated no positive finite density at VaR; returning NaN", RuntimeWarning)
                contributions.append(np.full(config.n_obligors, np.nan))
            else:
                contributions.append(accumulated / density)
        mean, se = mean_se(contributions)
        results.append({"alpha": alpha, "a": a, "mean_VaRC": mean, "se_VaRC": se,
                        "copula": copula, "density": np.mean(densities),
                        "var_source": source, "var_calibration_time": calibration_time,
                        "total_time": time.perf_counter() - start, "time_scope": "this_level"})
    return results
