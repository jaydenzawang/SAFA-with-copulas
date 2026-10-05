"""SAFA；保留原估计公式，同质 LGD 载荷由 config.rho 指定。"""
import time
import numpy as np
from scipy.stats import norm, beta


def conditional_density(epsilon, z_l, config):
    epsilon = np.clip(epsilon, config.probability_clip, 1 - config.probability_clip)
    v = norm.ppf(beta.cdf(epsilon, config.lgd_alpha, config.lgd_beta))
    s = np.sqrt(1 - config.rho ** 2)
    w = (v - config.rho * z_l) / s
    phi_v = norm.pdf(v)
    if phi_v <= 0:
        return 0.0
    return beta.pdf(epsilon, config.lgd_alpha, config.lgd_beta) * norm.pdf(w) / (phi_v * s)


def run(config):
    config.validate("safa")
    rng = np.random.default_rng(config.seed)
    n, inner = config.n_obligors, config.safa_inner
    rho, u = config.rho, np.full(n, config.exposure)
    s = np.sqrt(1 - rho ** 2)
    lgd_a, lgd_b = config.lgd_alpha, config.lgd_beta
    covariance = [[1, config.tau], [config.tau, 1]]
    x_d = norm.ppf(1 - np.linspace(config.p_low, config.p_high, n))
    results = []
    for alpha, a in zip(config.confidence_levels, config.var_values):
        start = time.perf_counter()
        contributions = []
        for _ in range(config.repetitions):
            accumulated = np.zeros(n)
            for _ in range(config.safa_outer):
                z_l, z_d = rng.multivariate_normal([0, 0], covariance)
                eta_l = rng.standard_normal((n, inner))
                eta_d = rng.standard_normal((n, inner))
                epsilon_all = beta.ppf(norm.cdf(rho * z_l + s * eta_l), lgd_a, lgd_b)
                defaults = (rho * z_d + s * eta_d) > x_d[:, None]
                losses = u[:, None] * epsilon_all * defaults
                sorted_minus = np.sort(losses.sum(axis=0)[None, :] - losses, axis=1)

                # 同质载荷允许各 obligor 共享条件 LGD 样本。
                epsilon = np.clip(beta.ppf(norm.cdf(rho * z_l + s * rng.standard_normal(inner)),
                                           lgd_a, lgd_b), config.safa_epsilon_clip,
                                  1 - config.safa_epsilon_clip)
                f_beta = beta.pdf(epsilon, lgd_a, lgd_b)
                v = norm.ppf(beta.cdf(epsilon, lgd_a, lgd_b))
                w = (v - rho * z_l) / s
                phi_v, phi_w = norm.pdf(v), norm.pdf(w)
                mask = phi_v > 0
                f_cond, term1 = np.zeros(inner), np.zeros(inner)
                f_cond[mask] = f_beta[mask] * phi_w[mask] / (phi_v[mask] * s)
                term1[mask] = f_beta[mask] / phi_v[mask] * (v[mask] - w[mask] / s)
                term2 = (lgd_a - 1) / epsilon - (lgd_b - 1) / (1 - epsilon)
                f_prime = f_cond * (term1 + term2)
                g = np.zeros(inner)
                valid = f_cond > 0
                g[valid] = 1 + epsilon[valid] * f_prime[valid] / f_cond[valid]
                p_cond = 1 - norm.cdf((x_d - rho * z_d) / s)
                prod_all = np.prod(1 - p_cond)
                inner_values = np.zeros(n)
                for i in range(n):
                    boundary = 0.0
                    if 0 < a / u[i] < 1:
                        denominator = 1 - p_cond[i]
                        prod_i = prod_all / denominator if denominator > 0 else 0.0
                        boundary = -a / u[i] * conditional_density(a / u[i], z_l, config) * prod_i
                    thresholds = a - u[i] * epsilon
                    cdf = np.searchsorted(sorted_minus[i], thresholds, side="right") / inner
                    inner_values[i] = np.sum(cdf * g) / inner + boundary
                accumulated += p_cond * inner_values
            accumulated /= config.safa_outer
            density = accumulated.sum() / a
            contributions.append(accumulated / density if density != 0 else np.zeros(n))
        results.append({"alpha": alpha, "a": a,
                        "mean_VaRC": np.mean(contributions, axis=0),
                        "se_VaRC": np.std(contributions, axis=0) / np.sqrt(config.repetitions),
                        "total_time": time.perf_counter() - start})
    return results
