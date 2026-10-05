"""PMC 及三个算法共享的单因子 copula / 组合模拟。

为保留五个运行模块，共享分布层放在本文件；导入不会进行模拟。
数学推导、skew-t 的定义和数值近似见 DERIVATION.md。
"""
import time
import warnings
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import beta, chi2, norm, skewnorm, t


class LatentMarginal:
    """rho Z + sqrt(1-rho²) eta 除以共享尺度后的边际分布。"""

    def __init__(self, config, copula, loading):
        self.config, self.copula = config, copula
        self.df = config.degrees_of_freedom
        delta = config.skew_shape / np.hypot(1.0, config.skew_shape)
        d = loading * delta if copula == "skew_t" else 0.0
        self.shape = d / np.sqrt(1 - d * d)
        self._cdf_table = None
        if copula == "skew_t" and self.shape != 0:
            nodes, weights = leggauss(config.skew_quadrature_order)
            angle = np.arctan(self.shape)
            self._angles = (nodes + 1) * angle / 2
            self._weights = weights * angle / (2 * np.pi)
            # Logit-spaced Student-t quantiles resolve both centre and long tails.
            bound = -logit(config.skew_table_tail)
            grid = t.ppf(expit(np.linspace(-bound, bound, config.skew_grid_size)), self.df)
            probabilities = self.cdf_exact(grid)
            if not np.isfinite(grid).all() or not np.isfinite(probabilities).all():
                raise ValueError("Skew-t table overflow; adjust df or table tail")
            if np.any(np.diff(probabilities) < -config.quadrature_tolerance):
                raise ValueError("Skew-t CDF quadrature is nonmonotone; increase quadrature order")
            probabilities = np.maximum.accumulate(probabilities)
            self._cdf_table = PchipInterpolator(grid, probabilities, extrapolate=False)
            # Invert the same monotone table, dropping double-precision tail ties.
            unique = np.r_[True, np.diff(probabilities) > 0]
            self._ppf_table = PchipInterpolator(probabilities[unique], grid[unique], extrapolate=False)
            self._x_bounds = (grid[0], grid[-1])
            self._p_bounds = (probabilities[0], probabilities[-1])

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        if self.copula == "gaussian":
            return norm.logpdf(x)
        result = t.logpdf(x, self.df)
        if self.shape != 0:
            h = self.shape * x * np.sqrt((self.df + 1) / (self.df + x * x))
            result = np.log(2.0) + result + t.logcdf(h, self.df + 1)
        return result

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def score(self, x):
        """d log h(x)/dx；用于独立检查条件 LGD 密度导数。"""
        x = np.asarray(x, dtype=float)
        if self.copula == "gaussian":
            return -x
        result = -(self.df + 1) * x / (self.df + x * x)
        if self.shape != 0:
            h = self.shape * x * np.sqrt((self.df + 1) / (self.df + x * x))
            derivative = self.shape * np.sqrt(self.df + 1) * self.df / (self.df + x * x) ** 1.5
            result += np.exp(t.logpdf(h, self.df + 1) - t.logcdf(h, self.df + 1)) * derivative
        return result

    def cdf_exact(self, x):
        """确定性积分（无插值）；不是 Jones-Faddy skew-t。"""
        x = np.asarray(x, dtype=float)
        if self.copula == "gaussian":
            return norm.cdf(x)
        result = t.cdf(x, self.df)
        if self.shape != 0:
            correction = np.zeros_like(x)
            for angle, weight in zip(self._angles, self._weights):
                correction += weight * np.exp(-self.df / 2 * np.log1p((x / np.cos(angle)) ** 2 / self.df))
            result = result - correction
        return np.clip(result, 0, 1)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        if self._cdf_table is None:
            return self.cdf_exact(x)
        flat = x.reshape(-1)
        out = np.asarray(self._cdf_table(flat))
        outside = (flat < self._x_bounds[0]) | (flat > self._x_bounds[1])
        if outside.any():
            out[outside] = self.cdf_exact(flat[outside])
        return np.clip(out.reshape(x.shape), 0, 1)

    def ppf(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=float)
        if np.any((probabilities <= 0) | (probabilities >= 1)):
            raise ValueError("Latent quantiles require probabilities strictly inside (0, 1)")
        if self.copula == "gaussian":
            return norm.ppf(probabilities)
        if self._cdf_table is None:
            return t.ppf(probabilities, self.df)
        flat = probabilities.reshape(-1)
        out = np.asarray(self._ppf_table(flat))
        outside = (flat < self._p_bounds[0]) | (flat > self._p_bounds[1])
        for index in np.flatnonzero(outside):
            # Adaptive reference inversion instead of extrapolating the table.
            target = flat[index]
            lo, hi = self._x_bounds
            while self.cdf_exact(lo) > target:
                lo *= 2
            while self.cdf_exact(hi) < target:
                hi *= 2
            out[index] = brentq(lambda x: self.cdf_exact(x) - target, lo, hi,
                                xtol=self.config.root_tolerance)
        return out.reshape(probabilities.shape)


class PortfolioModel:
    def __init__(self, config, copula):
        config.validate("pmc", copula)
        self.config, self.copula = config, copula
        self.default_marginal = LatentMarginal(config, copula, config.rho_d)
        self.lgd_marginal = (self.default_marginal if config.rho_l == config.rho_d
                             else LatentMarginal(config, copula, config.rho_l))
        self.pd = np.linspace(config.p_low, config.p_high, config.n_obligors)
        self.default_thresholds = self.default_marginal.ppf(1 - self.pd)
        self.s_d = np.sqrt(1 - config.rho_d ** 2)
        self.s_l = np.sqrt(1 - config.rho_l ** 2)
        # These are integration/IS coordinates, not a vector systematic Z.
        self.base_dimension = 1 if copula == "gaussian" else 2

    def sample_factors(self, rng, size):
        c = self.config
        if self.copula == "skew_t" and c.skew_shape != 0:
            z = skewnorm.rvs(c.skew_shape, size=size, random_state=rng)
        else:
            z = rng.standard_normal(size)
        scale = (np.ones(size) if self.copula == "gaussian"
                 else np.sqrt(rng.chisquare(c.degrees_of_freedom, size) / c.degrees_of_freedom))
        return z, scale

    def factors_from_base(self, base):
        """W~N(0,I) -> 一个 Z，以及 t 类模型的共享 S=sqrt(Lambda/nu)。"""
        c = self.config
        z = base[:, 0].copy()
        if self.copula == "skew_t" and c.skew_shape != 0:
            positive = base[:, 0] > 0
            z[~positive] = skewnorm.ppf(norm.cdf(base[~positive, 0]), c.skew_shape)
            z[positive] = skewnorm.isf(norm.sf(base[positive, 0]), c.skew_shape)
        if self.copula == "gaussian":
            return z, np.ones(len(base))
        positive = base[:, 1] > 0
        mixing = np.empty(len(base))
        mixing[~positive] = chi2.ppf(norm.cdf(base[~positive, 1]), c.degrees_of_freedom)
        mixing[positive] = chi2.isf(norm.sf(base[positive, 1]), c.degrees_of_freedom)
        if not np.isfinite(z).all() or np.any(mixing <= 0) or not np.isfinite(mixing).all():
            raise FloatingPointError("Factor transform overflow; reduce CE shift bounds")
        return z, np.sqrt(mixing / c.degrees_of_freedom)

    def lgd_from_latent(self, latent):
        c = self.config
        uniforms = self.lgd_marginal.cdf(latent).clip(c.probability_clip, 1 - c.probability_clip)
        return beta.ppf(uniforms, c.lgd_alpha, c.lgd_beta)

    def conditional_default_probabilities(self, z, scale):
        return norm.sf((np.asarray(scale)[..., None] * self.default_thresholds
                        - self.config.rho_d * np.asarray(z)[..., None]) / self.s_d)

    def losses_given_factors(self, z, scale, rng):
        c = self.config
        z, scale = np.broadcast_arrays(np.atleast_1d(z), np.atleast_1d(scale))
        shape = (len(z), c.n_obligors)
        x = (c.rho_d * z[:, None] + self.s_d * rng.standard_normal(shape)) / scale[:, None]
        y = (c.rho_l * z[:, None] + self.s_l * rng.standard_normal(shape)) / scale[:, None]
        individual = c.exposure * self.lgd_from_latent(y) * (x > self.default_thresholds)
        return individual.sum(axis=1), individual

    def sample_losses(self, rng, size):
        return self.losses_given_factors(*self.sample_factors(rng, size), rng)

    def conditional_lgd_logpdf(self, epsilon, z, scale):
        """log f(e | Z=z,S=s)，在 (0,1) 外为 -inf。"""
        c = self.config
        epsilon, z, scale = np.broadcast_arrays(np.asarray(epsilon, dtype=float), z, scale)
        out = np.full(epsilon.shape, -np.inf)
        valid = (epsilon > 0) & (epsilon < 1)
        if valid.any():
            e = epsilon[valid]
            # Clip only for finite numerical inverse-CDF evaluation.
            probs = beta.cdf(e, c.lgd_alpha, c.lgd_beta).clip(c.probability_clip, 1 - c.probability_clip)
            v = self.lgd_marginal.ppf(probs)
            w = (scale[valid] * v - c.rho_l * z[valid]) / self.s_l
            out[valid] = (beta.logpdf(e, c.lgd_alpha, c.lgd_beta) + np.log(scale[valid] / self.s_l)
                          + norm.logpdf(w) - self.lgd_marginal.logpdf(v))
        return out

    def conditional_lgd_pdf(self, epsilon, z, scale):
        return np.exp(self.conditional_lgd_logpdf(epsilon, z, scale))

    def conditional_lgd_score(self, epsilon, z, scale):
        """d log f(e|z,s)/de，详见推导；主 SAFA 用解析积分免去该导数。"""
        c = self.config
        epsilon = np.asarray(epsilon, dtype=float)
        if np.any((epsilon <= 0) | (epsilon >= 1)):
            raise ValueError("Conditional density score is defined only for 0 < epsilon < 1")
        probs = beta.cdf(epsilon, c.lgd_alpha, c.lgd_beta).clip(c.probability_clip, 1 - c.probability_clip)
        v = self.lgd_marginal.ppf(probs)
        w = (scale * v - c.rho_l * z) / self.s_l
        dv_de = np.exp(beta.logpdf(epsilon, c.lgd_alpha, c.lgd_beta) - self.lgd_marginal.logpdf(v))
        return ((c.lgd_alpha - 1) / epsilon - (c.lgd_beta - 1) / (1 - epsilon)
                + dv_de * (-scale * w / self.s_l - self.lgd_marginal.score(v)))


def mean_se(values):
    """所有算法统一为重复间样本标准误；R=1 无法估计 SE。"""
    values = np.asarray(values)
    se = (values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1
          else np.full(values.shape[1:], np.nan))
    return values.mean(axis=0), se


def collect_total_losses(model, seed, samples):
    rng = np.random.default_rng(seed)
    losses = np.empty(samples)
    for start in range(0, samples, model.config.batch_size):
        stop = min(start + model.config.batch_size, samples)
        losses[start:stop] = model.sample_losses(rng, stop - start)[0]
    return losses


def resolve_var_values(config, copula, model):
    """外部 VaR 按 copula 区分；否则用独立 PMC pilot 校准当前模型。"""
    start = time.perf_counter()
    if copula in config.var_values:
        values, source = np.asarray(config.var_values[copula]), "configured"
    else:
        losses = collect_total_losses(model, config.var_pilot_seed, config.var_pilot_samples)
        values = np.quantile(losses, config.confidence_levels)
        source = "independent_pmc_pilot"
    if np.any(values <= 0):
        raise ValueError("VaR is zero (loss has an atom at zero); continuous-density VaRC requires positive VaR")
    return values, source, time.perf_counter() - start


def estimate_loss_threshold(config, copula):
    model = PortfolioModel(config, copula)
    rng = np.random.default_rng(config.threshold_seed)
    total = 0.0
    for start in range(0, config.threshold_samples, config.batch_size):
        size = min(config.batch_size, config.threshold_samples - start)
        total += model.sample_losses(rng, size)[0].sum()
    return config.threshold_multiplier * total / config.threshold_samples


def run(config, copula):
    config.validate("pmc", copula)
    model = PortfolioModel(config, copula)
    started = time.perf_counter()
    levels, n = len(config.confidence_levels), config.n_obligors
    vars_ = np.empty((config.repetitions, levels))
    contributions = np.full((config.repetitions, levels, n), np.nan)
    counts = np.zeros((config.repetitions, levels), dtype=int)
    for r in range(config.repetitions):
        # Replay the same RNG/batches after estimating VaR: exact same paths,
        # with only O(samples + batch_size * n) memory rather than samples * n.
        seed = config.seed + r
        losses = collect_total_losses(model, seed, config.pmc_samples)
        vars_[r] = np.quantile(losses, config.confidence_levels)
        if np.any(vars_[r] <= 0):
            raise ValueError("PMC estimated a zero VaR; choose a higher confidence level")
        del losses
        rng = np.random.default_rng(seed)
        sums = np.zeros((levels, n))
        for start in range(0, config.pmc_samples, config.batch_size):
            size = min(config.batch_size, config.pmc_samples - start)
            total, individual = model.sample_losses(rng, size)
            for j, a in enumerate(vars_[r]):
                mask = np.abs(total - a) <= config.pmc_bandwidth
                counts[r, j] += mask.sum()
                sums[j] += individual[mask].sum(axis=0)
        nonempty = counts[r] > 0
        contributions[r, nonempty] = sums[nonempty] / counts[r, nonempty, None]
    if np.any(counts == 0):
        warnings.warn("Some PMC replications have no samples in the VaR band; returning NaN", RuntimeWarning)
    var_mean, var_se = mean_se(vars_)
    mean, se = mean_se(contributions)
    elapsed = time.perf_counter() - started
    return [{"alpha": alpha, "a": var_mean[j], "se_VaR": var_se[j],
             "mean_VaRC": mean[j], "se_VaRC": se[j], "samples": counts[:, j].mean(),
             "copula": copula, "var_source": "pmc_each_replication",
             "total_time": elapsed, "time_scope": "all_levels"}
            for j, alpha in enumerate(config.confidence_levels)]
