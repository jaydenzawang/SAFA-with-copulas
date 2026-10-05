"""所有模型、模拟和数值参数；算法及 copula 在 main.py 中选择。"""
from dataclasses import dataclass, field
from math import isfinite, sqrt


@dataclass(frozen=True)
class Config:
    # 所有债务人共用一个标量 Z；PD/LGD 的特质正态噪声相互独立。
    n_obligors: int = 10
    p_low: float = 0.01
    p_high: float = 0.10
    exposure: float = 1.0
    rho_d: float = sqrt(0.5)
    rho_l: float = sqrt(0.5)
    lgd_alpha: float = 2.0
    lgd_beta: float = 5.0
    degrees_of_freedom: float = 5.0
    skew_shape: float = 3.0  # Z ~ SN(location=0, scale=1, shape=gamma)，未中心化。

    confidence_levels: tuple = (0.95, 0.96, 0.97, 0.98, 0.99)
    # 可选外部 VaR，例如 {"t": (a95, a96, a97, a98, a99)}。
    # 缺省用当前 copula 和组合独立预估；不沿用旧双因子模型的 VaR。
    var_values: dict = field(default_factory=dict)
    var_pilot_samples: int = 1_000_000
    var_pilot_seed: int = 2026
    repetitions: int = 10
    seed: int = 42
    print_precision: int = 4
    batch_size: int = 100_000

    # PMC：批量生成后，仅保留总损失及落入窗口的贡献累计。
    pmc_samples: int = 10_000_000
    pmc_bandwidth: float = 0.01

    # CE-RQMC：在独立标准正态辅助坐标上进行均值平移。
    ce_seed: int = 2025
    ce_pilot_samples: int = 2_000_000
    ce_samples: int = 10_000_000
    ce_max_iterations: int = 5
    ce_tail_fraction: float = 0.1
    ce_bandwidth: float = 0.005
    ce_smoothing: float = 0.7
    ce_max_shift: float = 6.0
    ce_replication_seed_offset: int = 1000

    # SAFA：给定 (Z, S) 后，对自身 LGD 进行解析积分。
    safa_outer: int = 1000
    safa_inner: int = 1000

    # 可单独调用 pmc.estimate_loss_threshold；CE 已直接针对 VaR 窗口训练。
    threshold_multiplier: float = 3.0
    threshold_samples: int = 1_000_000
    threshold_seed: int = 2027

    # skew-t CDF 的确定性 Gauss-Legendre 积分和单调插值。
    skew_grid_size: int = 16385
    skew_quadrature_order: int = 64
    skew_table_tail: float = 1e-13
    probability_clip: float = 1e-12
    quadrature_tolerance: float = 1e-10
    root_tolerance: float = 1e-11

    def validate(self, algorithm, copula):
        if algorithm not in {"pmc", "ce_rqmc", "safa"}:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        if copula not in {"gaussian", "t", "skew_t"}:
            raise ValueError(f"Unknown copula: {copula}")
        for name in ("n_obligors", "repetitions", "batch_size", "pmc_samples",
                     "ce_pilot_samples", "ce_samples", "ce_max_iterations",
                     "var_pilot_samples", "safa_outer", "safa_inner",
                     "threshold_samples", "skew_grid_size", "skew_quadrature_order"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.skew_grid_size < 33 or self.skew_quadrature_order < 8:
            raise ValueError("skew_grid_size >= 33 and skew_quadrature_order >= 8 required")
        for name in ("seed", "ce_seed", "var_pilot_seed", "threshold_seed",
                     "ce_replication_seed_offset", "print_precision"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 < self.p_low <= self.p_high < 1:
            raise ValueError("Require 0 < p_low <= p_high < 1")
        if not all(abs(r) < 1 for r in (self.rho_d, self.rho_l)):
            raise ValueError("Scalar loadings rho_d and rho_l must lie in (-1, 1)")
        for name in ("exposure", "lgd_alpha", "lgd_beta", "degrees_of_freedom",
                     "pmc_bandwidth", "ce_bandwidth", "ce_max_shift",
                     "threshold_multiplier", "quadrature_tolerance", "root_tolerance"):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not isfinite(self.skew_shape):
            raise ValueError("skew_shape must be finite")
        if not self.confidence_levels or any(not 0 < x < 1 for x in self.confidence_levels):
            raise ValueError("Confidence levels must lie in (0, 1)")
        if not 0 < self.ce_tail_fraction < 1 or not 0 < self.ce_smoothing <= 1:
            raise ValueError("Invalid CE elite fraction or smoothing")
        if not 0 < self.skew_table_tail < self.probability_clip < 0.5:
            raise ValueError("Require 0 < skew_table_tail < probability_clip < 0.5")
        if not isinstance(self.var_values, dict):
            raise ValueError("var_values must be a dict keyed by copula name")
        for name, values in self.var_values.items():
            if name not in {"gaussian", "t", "skew_t"} or len(values) != len(self.confidence_levels):
                raise ValueError("Supply one VaR per confidence level for each configured copula")
            if any(not 0 < x < self.n_obligors * self.exposure for x in values):
                raise ValueError("Supplied VaR must be inside the positive loss support")


CONFIG = Config()
