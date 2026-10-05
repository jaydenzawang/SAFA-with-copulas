"""所有可调整参数集中于此；main.py 中只选择算法。"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class Config:
    # 组合参数：PD 在 [p_low, p_high] 间均匀分布，EAD 和载荷为同质参数。
    n_obligors: int = 10
    p_low: float = 0.01
    p_high: float = 0.10
    exposure: float = 1.0
    rho: float = sqrt(0.5)
    tau: float = 0.5
    lgd_alpha: float = 2.0
    lgd_beta: float = 5.0

    # CE / SAFA 使用与 confidence_levels 一一对应的外部 VaR。
    confidence_levels: tuple = (0.95, 0.96, 0.97, 0.98, 0.99)
    var_values: tuple = (1.1432, 1.3098, 1.5397, 1.8733, 2.4625)
    repetitions: int = 10
    seed: int = 42
    print_precision: int = 4

    # PMC
    pmc_samples: int = 10_000_000
    pmc_bandwidth: float = 0.01

    # CE-RQMC（保留原脚本独立种子和计算参数）
    ce_seed: int = 2025
    ce_pilot_samples: int = 2_000_000
    ce_samples: int = 10_000_000
    ce_max_iterations: int = 5
    ce_tail_fraction: float = 0.5
    ce_bandwidth: float = 0.005
    ce_shift_step: float = 0.05
    ce_replication_seed_offset: int = 1000
    # 设置为 None 时，运行 PMC 估计 threshold_multiplier × E[L]。
    loss_threshold: float | None = 0.6035668192876804
    threshold_multiplier: float = 3.0
    threshold_samples: int = 10_000_000
    threshold_repetitions: int = 10
    threshold_batch_size: int = 100_000

    # SAFA
    safa_outer: int = 1000
    safa_inner: int = 1000

    # 数值截断
    probability_clip: float = 1e-12
    safa_epsilon_clip: float = 1e-9

    def validate(self, algorithm):
        if algorithm not in {"pmc", "ce_rqmc", "safa"}:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        counts = (self.n_obligors, self.repetitions, self.pmc_samples,
                  self.ce_pilot_samples, self.ce_samples, self.ce_max_iterations,
                  self.threshold_samples, self.threshold_repetitions,
                  self.threshold_batch_size, self.safa_outer, self.safa_inner)
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in counts):
            raise ValueError("Sample counts, repetitions and portfolio size must be positive integers")
        if not 0 < self.p_low <= self.p_high < 1:
            raise ValueError("PDs must satisfy 0 < p_low <= p_high < 1")
        if not abs(self.rho) < 1 or not abs(self.tau) < 1:
            raise ValueError("rho and tau must be strictly between -1 and 1")
        if not all(x > 0 for x in (self.exposure, self.lgd_alpha, self.lgd_beta,
                                   self.pmc_bandwidth, self.ce_bandwidth, self.threshold_multiplier)):
            raise ValueError("Exposure, LGD shapes, bandwidths and threshold multiplier must be positive")
        if not self.confidence_levels or any(not 0 < x < 1 for x in self.confidence_levels):
            raise ValueError("Confidence levels must lie in (0, 1)")
        if algorithm != "pmc" and (len(self.confidence_levels) != len(self.var_values)
                                   or any(not 0 < x < self.n_obligors * self.exposure for x in self.var_values)):
            raise ValueError("Provide one valid VaR per confidence level")
        if not 0 < self.ce_tail_fraction < 1:
            raise ValueError("ce_tail_fraction must lie in (0, 1)")
        if not all(0 < x < 0.5 for x in (self.probability_clip, self.safa_epsilon_clip)):
            raise ValueError("Numerical clipping parameters must lie in (0, 0.5)")
        if self.loss_threshold is not None and not self.loss_threshold >= 0:
            raise ValueError("loss_threshold must be nonnegative or None")


CONFIG = Config()
