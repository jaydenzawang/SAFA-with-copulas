# 快速运行

**只需修改两个文件：`main.py` 选算法和 copula，`config.py` 改参数。**

### 1. 选择算法和 copula

在 `main.py` 中修改：

```python
ALGORITHM = "safa"      # 可选："pmc"、"ce_rqmc"、"safa"
COPULA = "skew_t"       # 可选："gaussian"、"t"、"skew_t"
```

### 2. 修改基本参数

在 `config.py` 的 `Config` 中修改对应数值，其余参数保持默认即可。

| 参数 | 含义 |
|---|---|
| `n_obligors` | 债务人数量 |
| `p_low`、`p_high` | 违约概率范围 |
| `exposure` | 每个债务人的 EAD |
| `rho_d`、`rho_l` | 违约和 LGD 的因子载荷 |
| `lgd_alpha`、`lgd_beta` | Beta LGD 的两个形状参数 |
| `degrees_of_freedom` | t / skew-t 的自由度 |
| `skew_shape` | skew-t 的偏度参数，设为 0 时退化为 t |
| `confidence_levels` | 置信水平，例如 `(0.95,)` 表示只算 95% |
| `repetitions` | 重复次数，计算标准误至少需要 2 次 |

**先试跑**：默认样本量较大，可以先把下面参数调小：

```python
repetitions: int = 2
var_pilot_samples: int = 20_000  # CE / SAFA 自动预估 VaR

pmc_samples: int = 20_000       # PMC 样本量
ce_pilot_samples: int = 2_000   # CE 预估样本量
ce_samples: int = 20_000        # CE 正式样本量
safa_outer: int = 100           # SAFA 外层样本量
safa_inner: int = 100           # SAFA 内层样本量
```

只需调整所选算法的样本量。正式计算时增大样本量和重复次数；`var_values` 保持默认即可自动估计 VaR。

