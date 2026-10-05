"""统一运行入口：修改 ALGORITHM 后执行 python codes/main.py。"""
import numpy as np

if __package__:
    from .config import CONFIG
    from . import pmc, ce_rqmc, safa
else:
    from config import CONFIG
    import pmc
    import ce_rqmc
    import safa

# 在这里选择算法："pmc"、"ce_rqmc" 或 "safa"。
ALGORITHM = "pmc"
ALGORITHMS = {"pmc": pmc.run, "ce_rqmc": ce_rqmc.run, "safa": safa.run}


def main():
    CONFIG.validate(ALGORITHM)
    results = ALGORITHMS[ALGORITHM](CONFIG)
    with np.printoptions(precision=CONFIG.print_precision, suppress=True):
        print(f"Algorithm: {ALGORITHM}")
        if ALGORITHM == "ce_rqmc":
            print(f"Loss threshold: {results[0]['loss_threshold']}")
            print(f"mu*: {results[0]['mu_star']}")
            print(f"Shared pilot time: {results[0]['pilot_time']:.2f} s")
        for result in results:
            print(f"\nConfidence: {result['alpha']}, VaR: {result['a']:.{CONFIG.print_precision}f}")
            if "se_VaR" in result:
                print(f"VaR SE: {result['se_VaR']:.{CONFIG.print_precision}f}")
                print(f"Mean samples in VaR band: {result['samples']:.1f}")
            print("Obligor       VaRC       SE")
            for i, (mean, se) in enumerate(zip(result["mean_VaRC"], result["se_VaRC"]), 1):
                print(f"{i:7d} {mean:10.{CONFIG.print_precision}f} {se:10.{CONFIG.print_precision}f}")
            print(f"Time: {result['total_time']:.2f} s")
    return results


if __name__ == "__main__":
    main()
