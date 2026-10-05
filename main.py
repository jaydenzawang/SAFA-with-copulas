"""统一入口：选择算法和 copula，再运行 python3 codes/main.py。"""
import numpy as np

if __package__:
    from .config import CONFIG
    from . import pmc, ce_rqmc, safa
else:
    from config import CONFIG
    import pmc
    import ce_rqmc
    import safa

ALGORITHM = "pmc"       # "pmc", "ce_rqmc", "safa"
COPULA = "gaussian"     # "gaussian", "t", "skew_t"
ALGORITHMS = {"pmc": pmc.run, "ce_rqmc": ce_rqmc.run, "safa": safa.run}


def main():
    CONFIG.validate(ALGORITHM, COPULA)
    print(f"Algorithm: {ALGORITHM}; copula: {COPULA}; systematic factor: scalar Z", flush=True)
    results = ALGORITHMS[ALGORITHM](CONFIG, COPULA)
    precision = CONFIG.print_precision
    with np.printoptions(precision=precision, suppress=True):
        print(f"VaR source: {results[0]['var_source']}")
        if "var_calibration_time" in results[0]:
            print(f"Shared VaR calibration time: {results[0]['var_calibration_time']:.2f} s")
            print("VaRC SE is conditional on the chosen VaR and fitted proposal; excludes VaR pilot uncertainty.")
        for result in results:
            print(f"\nConfidence: {result['alpha']}; VaR: {result['a']:.{precision}f}")
            if "se_VaR" in result:
                print(f"VaR SE: {result['se_VaR']:.{precision}f}")
            if "samples" in result:
                print(f"Mean samples in VaR band: {result['samples']:.1f}")
            if ALGORITHM == "ce_rqmc":
                print(f"CE auxiliary-normal shift: {result['mu_star']}")
                print(f"Mean band ESS: {result['band_ess']:.1f}; pilot time: {result['pilot_time']:.2f} s")
            if ALGORITHM == "safa":
                print(f"Estimated loss density: {result['density']:.{precision}f}")
            print("Obligor       VaRC       SE")
            for i, (mean, se) in enumerate(zip(result["mean_VaRC"], result["se_VaRC"]), 1):
                print(f"{i:7d} {mean:10.{precision}f} {se:10.{precision}f}")
            if result["time_scope"] == "this_level":
                print(f"Time for this level: {result['total_time']:.2f} s")
        if results[0]["time_scope"] == "all_levels":
            print(f"\nTotal time for all levels: {results[0]['total_time']:.2f} s")
    return results


if __name__ == "__main__":
    main()
