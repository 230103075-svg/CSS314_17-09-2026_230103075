"""
Task 23 — Statistical fit of Gustafson's law to the measured weak-scaling data.

Usage (from the repository root, after bench.py):
    py benchmarks/fit_gustafson.py

Reads  benchmarks/results.csv
Writes benchmarks/fit_results.json

Method
  T1        = mean wall time at P = 1
  S_run(P)  = P * T1 / wall_run(P)              (scaled speedup of one run)
  per P     : mean, sample std (ddof=1), standard error = std / sqrt(runs)
  alpha     = sum (P-1)(P-S) / sum (P-1)^2      (least squares, S = P - alpha(P-1))
  R^2       = 1 - SS_res / SS_tot
  Karp-Flatt e(P) = (1/S - 1/P) / (1 - 1/P)      (P > 1)
No intermediate rounding: all values are stored at full float precision.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "benchmarks" / "results.csv"
DST = ROOT / "benchmarks" / "fit_results.json"


def mean(v):
    return sum(v) / len(v)


def std(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def main():
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found: run benchmarks/weak_scale_bench/bench.py first")

    wall, comp = defaultdict(list), defaultdict(list)
    with open(SRC, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            P = int(r["P"])
            wall[P].append(float(r["wall_s"]))
            comp[P].append(float(r["worker_compute_mean_s"]))

    Ps = sorted(wall)
    T1 = mean(wall[1])
    C1 = mean(comp[1])

    per_p = []
    for P in Ps:
        s_runs = [P * T1 / w for w in wall[P]]
        n = len(s_runs)
        S = mean(s_runs)
        sd = std(s_runs)
        per_p.append({
            "P": P, "runs": n,
            "wall_mean_s": mean(wall[P]), "wall_std_s": std(wall[P]),
            "wall_se_s": std(wall[P]) / math.sqrt(n),
            "speedup_mean": S, "speedup_std": sd, "speedup_se": sd / math.sqrt(n),
            "efficiency": S / P,
            "worker_compute_mean_s": mean(comp[P]),
            "compute_slowdown_vs_P1": mean(comp[P]) / C1,
            "karp_flatt": None if P == 1 else (1 / S - 1 / P) / (1 - 1 / P),
        })

    num = sum((d["P"] - 1) * (d["P"] - d["speedup_mean"]) for d in per_p)
    den = sum((d["P"] - 1) ** 2 for d in per_p)
    alpha = num / den if den else 0.0

    S_obs = [d["speedup_mean"] for d in per_p]
    S_bar = mean(S_obs)
    ss_res = 0.0
    for d in per_p:
        fit = d["P"] - alpha * (d["P"] - 1)
        d["speedup_fit"] = fit
        d["residual"] = d["speedup_mean"] - fit
        ss_res += d["residual"] ** 2
    ss_tot = sum((s - S_bar) ** 2 for s in S_obs)
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0

    # alpha standard error (one-parameter regression through the model)
    dof = len(per_p) - 1
    alpha_se = math.sqrt(ss_res / dof / den) if dof > 0 and den else 0.0

    out = {"T1_mean_s": T1, "alpha": alpha, "alpha_se": alpha_se,
           "r_squared": r2, "ss_res": ss_res, "ss_tot": ss_tot,
           "asymptotic_efficiency": 1 - alpha, "per_P": per_p}
    DST.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"T1 (mean, P=1)     = {T1} s")
    print(f"alpha (LSQ)        = {alpha}  (SE {alpha_se})")
    print(f"R^2                = {r2}")
    print(f"limit E = 1-alpha  = {1 - alpha}\n")
    print(f"{'P':>3} {'S mean':>12} {'SE':>10} {'S fit':>12} {'resid':>11} {'E':>9} {'compute x':>10}")
    for d in per_p:
        print(f"{d['P']:>3} {d['speedup_mean']:>12.6f} {d['speedup_se']:>10.6f} "
              f"{d['speedup_fit']:>12.6f} {d['residual']:>11.6f} {d['efficiency']:>9.6f} "
              f"{d['compute_slowdown_vs_P1']:>10.6f}")
    print(f"\nWritten {DST}")


if __name__ == "__main__":
    main()
