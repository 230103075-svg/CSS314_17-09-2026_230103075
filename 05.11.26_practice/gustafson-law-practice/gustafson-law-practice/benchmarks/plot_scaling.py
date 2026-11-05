"""
Task 24 (plot) — Measured weak scaling vs ideal, fitted Gustafson and Amdahl.

Usage (from the repository root, after fit_gustafson.py):
    py benchmarks/plot_scaling.py

Reads  benchmarks/fit_results.json
Writes plots/scaling_comparison.png and plots/scaling_comparison.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    fit = json.loads((ROOT / "benchmarks" / "fit_results.json").read_text(encoding="utf-8"))
    a = fit["alpha"]
    d = fit["per_P"]
    P = [x["P"] for x in d]
    S = [x["speedup_mean"] for x in d]
    SE = [x["speedup_se"] for x in d]
    Pd = [1 + i * (max(P) - 1) / 200 for i in range(201)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    ax1.plot(Pd, Pd, ":", color="black", lw=1.2, label="Ideal S = P")
    ax1.plot(Pd, [p - a * (p - 1) for p in Pd], "-", color="#1f77b4", lw=2,
             label=f"Gustafson fit, α = {a:.6f}, R² = {fit['r_squared']:.6f}")
    if a > 0:
        ax1.plot(Pd, [1 / (a + (1 - a) / p) for p in Pd], "--", color="#d62728", lw=1.6,
                 label=f"Amdahl, same s = {a:.6f}")
    ax1.errorbar(P, S, yerr=SE, fmt="o", color="#2ca02c", ms=5, capsize=3,
                 label="Measured mean ± SE")
    ax1.set_xlabel("Worker processes P")
    ax1.set_ylabel("Scaled speedup S(P)")
    ax1.set_title("Weak scaling: measured vs models")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(P, [x["efficiency"] for x in d], "o-", color="#2ca02c", label="Measured E = S/P")
    ax2.plot(Pd, [(p - a * (p - 1)) / p for p in Pd], "-", color="#1f77b4",
             label="Gustafson fit E")
    ax2.plot(P, [1 / x["compute_slowdown_vs_P1"] for x in d], "s--", color="#ff7f0e", ms=4,
             label="Per-worker compute speed vs P=1")
    ax2.axhline(1 - a, color="#1f77b4", ls=":", lw=1, label=f"1 − α = {1 - a:.6f}")
    ax2.set_xlabel("Worker processes P")
    ax2.set_ylabel("Efficiency")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Efficiency and per-core slowdown")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    (ROOT / "plots").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "plots" / f"scaling_comparison.{ext}", dpi=200)
    print(f"Written {ROOT / 'plots' / 'scaling_comparison.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
