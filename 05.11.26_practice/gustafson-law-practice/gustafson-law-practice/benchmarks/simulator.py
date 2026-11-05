"""
Task 21 — Parametric speedup simulator: Amdahl vs Gustafson.

Usage (from the repository root):
    py benchmarks/simulator.py
    py benchmarks/simulator.py --alphas 0.001 0.01 0.05 0.1 0.2 --pmax 65536

Writes:
    benchmarks/simulator_results.csv   every (alpha, P) pair, both laws
    plots/simulator_overlay.png/.pdf   strong vs weak scaling overlay
Domain: 1 <= P <= 65,536 (powers of two plus P=1), 0.001 <= alpha <= 0.20.
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def amdahl(P, s):
    return 1.0 / (s + (1.0 - s) / P)


def gustafson(P, a):
    return P - a * (P - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.001, 0.01, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--pmax", type=int, default=65536)
    args = ap.parse_args()

    for a in args.alphas:
        if not 0.001 <= a <= 0.20:
            raise SystemExit(f"alpha {a} outside the supported range [0.001, 0.20]")
    if not 1 <= args.pmax <= 65536:
        raise SystemExit("pmax must be between 1 and 65,536")

    Ps, p = [], 1
    while p <= args.pmax:
        Ps.append(p)
        p *= 2

    rows = []
    for a in args.alphas:
        for P in Ps:
            sa, sg = amdahl(P, a), gustafson(P, a)
            rows.append({"alpha": a, "P": P,
                         "amdahl_speedup": sa, "amdahl_efficiency": sa / P,
                         "gustafson_speedup": sg, "gustafson_efficiency": sg / P,
                         "gustafson_over_amdahl": sg / sa})

    out_csv = ROOT / "benchmarks" / "simulator_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Written {out_csv}  ({len(rows)} rows)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: CSV written, plot skipped "
              "(py -m pip install matplotlib)")
        return

    (ROOT / "plots").mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    cmap = plt.get_cmap("viridis")
    for i, a in enumerate(args.alphas):
        c = cmap(i / max(1, len(args.alphas) - 1))
        ax1.plot(Ps, [gustafson(P, a) for P in Ps], "-", color=c, lw=1.8,
                 label=f"Gustafson α={a}")
        ax1.plot(Ps, [amdahl(P, a) for P in Ps], "--", color=c, lw=1.4,
                 label=f"Amdahl s={a}")
        ax2.plot(Ps, [gustafson(P, a) / P for P in Ps], "-", color=c, lw=1.8)
        ax2.plot(Ps, [amdahl(P, a) / P for P in Ps], "--", color=c, lw=1.4)
    ax1.plot(Ps, Ps, ":", color="black", lw=1, label="ideal S = P")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Processors P")
        ax.grid(True, which="both", alpha=0.3)
    ax1.set_yscale("log", base=2)
    ax1.set_ylabel("Speedup S(P)")
    ax1.set_title("Weak (solid, Gustafson) vs strong (dashed, Amdahl) scaling")
    ax1.legend(fontsize=7, ncol=2)
    ax2.set_ylabel("Efficiency E(P) = S/P")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Efficiency: Gustafson → 1 − α, Amdahl → 0")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "plots" / f"simulator_overlay.{ext}", dpi=200)
    print(f"Written {ROOT / 'plots' / 'simulator_overlay.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
