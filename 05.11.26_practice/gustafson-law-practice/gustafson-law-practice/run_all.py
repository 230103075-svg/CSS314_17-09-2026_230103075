"""
Runs the whole empirical pipeline (Tasks 6-10, 16, 17, 21-24) in order.

    py run_all.py

1. calculations/compute_calculations.py   -> calculations/task06..task17 tables
2. benchmarks/simulator.py                -> simulator_results.csv, plots/simulator_overlay
3. benchmarks/weak_scale_bench/bench.py   -> benchmarks/results.csv, telemetry.json
4. benchmarks/fit_gustafson.py            -> benchmarks/fit_results.json
5. benchmarks/plot_scaling.py             -> plots/scaling_comparison.png/.pdf
6. benchmarks/make_audit.py               -> analysis/task24_hardware_audit.md
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "calculations/compute_calculations.py",
    "benchmarks/simulator.py",
    "benchmarks/weak_scale_bench/bench.py",
    "benchmarks/fit_gustafson.py",
    "benchmarks/plot_scaling.py",
    "benchmarks/make_audit.py",
]

if __name__ == "__main__":
    for s in STEPS:
        print(f"\n>>> {s}")
        r = subprocess.run([sys.executable, str(ROOT / s)], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"Step failed: {s}")
    print("\nALL DONE. Upload the whole folder to GitHub.")
