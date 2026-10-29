"""
Part 1 — Experiment A: The Stubborn Serial Tail

Usage:
    py exp_a_serial.py

Compute work is spread cleanly over a process pool, but the master first
sleeps 0.40 s before dispatching anything. That sleep is a fixed serial
section, so Karp-Flatt e(p) should stay roughly CONSTANT across p.

Each P is measured REPEATS times and the median is used.
Writes results/exp_a.json
"""
import json
import math
import statistics
import time
from multiprocessing import Pool
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
CORES = [1, 2, 4]
REPEATS = 3


def heavy_math(n):
    return sum(math.sin(i) * math.cos(i) for i in range(1, n))


def run_experiment(workers, total_chunks=16, chunk_size=2_000_000):
    t0 = time.perf_counter()
    time.sleep(0.40)  # Intentional sequential bottleneck (~12% serial fraction)
    with Pool(processes=workers) as pool:
        pool.map(heavy_math, [chunk_size] * total_chunks)
    return time.perf_counter() - t0


def karp_flatt(p, tp, t1):
    return ((p * tp / t1) - 1.0) / (p - 1.0)


def main():
    print("=" * 66)
    print("EXPERIMENT A — STUBBORN SERIAL TAIL (0.40 s master sleep)")
    print("=" * 66)
    times = {}
    for p in CORES:
        runs = []
        for r in range(REPEATS):
            print(f"  P={p} run {r + 1}/{REPEATS} ...", flush=True)
            runs.append(run_experiment(p))
        times[p] = {"median_s": statistics.median(runs), "runs_s": runs}

    t1 = times[1]["median_s"]
    print()
    print(f"{'P':>2} | {'T_p (s)':>9} | {'Speedup':>8} | {'Karp-Flatt e':>12}")
    print("-" * 42)
    print(f"{1:>2} | {t1:>9.3f} | {1.0:>7.2f}x | {'-':>12}")
    e = {}
    for p in CORES[1:]:
        tp = times[p]["median_s"]
        e[p] = karp_flatt(p, tp, t1)
        print(f"{p:>2} | {tp:>9.3f} | {t1 / tp:>7.2f}x | {e[p]:>12.3f}")

    ratio = e[4] / e[2]
    flat = 0.80 <= ratio <= 1.25
    print(f"\ne(4) / e(2) = {ratio:.3f}  -> rubric window [0.80, 1.25]: "
          f"{'PASS' if flat else 'OUTSIDE'}")
    print("A flat e(p) means the loss is a fixed serial code path (the sleep),")
    print("not contention: the same fraction of work stays serial at every p.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "exp_a.json").write_text(json.dumps({
        "experiment": "A_serial_tail", "repeats": REPEATS,
        "T": {str(p): times[p]["median_s"] for p in CORES},
        "runs": {str(p): times[p]["runs_s"] for p in CORES},
        "speedup": {str(p): t1 / times[p]["median_s"] for p in CORES},
        "e": {str(p): e[p] for p in CORES[1:]},
        "e4_over_e2": ratio, "rubric_pass": flat,
    }, indent=2), encoding="utf-8")
    print(f"\nWritten: {RESULTS / 'exp_a.json'}")


if __name__ == "__main__":
    main()
