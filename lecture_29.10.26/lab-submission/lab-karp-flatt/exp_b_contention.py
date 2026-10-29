"""
Part 2 — Experiment B: The Contention Traffic Jam

Usage:
    py exp_b_contention.py

No serial sleep any more: the work is fully parallel in principle, but
every one of the 2,000,000 increments takes one shared mutex and touches
one shared integer. The lock and the counter's cache line bounce between
cores, so adding processes makes the run SLOWER and Karp-Flatt e(p)
surges instead of staying flat.

Each P is measured REPEATS times and the median is used.
Writes results/exp_b.json
"""
import json
import statistics
import time
from multiprocessing import Lock, Process, Value
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
CORES = [1, 2, 4]
REPEATS = 3


def contended_worker(shared_counter, lock, iterations):
    for _ in range(iterations):
        with lock:  # Heavy lock contention & cache-line bouncing
            shared_counter.value += 1


def run_contended(p, total_increments=2_000_000):
    increments_per_worker = total_increments // p
    counter = Value('i', 0)
    lock = Lock()
    procs = [
        Process(target=contended_worker, args=(counter, lock, increments_per_worker))
        for _ in range(p)
    ]
    t0 = time.perf_counter()
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    t1 = time.perf_counter()
    assert counter.value == total_increments
    return t1 - t0


if __name__ == "__main__":
    print("=" * 66)
    print("EXPERIMENT B — LOCK CONTENTION (2,000,000 locked increments)")
    print("=" * 66)
    runtimes, all_runs = {}, {}
    for p in CORES:
        runs = []
        for r in range(REPEATS):
            print(f"  P={p} run {r + 1}/{REPEATS} ...", flush=True)
            runs.append(run_contended(p))
        all_runs[p] = runs
        runtimes[p] = statistics.median(runs)

    t1 = runtimes[1]
    print()
    print(f"P=1 | Runtime={t1:.3f}s | Speedup=1.00x | (baseline)")
    e = {}
    for p in [2, 4]:
        tp = runtimes[p]
        sp = t1 / tp
        e[p] = ((p * tp / t1) - 1.0) / (p - 1.0)
        print(f"P={p} | Runtime={tp:.3f}s | Speedup={sp:.2f}x | Karp-Flatt e={e[p]:.3f}")

    slower = runtimes[4] > t1
    surge = e[4] > e[2]
    print(f"\nT4 > T1 (negative speedup)    : {'YES' if slower else 'NO'}  "
          f"({runtimes[4]:.3f}s vs {t1:.3f}s)")
    print(f"e(4) > e(2) (contention surge): {'YES' if surge else 'NO'}  "
          f"({e[4]:.3f} vs {e[2]:.3f})")
    print("All counters reached exactly 2,000,000 (asserted on every run).")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "exp_b.json").write_text(json.dumps({
        "experiment": "B_lock_contention", "repeats": REPEATS,
        "T": {str(p): runtimes[p] for p in CORES},
        "runs": {str(p): all_runs[p] for p in CORES},
        "speedup": {str(p): t1 / runtimes[p] for p in CORES},
        "e": {str(p): e[p] for p in [2, 4]},
        "rubric_pass": bool(slower and surge),
    }, indent=2), encoding="utf-8")
    print(f"\nWritten: {RESULTS / 'exp_b.json'}")
