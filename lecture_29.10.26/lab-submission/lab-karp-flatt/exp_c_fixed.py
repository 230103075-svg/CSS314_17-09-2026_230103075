"""
Part 3 — Experiment C: The Privatization Fix

Usage:
    py exp_c_fixed.py

Golden Rule 8: privatize before synchronizing. Each worker counts in a
private local variable with zero locks, and the results are combined by a
single reduction at the end.

Two timings are reported for every P:

  handout  : exactly the handout's run_privatized(), where the timer also
             covers creating and tearing down the process pool.
  warm pool: the pool is created BEFORE the timer starts, so only the
             counting and the one final reduction are timed.

On Windows a new process is spawned from scratch, which costs a fixed
amount per pool. That cost is serial overhead, not contention, and the
second column shows how much of the gap to linear scaling it explains.

Each point is measured REPEATS times and the median is used.
Writes results/exp_c.json
"""
import json
import statistics
import time
from multiprocessing import Pool
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
CORES = [1, 2, 4]
REPEATS = 3
TOTAL = 20_000_000


def private_worker(iterations):
    # Thread-local accumulation in private memory space (zero shared state)
    local_sum = 0
    for _ in range(iterations):
        local_sum += 1
    return local_sum


def run_privatized(p, total_increments=TOTAL):
    increments_per_worker = total_increments // p
    tasks = [increments_per_worker] * p
    t0 = time.perf_counter()
    with Pool(processes=p) as pool:
        results = pool.map(private_worker, tasks)
    total = sum(results)  # Synchronize exactly ONCE at the end
    t1 = time.perf_counter()
    assert total == total_increments
    return t1 - t0


def run_privatized_warm(pool, p, total_increments=TOTAL):
    """Same work and reduction, but the pool already exists."""
    increments_per_worker = total_increments // p
    tasks = [increments_per_worker] * p
    t0 = time.perf_counter()
    results = pool.map(private_worker, tasks)
    total = sum(results)
    t1 = time.perf_counter()
    assert total == total_increments
    return t1 - t0


def kf(p, tp, t1):
    return ((p * tp / t1) - 1.0) / (p - 1.0)


def main():
    print("=" * 66)
    print("EXPERIMENT C — PRIVATIZED STATE (20,000,000 increments)")
    print("=" * 66)
    hand, warm = {}, {}
    for p in CORES:
        print(f"  P={p} handout timing ...", flush=True)
        hand[p] = [run_privatized(p) for _ in range(REPEATS)]
        print(f"  P={p} warm-pool timing ...", flush=True)
        with Pool(processes=p) as pool:
            pool.map(private_worker, [1] * p)          # wake every worker
            warm[p] = [run_privatized_warm(pool, p) for _ in range(REPEATS)]

    results = {}
    for name, data in (("handout", hand), ("warm_pool", warm)):
        T = {p: statistics.median(data[p]) for p in CORES}
        t1 = T[1]
        results[name] = {
            "T": T,
            "speedup": {p: t1 / T[p] for p in CORES},
            "efficiency": {p: t1 / T[p] / p for p in CORES},
            "e": {p: kf(p, T[p], t1) for p in CORES[1:]},
            "runs": data,
        }

    for name, title in (("handout", "HANDOUT TIMING (pool start-up included)"),
                        ("warm_pool", "WARM POOL (start-up excluded)")):
        r = results[name]
        print(f"\n{title}")
        print(f"{'P':>2} | {'T_p (s)':>9} | {'Speedup':>8} | {'Eff %':>6} | {'Karp-Flatt e':>12}")
        print("-" * 52)
        for p in CORES:
            e = "-" if p == 1 else f"{r['e'][p]:.3f}"
            print(f"{p:>2} | {r['T'][p]:>9.4f} | {r['speedup'][p]:>7.2f}x | "
                  f"{r['efficiency'][p] * 100:>5.1f} | {e:>12}")

    print()
    for name in ("handout", "warm_pool"):
        r = results[name]
        ok = r["T"][4] < r["T"][1] and r["efficiency"][4] > 0.70
        print(f"Rubric (T4 < T1 and E4 > 70%), {name:<9}: "
              f"{'PASS' if ok else 'NOT MET'}  "
              f"(E4 = {r['efficiency'][4] * 100:.1f}%)")
        results[name]["rubric_pass"] = ok
    h, w = results["handout"], results["warm_pool"]
    print(f"\nPool start-up accounts for "
          f"{(h['T'][4] - w['T'][4]) * 1000:.0f} ms of T4 "
          f"({(h['T'][4] - w['T'][4]) / h['T'][4] * 100:.0f}% of it).")
    print("That overhead is a fixed serial cost of spawning processes on this")
    print("OS, not contention: the counting itself has no shared state at all.")

    RESULTS.mkdir(exist_ok=True)
    def keys(d):                                            # noqa: E306
        return {str(k): v for k, v in d.items()}
    (RESULTS / "exp_c.json").write_text(json.dumps({
        "experiment": "C_privatized", "repeats": REPEATS, "total": TOTAL,
        **{name: {k: keys(v) if isinstance(v, dict) else v
                  for k, v in res.items()} for name, res in results.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nWritten: {RESULTS / 'exp_c.json'}")


if __name__ == "__main__":
    main()
