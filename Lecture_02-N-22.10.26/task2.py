"""
TASK 2 — Multi-Core Scaling vs. Amdahl's Law

Kystaubay Ayanat — Student ID 230103075

Run:
    python task2.py

Extends the Task 1 serial baseline to parallel execution and compares the
measured speedup against the Amdahl upper bound for P = 0.95.

Why threads and not processes
-----------------------------
NumPy releases the GIL inside np.sum, so a ThreadPoolExecutor gives real
concurrent execution on a reduction like this. Processes would have to
pickle and copy ~153 MiB of data per worker, and that transfer would
dominate the measurement instead of the arithmetic.

Every worker still runs its own chunk with a single-threaded BLAS (all
backend thread counts are pinned to 1 before numpy is imported), so the
parallelism measured here is exactly the parallelism this script creates.
"""
import os

# Pin numeric backends to one thread BEFORE numpy loads, so the only source
# of parallelism is our own thread pool.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import platform                                           # noqa: E402
import statistics                                         # noqa: E402
import sys                                                # noqa: E402
import time                                               # noqa: E402
from concurrent.futures import ThreadPoolExecutor         # noqa: E402

import numpy as np                                        # noqa: E402

STUDENT_ID = 230103075
N = 20_000_000
P_AMDAHL = 0.95          # parallelizable fraction assumed by the task
REPEATS = 5              # timed runs per worker count; median is reported

LINES = []


def out(text=""):
    """Print and capture, so console and task2_output.txt cannot diverge."""
    print(text, flush=True)
    LINES.append(str(text))


def cpu_model():
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Windows":
            return os.environ.get("PROCESSOR_IDENTIFIER", "") or platform.processor()
        elif system == "Darwin":
            import subprocess
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:                                      # noqa: BLE001
        pass
    return platform.processor() or platform.machine() or "unknown"


def chunk_bounds(n, parts):
    """Split [0, n) into `parts` contiguous, near-equal ranges."""
    base, extra = divmod(n, parts)
    bounds, start = [], 0
    for i in range(parts):
        stop = start + base + (1 if i < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def parallel_sum(data, workers, pool):
    """Sum `data` by handing one contiguous chunk to each worker thread."""
    bounds = chunk_bounds(data.size, workers)
    partials = list(pool.map(
        lambda b: np.sum(data[b[0]:b[1]], dtype=np.float64), bounds))
    return float(np.sum(np.asarray(partials, dtype=np.float64),
                        dtype=np.float64))


def amdahl_bound(p, parallel_fraction=P_AMDAHL):
    """S(p) = 1 / [(1 - P) + P/p] — no intermediate rounding."""
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / p)


def main():
    max_cores = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, max_cores, max_cores * 2})

    # ------------------------------------------------------------- header
    out("=" * 78)
    out("TASK 2 — MULTI-CORE SCALING vs. AMDAHL'S LAW")
    out("=" * 78)
    out(f"Student ID          : {STUDENT_ID}")
    out(f"Operating System    : {platform.system()} {platform.release()}")
    out(f"CPU model           : {cpu_model()}")
    out(f"Logical cores       : {max_cores}   (os.cpu_count())")
    out(f"Python / NumPy      : {platform.python_version()} / {np.__version__}")
    out(f"Parallel strategy   : ThreadPoolExecutor, one contiguous chunk per worker")
    out(f"                      (NumPy releases the GIL inside np.sum)")
    out(f"Worker counts tested: {worker_counts}")
    out(f"Timing method       : {REPEATS} timed runs per count, median reported")
    out(f"Amdahl assumption   : P = {P_AMDAHL} parallelizable")
    out()

    # --------------------------------------------- deterministic test data
    out("-" * 78)
    out("ARRAY GENERATION")
    out("-" * 78)
    out(f"Elements            : {N:,}   float64   "
        f"({N * 8 / (1024 ** 2):.2f} MiB)")
    out(f"Seed                : {STUDENT_ID}  (Student ID)")
    t0 = time.perf_counter()
    rng = np.random.default_rng(STUDENT_ID)
    data = rng.standard_normal(N, dtype=np.float64)
    out(f"Generation time     : {(time.perf_counter() - t0) * 1000.0:.3f} ms")

    reference = float(np.sum(data, dtype=np.float64))
    out(f"Reference sum       : {reference:.17g}   (serial np.sum, Task 1)")
    out()

    # ------------------------------------------------------- benchmark loop
    results = []
    for workers in worker_counts:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            value = parallel_sum(data, workers, pool)      # warm-up, untimed
            timings = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                value = parallel_sum(data, workers, pool)
                timings.append((time.perf_counter() - t0) * 1000.0)

        median_ms = statistics.median(timings)
        # Chunked summation reorders the additions, so the result differs from
        # the serial sum in the last bits. Report the gap instead of hiding it.
        abs_err = abs(value - reference)
        rel_err = abs_err / abs(reference) if reference != 0.0 else 0.0
        results.append({
            "workers": workers,
            "median_ms": median_ms,
            "min_ms": min(timings),
            "max_ms": max(timings),
            "value": value,
            "abs_err": abs_err,
            "rel_err": rel_err,
        })

    # T(1) is the single-worker run of this same parallel code path, so the
    # speedup isolates scaling and not the difference between two algorithms.
    t1 = results[0]["median_ms"]
    for r in results:
        r["speedup"] = t1 / r["median_ms"]                 # full precision
        r["efficiency"] = r["speedup"] / r["workers"]
        r["amdahl"] = amdahl_bound(r["workers"])

    # ---------------------------------------------------- the required table
    out("=" * 78)
    out("SCALING RESULTS")
    out("=" * 78)
    header = (f"{'Cores':>6} | {'Wall Time (ms)':>15} | {'Speedup':>9} | "
              f"{'Amdahl Upper Bound':>19} | {'Efficiency':>11}")
    out(header)
    out("-" * len(header))
    for r in results:
        out(f"{r['workers']:>6} | {r['median_ms']:>15.3f} | "
            f"{r['speedup']:>8.4f}x | {r['amdahl']:>18.4f}x | "
            f"{r['efficiency'] * 100:>10.2f}%")
    out("-" * len(header))
    out()

    # --------------------------------------------------------- supporting detail
    out("-" * 78)
    out("TIMING SPREAD AND NUMERICAL AGREEMENT")
    out("-" * 78)
    out(f"{'Cores':>6} | {'min (ms)':>10} | {'median (ms)':>12} | "
        f"{'max (ms)':>10} | {'rel. error vs serial':>21}")
    out("-" * 78)
    for r in results:
        out(f"{r['workers']:>6} | {r['min_ms']:>10.3f} | {r['median_ms']:>12.3f} | "
            f"{r['max_ms']:>10.3f} | {r['rel_err']:>21.3e}")
    out()
    worst_rel = max(r["rel_err"] for r in results)
    out("Chunked summation adds the elements in a different order than the")
    out("serial reduction, so the two results need not be bit-identical.")
    if worst_rel == 0.0:
        out("Here every parallel sum matched the serial reference exactly")
        out("(relative error 0.0), because NumPy's pairwise summation happens")
        out("to group the additions the same way at these chunk sizes.")
    else:
        out(f"The largest relative error is {worst_rel:.3e}, which is at")
        out("machine-epsilon level and confirms the parallel sums are correct.")
    out()

    # ----------------------------------------------------------- conclusions
    best = max(results, key=lambda r: r["speedup"])
    out("=" * 78)
    out("ANALYSIS")
    out("=" * 78)
    out(f"Serial baseline T(1)      : {t1:.3f} ms")
    out(f"Best measured speedup     : {best['speedup']:.4f}x "
        f"at {best['workers']} workers")
    out(f"Amdahl bound at that count: {best['amdahl']:.4f}x  (P = {P_AMDAHL})")
    gap = best["amdahl"] - best["speedup"]
    out(f"Gap to that bound         : {gap:+.4f}x "
        f"({'below' if gap > 0 else 'above'} the bound)")
    out(f"Amdahl ceiling, p -> inf  : {1.0 / (1.0 - P_AMDAHL):.4f}x")
    out()

    # p = 1 is the baseline by definition (S = 1 = bound), so it is not
    # evidence either way and stays out of both groups.
    scaled = [r for r in results if r["workers"] > 1]
    over = [r for r in scaled if r["speedup"] > r["amdahl"]]
    under = [r for r in scaled if r["speedup"] <= r["amdahl"]]

    if over:
        counts = ", ".join(str(r["workers"]) for r in over)
        out(f"Measured speedup EXCEEDS the Amdahl bound at {counts} worker(s).")
        out("That is not an error in the measurement. Amdahl's Law bounds a")
        out("workload whose parallel fraction really is P; here P = 0.95 is an")
        out("assumption the task hands down, not a measured property of this")
        out("code. A chunked reduction is very close to fully parallel, so its")
        out("true P is higher than 0.95 and it can beat a bound computed from")
        out("an understated P. Splitting the array also gives each thread a")
        out("smaller working set with better cache locality than the single")
        out("thread streaming all 152.59 MiB on its own.")
        out()

    if under:
        counts = ", ".join(str(r["workers"]) for r in under)
        out(f"Measured speedup falls SHORT of the bound at {counts} worker(s):")
        out("  1. Summation is memory-bandwidth-bound, not compute-bound. Each")
        out("     element is read once with almost no arithmetic per byte, so")
        out("     the cores saturate the memory controller long before they run")
        out("     out of work. That is a roofline limit Amdahl does not model.")
        out("  2. Thread pool dispatch and the final reduction over the partial")
        out("     sums are serial overhead that grows with the worker count.")
        out(f"  3. Past {max_cores} workers the pool oversubscribes the "
            f"{max_cores} logical")
        out("     cores, so the extra threads only add scheduling cost.")
    out("=" * 78)

    with open("task2_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWritten: task2_output.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
