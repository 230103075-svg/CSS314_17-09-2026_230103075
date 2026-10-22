"""
TASK 5 — The Overhead Crossover Point

Kystaubay Ayanat — Student ID 230103075

Run:
    py task5.py

Benchmarks y[i] = a * x[i] + b at N = 100 .. 10,000,000, serial against a
thread pool using every logical core, and locates the N at which going
parallel stops costing more than it saves.

Three numbers are measured rather than assumed:

  c  per-element cost of the serial kernel      (seconds / element)
  O  fixed cost of dispatching work to the pool (seconds, N-independent)
  p  worker count

From those the crossover follows directly. Parallel beats serial when

        O + N*c/p  <  N*c      =>      N  >  O / (c * (1 - 1/p))

The script computes that prediction and then checks it against the
measured curve, so the answer is not just an observation but a number
that can be explained.

Threads, not processes: NumPy releases the GIL inside its element-wise
kernels, so threads give real parallelism here. Processes would have to
pickle and copy up to 80 MB per worker, which would swamp the very
overhead this task is trying to isolate.
"""
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103075
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
ALPHA = np.float64(2.5)
BETA = np.float64(1.25)
TARGET_SECONDS = 0.05          # per timing batch
MEDIAN_OF = 5                  # independent batches per measurement

LINES = []


def out(text=""):
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


def bench(fn, precise=False):
    """Seconds per call, median of auto-sized batches.

    `precise` spends roughly 3x longer per measurement. It is used near the
    crossover, where serial and parallel are within a few percent of each
    other and ordinary noise would decide the answer.
    """
    target = TARGET_SECONDS * 3 if precise else TARGET_SECONDS
    medians = MEDIAN_OF * 2 + 1 if precise else MEDIAN_OF

    fn()                                                   # warm-up
    t0 = time.perf_counter()
    fn()
    single = max(time.perf_counter() - t0, 1e-9)
    reps = int(max(3, min(200_000, target / single)))

    batches = []
    for _ in range(medians):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        batches.append((time.perf_counter() - t0) / reps)
    return statistics.median(batches)


def chunk_bounds(n, parts):
    base, extra = divmod(n, parts)
    bounds, start = [], 0
    for i in range(parts):
        stop = start + base + (1 if i < extra else 0)
        if stop > start:
            bounds.append((start, stop))
        start = stop
    return bounds


def make_serial(x, y):
    def run():
        np.multiply(x, ALPHA, out=y)
        np.add(y, BETA, out=y)
    return run


def make_parallel(x, y, pool, bounds):
    def piece(b):
        lo, hi = b
        np.multiply(x[lo:hi], ALPHA, out=y[lo:hi])
        np.add(y[lo:hi], BETA, out=y[lo:hi])

    def run():
        list(pool.map(piece, bounds))
    return run


def main():
    workers = os.cpu_count() or 1

    out("=" * 78)
    out("TASK 5 — THE OVERHEAD CROSSOVER POINT")
    out("=" * 78)
    out(f"Student ID          : {STUDENT_ID}")
    out(f"Operating System    : {platform.system()} {platform.release()}")
    out(f"CPU model           : {cpu_model()}")
    out(f"Logical cores       : {workers}")
    out(f"Workers             : {workers}  (all available cores)")
    out(f"Kernel              : y[i] = {ALPHA} * x[i] + {BETA}   (float64)")
    out(f"Sizes tested        : {SIZES}")
    out(f"Timing method       : auto-sized batches, median of {MEDIAN_OF}")
    out(f"Parallel strategy   : persistent ThreadPoolExecutor "
        f"(NumPy releases the GIL)")
    out()

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        # ------------------------------------------- fixed dispatch overhead
        out("-" * 78)
        out("1. FIXED COST OF GOING PARALLEL")
        out("-" * 78)
        noop_bounds = list(range(workers))
        dispatch = bench(lambda: list(pool.map(lambda _: None, noop_bounds)))
        out(f"Dispatch {workers} no-op tasks to a warm pool : "
            f"{dispatch * 1e6:.3f} us")

        def spawn_fresh():
            with ThreadPoolExecutor(max_workers=workers) as p:
                list(p.map(lambda _: None, noop_bounds))
        fresh = bench(spawn_fresh)
        out(f"Create a pool, dispatch, tear it down      : "
            f"{fresh * 1e6:.3f} us")
        out()
        out("This is the price paid before a single element is touched. The")
        out("array has to be large enough that the work saved by splitting it")
        out("exceeds this number, or parallelism is a net loss.")
        out()

        # -------------------------------------------------- the main sweep
        out("-" * 78)
        out("2. SERIAL vs PARALLEL ACROSS N")
        out("-" * 78)
        header = (f"{'N':>12} | {'Serial (us)':>13} | {'Parallel (us)':>14} | "
                  f"{'Speedup':>9} | {'Verdict':>8}")
        out(header)
        out("-" * len(header))

        results = []
        for n in SIZES:
            x = np.random.rand(n).astype(np.float64)
            y = np.empty_like(x)
            bounds = chunk_bounds(n, workers)

            t_ser = bench(make_serial(x, y))
            t_par = bench(make_parallel(x, y, pool, bounds))
            speedup = t_ser / t_par

            # correctness: the parallel path must produce the same array
            expected = ALPHA * x + BETA
            make_parallel(x, y, pool, bounds)()
            ok = np.array_equal(y, expected)

            results.append({"n": n, "serial": t_ser, "parallel": t_par,
                            "speedup": speedup, "correct": ok})
            verdict = "FASTER" if speedup > 1.0 else "slower"
            out(f"{n:>12,} | {t_ser * 1e6:>13.3f} | {t_par * 1e6:>14.3f} | "
                f"{speedup:>8.4f}x | {verdict:>8}")

        out("-" * len(header))
        bad = [r["n"] for r in results if not r["correct"]]
        if bad:
            out(f"WARNING: parallel result differed from serial at N = {bad}")
        else:
            out("All parallel results matched the serial result exactly.")
        out()

        # ------------------------------------------- locate the bracket
        out("=" * 78)
        out("3. CROSSOVER POINT")
        out("=" * 78)

        bracket = None
        for lo, hi in zip(results, results[1:]):
            if lo["speedup"] <= 1.0 < hi["speedup"]:
                bracket = (lo, hi)
                break

        if results[0]["speedup"] > 1.0:
            out("Parallel is already faster at the smallest N tested "
                f"({results[0]['n']:,}).")
            out("The crossover lies below the range this task specifies.")
            crossover = None
        elif bracket is None:
            out("Parallel never overtook serial across the tested range.")
            out("On this machine the crossover lies above "
                f"N = {results[-1]['n']:,}.")
            crossover = None
        else:
            lo, hi = bracket
            out(f"Sign change found between N = {lo['n']:,} "
                f"(speedup {lo['speedup']:.4f}x) and "
                f"N = {hi['n']:,} (speedup {hi['speedup']:.4f}x).")
            out()
            out("Refining inside that bracket with 12 geometric steps:")
            out()
            sub_header = (f"{'N':>12} | {'Serial (us)':>13} | "
                          f"{'Parallel (us)':>14} | {'Speedup':>9}")
            out(sub_header)
            out("-" * len(sub_header))

            probes = np.unique(np.geomspace(lo["n"], hi["n"], 12).astype(np.int64))
            crossover = None
            prev = None
            for n in probes:
                n = int(n)
                x = np.random.rand(n).astype(np.float64)
                y = np.empty_like(x)
                bounds = chunk_bounds(n, workers)
                t_ser = bench(make_serial(x, y), precise=True)
                t_par = bench(make_parallel(x, y, pool, bounds), precise=True)
                s = t_ser / t_par
                out(f"{n:>12,} | {t_ser * 1e6:>13.3f} | {t_par * 1e6:>14.3f} | "
                    f"{s:>8.4f}x")
                if crossover is None and s > 1.0:
                    crossover = (prev, n, s)
                prev = n
            out("-" * len(sub_header))
            out()

            if crossover is None:
                out(f"Inside the bracket the speedup never exceeded 1.0, so the")
                out(f"crossover sits just below N = {hi['n']:,}.")
            else:
                below, at, s_at = crossover
                out("=" * 78)
                out("ANSWER — THE CROSSOVER RANGE")
                out("=" * 78)
                if below is None:
                    # The first refined probe already won, so the turn happens
                    # right at the bracket's lower edge.
                    out(f"Multi-threading is SLOWER  at N = {lo['n']:,} "
                        f"in the coarse sweep (speedup {lo['speedup']:.4f}x)")
                    out(f"Multi-threading is FASTER  at N = {at:,} "
                        f"on the longer re-measurement (speedup {s_at:.4f}x)")
                    out()
                    out(f"CROSSOVER RANGE: N ~ {at:,}")
                    out()
                    out("The same N lands on opposite sides of 1.0x in the two")
                    out("passes, which means the two timings differ by less than")
                    out("the run-to-run noise. The crossover sits essentially at")
                    out(f"this N: around {at:,} elements, serial and parallel cost")
                    out("the same on this machine.")
                else:
                    out(f"Multi-threading is SLOWER  at N = {below:,}")
                    out(f"Multi-threading is FASTER  at N = {at:,}  "
                        f"(speedup {s_at:.4f}x)")
                    out()
                    out(f"CROSSOVER RANGE: {below:,} < N <= {at:,}")
                out("=" * 78)
        out()

        # ------------------------------------------------ predict vs measure
        out("-" * 78)
        out("4. PREDICTED CROSSOVER FROM FIRST PRINCIPLES")
        out("-" * 78)
        biggest = results[-1]
        c = biggest["serial"] / biggest["n"]
        out(f"Per-element serial cost c  : {c * 1e9:.4f} ns "
            f"(from N = {biggest['n']:,})")
        out(f"Dispatch overhead O        : {dispatch * 1e6:.3f} us")
        out(f"Workers p                  : {workers}")
        if workers > 1:
            predicted = dispatch / (c * (1.0 - 1.0 / workers))
            out()
            out(f"N* = O / (c * (1 - 1/p)) = {predicted:,.0f} elements")
            out()
            out("That model assumes the parallel part scales perfectly by p.")
            out("It does not: this kernel reads 8 bytes and writes 8 bytes per")
            out("element while doing two flops, so past the cache it is limited")
            out("by memory bandwidth, not by core count. Real scaling is lower")
            out("than p, which pushes the true crossover ABOVE this estimate.")
            out("Treat N* as a lower bound.")
        out()

        out("-" * 78)
        out("5. ANALYSIS")
        out("-" * 78)
        out("At small N the array fits in cache and the whole kernel finishes")
        out("in less time than it takes to hand work to the pool and collect")
        out("the results. The thread pool is pure loss: the speedup column")
        out("sits far below 1.0 and the parallel column is nearly flat,")
        out("because what is being timed is dispatch, not arithmetic.")
        out()
        out("As N grows the arithmetic starts to dominate that fixed cost. The")
        out("parallel time grows roughly N*c/p against the serial N*c, so the")
        out("two curves converge and then cross. Past the crossover the fixed")
        out("overhead is amortised over enough elements to stop mattering.")
        out()
        out("The gain does not keep growing with core count. y = a*x + b does")
        out("two floating-point operations per 16 bytes moved, so once the")
        out("array leaves cache every core is waiting on the same memory")
        out("controller. That is a roofline ceiling, and it is why the speedup")
        out("at the largest N is far below the worker count.")
        out()
        out("Practical reading: splitting an array across cores is only worth")
        out("it once the work per call clears the dispatch cost measured in")
        out("section 1. Below that threshold the correct optimisation is to")
        out("stay on one thread.")
        out("=" * 78)

    finally:
        pool.shutdown(wait=True)

    with open("task5_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWritten: task5_output.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
