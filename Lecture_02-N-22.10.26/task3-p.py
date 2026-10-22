"""
TASK 3 — Prefix Scan (Step Efficiency vs. Work Efficiency)

Kystaubay Ayanat — Student ID 230103075

Run:
    py task3-p.py

Inclusive prefix sum over 2,000,000 integers in the range 1..10, computed
two ways and verified against each other.

  SERIAL    output[i] = output[i-1] + input[i]
            One pass, N additions, and every step depends on the one
            before it. There is no parallelism in this formulation at all.

  PARALLEL  the standard two-pass chunked scan:
              pass 1  each worker sums its own chunk           (read only)
              reduce  exclusive scan over the chunk sums       (tiny, serial)
              pass 2  each worker scans its chunk starting from its offset

The dependency that makes the serial version sequential is real, but it
only binds *within* a chunk. Once a worker knows how much came before its
chunk, it can scan that chunk without talking to anyone. That one number
per chunk is all the communication the algorithm needs.

Step efficiency vs work efficiency
----------------------------------
The serial scan performs N additions, which is optimal work. The parallel
scan touches every element twice, once to sum and once to scan, so it
performs about 2N additions. It trades 2x the total work for the ability
to spread that work over p cores, which caps the speedup at roughly p/2
before any overhead is counted. That trade is the whole subject of this
task: fewer sequential STEPS bought with more total WORK.
"""
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from numba import njit
except ImportError:
    print("numba is required:  py -m pip install numba", file=sys.stderr)
    raise SystemExit(1)

STUDENT_ID = 230103075
N = 2_000_000
LOW, HIGH = 1, 10               # inclusive value range
REPEATS = 5

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


# ---------------------------------------------------------------- kernels
@njit(nogil=True)
def scan_serial(x, out_arr):
    """output[i] = output[i-1] + input[i], strictly sequential."""
    running = np.int64(0)
    for i in range(x.size):
        running += x[i]
        out_arr[i] = running
    return running


@njit(nogil=True)
def chunk_sum(x, lo, hi):
    """Pass 1: total of one chunk. Reads only, writes nothing."""
    s = np.int64(0)
    for i in range(lo, hi):
        s += x[i]
    return s


@njit(nogil=True)
def scan_from_offset(x, out_arr, lo, hi, offset):
    """Pass 2: inclusive scan of one chunk, seeded with everything before it."""
    running = offset
    for i in range(lo, hi):
        running += x[i]
        out_arr[i] = running
    return running


def chunk_bounds(n, parts):
    base, extra = divmod(n, parts)
    bounds, start = [], 0
    for i in range(parts):
        stop = start + base + (1 if i < extra else 0)
        if stop > start:
            bounds.append((start, stop))
        start = stop
    return bounds


def scan_parallel(x, out_arr, bounds, pool):
    """Two passes over the data plus one tiny serial reduction between them."""
    sums = list(pool.map(lambda b: chunk_sum(x, b[0], b[1]), bounds))

    # Exclusive scan over the chunk sums: offsets[k] = everything before chunk k.
    offsets, acc = [], np.int64(0)
    for s in sums:
        offsets.append(acc)
        acc += s

    list(pool.map(lambda t: scan_from_offset(x, out_arr, t[0][0], t[0][1], t[1]),
                  zip(bounds, offsets)))
    return acc


def main():
    logical = os.cpu_count() or 1
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
    except Exception:                                      # noqa: BLE001
        physical = logical
    workers = logical

    out("=" * 78)
    out("TASK 3 — PREFIX SCAN (STEP EFFICIENCY vs. WORK EFFICIENCY)")
    out("=" * 78)
    out(f"Student ID           : {STUDENT_ID}")
    out(f"Operating System     : {platform.system()} {platform.release()}")
    out(f"CPU model            : {cpu_model()}")
    out(f"Logical cores        : {logical}")
    out(f"Physical cores       : {physical}")
    out(f"Workers              : {workers}")
    out(f"Elements             : {N:,}  int64, values {LOW}..{HIGH}")
    out(f"Seed                 : {STUDENT_ID}  (Student ID)")
    out(f"Timing method        : {REPEATS} timed runs each, median reported")
    out()

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.integers(LOW, HIGH + 1, size=N, dtype=np.int64)
    out(f"Input sum            : {int(data.sum()):,}  "
        f"(the value the last scan element must hold)")
    out(f"Input min / max      : {int(data.min())} / {int(data.max())}")
    out()

    serial_out = np.zeros(N, dtype=np.int64)
    parallel_out = np.zeros(N, dtype=np.int64)
    bounds = chunk_bounds(N, workers)

    # Compile every kernel before timing anything.
    tiny = np.ones(8, dtype=np.int64)
    tiny_out = np.zeros(8, dtype=np.int64)
    scan_serial(tiny, tiny_out)
    chunk_sum(tiny, 0, 8)
    scan_from_offset(tiny, tiny_out, 0, 8, np.int64(0))

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        # The pool is created once and reused. Spawning threads per call would
        # cost more than this whole scan takes, and would measure the OS.
        scan_parallel(data, parallel_out, bounds, pool)     # warm-up

        serial_times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            scan_serial(data, serial_out)
            serial_times.append((time.perf_counter() - t0) * 1000.0)

        parallel_times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            scan_parallel(data, parallel_out, bounds, pool)
            parallel_times.append((time.perf_counter() - t0) * 1000.0)
    finally:
        pool.shutdown(wait=True)

    serial_ms = statistics.median(serial_times)
    parallel_ms = statistics.median(parallel_times)
    speedup = serial_ms / parallel_ms

    out("-" * 78)
    out("CHUNK LAYOUT")
    out("-" * 78)
    out(f"Chunks               : {len(bounds)}")
    out(f"Elements per chunk   : {bounds[0][1] - bounds[0][0]:,} "
        f"(last chunk {bounds[-1][1] - bounds[-1][0]:,})")
    out(f"Serial reduction     : {len(bounds)} values — "
        f"{len(bounds) / N * 100:.5f}% of the array")
    out()

    # ------------------------------------------------------- verification
    last_serial = int(serial_out[-1])
    last_parallel = int(parallel_out[-1])
    assertion_holds = last_parallel == last_serial
    full_match = np.array_equal(serial_out, parallel_out)
    expected_total = int(data.sum())

    out("-" * 78)
    out("VERIFICATION")
    out("-" * 78)
    out(f"serial_output[-1]    : {last_serial:,}")
    out(f"parallel_output[-1]  : {last_parallel:,}")
    out(f"sum(input)           : {expected_total:,}")
    out(f"Required assertion   : parallel_output[-1] == serial_output[-1]")
    out(f"  -> {'HOLDS' if assertion_holds else 'FAILED'}")
    out(f"Stronger check, all {N:,} elements identical: "
        f"{'YES' if full_match else 'NO'}")
    out()

    status = "PASS" if (assertion_holds and full_match) else "FAIL"

    # ------------------------------------------------------- the headline
    out("=" * 78)
    out("RESULTS")
    out("=" * 78)
    header = (f"{'Serial Time (ms)':>18} | {'Parallel Time (ms)':>20} | "
              f"{'Speedup':>9} | {'Verification':>13}")
    out(header)
    out("-" * len(header))
    out(f"{serial_ms:>18.3f} | {parallel_ms:>20.3f} | {speedup:>8.4f}x | "
        f"{status:>13}")
    out("-" * len(header))
    out()
    out(f"Serial   spread      : min {min(serial_times):.3f} ms, "
        f"max {max(serial_times):.3f} ms")
    out(f"Parallel spread      : min {min(parallel_times):.3f} ms, "
        f"max {max(parallel_times):.3f} ms")
    out()

    # ---------------------------------------------------------- analysis
    out("-" * 78)
    out("STEP EFFICIENCY vs WORK EFFICIENCY")
    out("-" * 78)
    out(f"Serial additions     : {N:,}        (one pass, optimal work)")
    out(f"Parallel additions   : {2 * N:,}        (sum pass + scan pass)")
    out(f"Work ratio           : 2.00x")
    out(f"Ceiling from that    : {workers} workers / 2 passes = "
        f"{workers / 2:.2f}x maximum speedup")
    out(f"Measured             : {speedup:.4f}x")
    out()
    if speedup >= workers / 2 * 0.7:
        out("The measured speedup is close to that ceiling, so the two passes")
        out("are spreading across the cores about as well as they can.")
    elif speedup > 1.0:
        out("The measured speedup sits below the ceiling. At two million")
        out("elements the array is small: dispatching work to the pool and")
        out("collecting the chunk sums costs a fixed amount that does not")
        out("shrink with core count, and here it is a visible fraction of a")
        out("scan that only takes a few milliseconds.")
    else:
        out("The parallel version is SLOWER than the serial one. That is a")
        out("real result, not a bug: this scan is short enough that doubling")
        out("the work and paying the dispatch cost outweighs splitting it.")
        out("The parallel scan wins on larger inputs or more cores; below the")
        out("crossover the correct choice is the serial scan.")
    out()
    out("The serial scan is WORK-efficient: N additions, the theoretical")
    out("minimum. But it is not STEP-efficient at all, because every one of")
    out("those N additions waits for the previous one. Its critical path is N")
    out("steps long, so no number of cores can help it.")
    out()
    out("The chunked scan shortens the critical path to roughly N/p plus the")
    out(f"{len(bounds)}-element reduction, at the price of doing 2N additions")
    out("instead of N. That is the trade in the task's title. The classic")
    out("textbook pair makes the same point more sharply: Hillis-Steele scan")
    out("finishes in O(log n) steps but performs O(n log n) work, while")
    out("Blelloch scan keeps work at O(n) and takes 2 log n steps. One buys")
    out("latency with work, the other buys work with latency.")
    out()
    out("Note how little communication the chunked version needs: one integer")
    out(f"per chunk, {len(bounds)} numbers total out of {N:,}. The dependency "
        f"that")
    out("makes prefix sum look inherently sequential turns out to bind only")
    out("inside a chunk; across chunks it compresses to a single offset.")
    out("=" * 78)

    with open("task3-p_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWritten: task3-p_output.txt", file=sys.stderr)

    # The task requires a non-zero exit code if verification fails.
    if status != "PASS":
        print("VERIFICATION FAILED — exiting with code 1", file=sys.stderr)
        sys.exit(1)
    assert parallel_out[-1] == serial_out[-1]


if __name__ == "__main__":
    main()
