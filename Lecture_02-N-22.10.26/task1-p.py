"""
TASK 1 — Row-Major vs. Column-Major Parallel Cache Traversal

Kystaubay Ayanat — Student ID 230103075

Run:
    py -m pip install numba psutil
    py task1.py

An 8,000 x 8,000 float64 matrix (512 MB) is summed by every physical core
in two different traversal orders:

  Mode A  ROW-MAJOR    each worker takes a slice of ROWS and walks the
                       columns inside them. NumPy stores C-order, so this
                       walks memory front to back, one address after the
                       next.

  Mode B  COLUMN-MAJOR each worker takes a slice of COLUMNS and walks the
                       rows inside them. Consecutive reads are 8,000
                       elements = 64,000 bytes apart, so every single read
                       lands on a different cache line.

Identical arithmetic, identical element count, identical number of adds.
The only difference is the order the addresses are visited in.

Explicit loops, not NumPy
-------------------------
np.sum on a strided view is free to copy the data or pick its own
traversal order, which would measure NumPy's optimiser rather than the
cache. Compiled nested loops read exactly where they are told to. The
order also survives the optimiser: floating-point addition is not
associative, so LLVM may not reorder the accumulation into the other
traversal.
"""
import os
import platform
import statistics
import sys
import threading
import time

import numpy as np

try:
    from numba import njit
except ImportError:
    print("numba is required:  py -m pip install numba", file=sys.stderr)
    raise SystemExit(1)

STUDENT_ID = 230103075
SIZE = 8_000
CACHE_LINE = 64                 # bytes
ITEM = 8                        # float64
REPEATS = 3

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
def sum_rows(M, r0, r1):
    """Mode A: own a band of rows, walk each one left to right."""
    total = 0.0
    ncols = M.shape[1]
    for i in range(r0, r1):
        for j in range(ncols):
            total += M[i, j]
    return total


@njit(nogil=True)
def sum_cols(M, c0, c1):
    """Mode B: own a band of columns, walk each one top to bottom."""
    total = 0.0
    nrows = M.shape[0]
    for j in range(c0, c1):
        for i in range(nrows):
            total += M[i, j]
    return total


def chunk_bounds(n, parts):
    base, extra = divmod(n, parts)
    bounds, start = [], 0
    for i in range(parts):
        stop = start + base + (1 if i < extra else 0)
        if stop > start:
            bounds.append((start, stop))
        start = stop
    return bounds


def run_mode(kernel, M, n_workers):
    """Split the matrix across n_workers threads, return (ms, total)."""
    bounds = chunk_bounds(SIZE, n_workers)
    partials = [0.0] * len(bounds)

    def worker(wid):
        lo, hi = bounds[wid]
        partials[wid] = kernel(M, lo, hi)

    threads = [threading.Thread(target=worker, args=(w,))
               for w in range(len(bounds))]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, float(sum(partials))


def measure(kernel, M, n_workers):
    run_mode(kernel, M, n_workers)                         # warm-up
    runs = [run_mode(kernel, M, n_workers) for _ in range(REPEATS)]
    times = sorted(r[0] for r in runs)
    return statistics.median(times), runs[-1][1], min(times), max(times)


def main():
    logical = os.cpu_count() or 1
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
    except Exception:                                      # noqa: BLE001
        physical = logical

    elements = SIZE * SIZE
    nbytes = elements * ITEM

    out("=" * 78)
    out("TASK 1 — ROW-MAJOR vs. COLUMN-MAJOR PARALLEL CACHE TRAVERSAL")
    out("=" * 78)
    out(f"Student ID           : {STUDENT_ID}")
    out(f"Operating System     : {platform.system()} {platform.release()}")
    out(f"CPU model            : {cpu_model()}")
    out(f"LOGICAL CORE COUNT   : {logical}")
    out(f"Physical core count  : {physical}")
    out(f"Worker threads       : {physical}  (all physical cores)")
    out(f"Matrix               : {SIZE:,} x {SIZE:,} float64")
    out(f"Elements             : {elements:,}")
    out(f"Memory footprint     : {nbytes / 1e6:.1f} MB "
        f"({nbytes / (1024 ** 3):.2f} GiB)")
    out(f"Seed                 : {STUDENT_ID}  (Student ID)")
    out(f"Memory layout        : C-order (row-major), row stride "
        f"{SIZE * ITEM:,} bytes")
    out(f"Timing method        : {REPEATS} timed runs per mode, median reported")
    out()

    out("-" * 78)
    out("ALLOCATION")
    out("-" * 78)
    t0 = time.perf_counter()
    rng = np.random.default_rng(STUDENT_ID)
    M = rng.standard_normal((SIZE, SIZE), dtype=np.float64)
    alloc_ms = (time.perf_counter() - t0) * 1000.0
    out(f"Allocated and filled : {alloc_ms:.1f} ms")
    out(f"C-contiguous         : {M.flags['C_CONTIGUOUS']}")
    reference = float(np.sum(M, dtype=np.float64))
    out(f"Reference sum        : {reference:.17g}   (np.sum)")
    out()

    # compile both kernels before any timing happens
    tiny = np.zeros((4, 4), dtype=np.float64)
    sum_rows(tiny, 0, 4)
    sum_cols(tiny, 0, 4)

    # ---------------------------------------------------------- the runs
    out("=" * 78)
    out("RESULTS")
    out("=" * 78)

    a_ms, a_sum, a_min, a_max = measure(sum_rows, M, physical)
    b_ms, b_sum, b_min, b_max = measure(sum_cols, M, physical)

    # single-threaded reference, for scaling context
    a1_ms, _, _, _ = measure(sum_rows, M, 1)
    b1_ms, _, _, _ = measure(sum_cols, M, 1)

    header = (f"{'Mode':<28} | {'Time (ms)':>11} | {'GB/s':>8} | "
              f"{'ns/element':>11}")
    out(header)
    out("-" * len(header))
    for label, ms in (("A  Row-major, 1 thread", a1_ms),
                      (f"A  Row-major, {physical} threads", a_ms),
                      ("B  Column-major, 1 thread", b1_ms),
                      (f"B  Column-major, {physical} threads", b_ms)):
        gbs = nbytes / (ms / 1000.0) / 1e9
        out(f"{label:<28} | {ms:>11.3f} | {gbs:>8.2f} | "
            f"{ms * 1e6 / elements:>11.4f}")
    out("-" * len(header))
    out()
    out(f"Run spread, Mode A ({physical} threads): "
        f"min {a_min:.3f} ms, max {a_max:.3f} ms")
    out(f"Run spread, Mode B ({physical} threads): "
        f"min {b_min:.3f} ms, max {b_max:.3f} ms")
    out()

    # correctness: both traversals must sum to the same matrix
    ok_a = np.isclose(a_sum, reference, rtol=1e-9)
    ok_b = np.isclose(b_sum, reference, rtol=1e-9)
    out(f"Mode A sum : {a_sum:.17g}   {'MATCHES' if ok_a else 'MISMATCH'}")
    out(f"Mode B sum : {b_sum:.17g}   {'MATCHES' if ok_b else 'MISMATCH'}")
    out("Both modes add the same 64,000,000 numbers in a different order, so")
    out("they agree to floating-point rounding rather than bit-for-bit.")
    out()

    # ------------------------------------------------------- the deliverable
    penalty = b_ms / a_ms
    penalty_1t = b1_ms / a1_ms

    out("=" * 78)
    out("DELIVERABLE")
    out("=" * 78)
    out(f"Student ID                 : {STUDENT_ID}")
    out(f"Logical core count         : {logical}")
    out()
    out(f"Mode A runtime (row-major) : {a_ms:.3f} ms")
    out(f"Mode B runtime (col-major) : {b_ms:.3f} ms")
    out()
    out(f"CACHE PENALTY FACTOR       : {penalty:.4f}x")
    out(f"  (Mode B time / Mode A time, both on {physical} threads)")
    out(f"  single-threaded, for reference : {penalty_1t:.4f}x")
    out("=" * 78)
    out()

    # ---------------------------------------------------------- analysis
    out("-" * 78)
    out("WHY")
    out("-" * 78)
    per_line = CACHE_LINE // ITEM
    out(f"A cache line is {CACHE_LINE} bytes, so it holds {per_line} float64 values.")
    out("The hardware never loads one number; it loads the whole line.")
    out()
    out(f"Mode A walks a row. Addresses advance by {ITEM} bytes, so one line")
    out(f"load serves the next {per_line} reads. Of the {CACHE_LINE} bytes "
        f"pulled from RAM, all")
    out(f"{CACHE_LINE} are used. The hardware prefetcher also recognises the")
    out("straight-line pattern and starts fetching ahead, so the core rarely")
    out("waits at all.")
    out()
    out(f"Mode B walks a column. Consecutive reads are one full row apart:")
    out(f"{SIZE:,} elements x {ITEM} bytes = {SIZE * ITEM:,} bytes. Every read")
    out(f"therefore lands on a new cache line, and only {ITEM} of the "
        f"{CACHE_LINE} bytes")
    out(f"fetched are ever used before the line is evicted. That is "
        f"{(1 - ITEM / CACHE_LINE) * 100:.1f}% of")
    out(f"the memory bandwidth spent moving bytes nobody reads, and it predicts")
    out(f"a penalty of about {CACHE_LINE // ITEM}x from bandwidth waste alone.")
    out()
    out(f"Measured penalty was {penalty:.2f}x.")
    if penalty > per_line:
        out(f"That is worse than the {per_line}x bandwidth argument predicts, "
            f"which is")
        out("expected. The stride also defeats the prefetcher, and with a row")
        out(f"stride of {SIZE * ITEM:,} bytes each access tends to fall in a")
        out("different memory page, so the TLB misses as well. Cache misses,")
        out("failed prefetch and TLB pressure compound.")
    elif penalty > 1.5:
        out(f"That is in line with the bandwidth argument. Part of the "
            f"theoretical {per_line}x")
        out("is recovered because several columns are in flight at once across")
        out("the worker threads, so some lines get reused before eviction.")
    else:
        out("That is milder than expected. With this matrix and this core")
        out("count the working set per thread may be small enough that lines")
        out("survive in cache between uses.")
    out()
    out("The practical rule: iterate in the order the array is stored. In C,")
    out("C++, Python/NumPy and Java that is row-major, so the rightmost index")
    out("belongs in the innermost loop. In Fortran, MATLAB and Julia the")
    out("storage is column-major and the rule inverts. Getting this backwards")
    out("costs an order of magnitude and changes nothing about the output.")
    out("=" * 78)

    with open("task1_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWritten: task1_output.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
