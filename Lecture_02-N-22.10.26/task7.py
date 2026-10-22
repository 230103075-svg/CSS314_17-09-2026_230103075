"""
TASK 7 — Arithmetic Intensity vs. Memory Bottlenecks

Kystaubay Ayanat — Student ID 230103075

Run:
    py task7.py

Two kernels over the same 30,000,000-element float64 array, each run on
1, 2 and all available cores:

  A  MEMORY-BOUND   B[i] = A[i]
                    16 bytes moved per element, zero arithmetic.

  B  COMPUTE-BOUND  B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])
                    the same 16 bytes moved, but hundreds of cycles of
                    transcendental work per element.

The two differ only in arithmetic intensity: flops performed per byte
moved. That single difference decides whether adding cores helps.

Scaling efficiency = Speedup / Cores. An efficiency near 1.0 means each
added core did a full core's worth of work. An efficiency that collapses
as cores are added means the cores are queueing for something they all
share — here, the memory controller.

Threads, not processes: NumPy releases the GIL inside these kernels, so
threads give true parallelism without copying 240 MB per worker.
"""
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103075
N = 30_000_000
REPEATS = 3
BYTES_PER_ELEMENT = 16          # one 8-byte read + one 8-byte write

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


def chunk_bounds(n, parts):
    base, extra = divmod(n, parts)
    bounds, start = [], 0
    for i in range(parts):
        stop = start + base + (1 if i < extra else 0)
        if stop > start:
            bounds.append((start, stop))
        start = stop
    return bounds


# ---------------------------------------------------------------- kernels
def kernel_memory(a, b, scratch, lo, hi):
    """B[i] = A[i] — pure data movement."""
    np.copyto(b[lo:hi], a[lo:hi])


def kernel_compute(a, b, scratch, lo, hi):
    """B[i] = sin(A[i]) * cos(A[i]) + tan(A[i]).

    Every step writes through `out=` so the loop allocates nothing; what
    is timed is arithmetic, not the memory allocator.
    """
    np.sin(a[lo:hi], out=b[lo:hi])
    np.cos(a[lo:hi], out=scratch[lo:hi])
    np.multiply(b[lo:hi], scratch[lo:hi], out=b[lo:hi])
    np.tan(a[lo:hi], out=scratch[lo:hi])
    np.add(b[lo:hi], scratch[lo:hi], out=b[lo:hi])


def run_kernel(kernel, a, b, scratch, cores, pool):
    """One full pass over the array, split across `cores` workers."""
    bounds = chunk_bounds(N, cores)
    if cores == 1:
        lo, hi = bounds[0]
        t0 = time.perf_counter()
        kernel(a, b, scratch, lo, hi)
        return time.perf_counter() - t0

    def piece(bnd):
        kernel(a, b, scratch, bnd[0], bnd[1])

    t0 = time.perf_counter()
    list(pool.map(piece, bounds))
    return time.perf_counter() - t0


def measure(kernel, a, b, scratch, cores, pool):
    run_kernel(kernel, a, b, scratch, cores, pool)          # warm-up
    times = [run_kernel(kernel, a, b, scratch, cores, pool)
             for _ in range(REPEATS)]
    return statistics.median(times)


def main():
    max_cores = os.cpu_count() or 1
    core_counts = sorted({1, 2, max_cores})

    out("=" * 78)
    out("TASK 7 — ARITHMETIC INTENSITY vs. MEMORY BOTTLENECKS")
    out("=" * 78)
    out(f"Student ID          : {STUDENT_ID}")
    out(f"Operating System    : {platform.system()} {platform.release()}")
    out(f"CPU model           : {cpu_model()}")
    out(f"Logical cores       : {max_cores}")
    out(f"Core counts tested  : {core_counts}")
    out(f"Elements            : {N:,}   float64")
    out(f"Array size          : {N * 8 / 1e6:.1f} MB each "
        f"(A, B and one scratch buffer = {N * 8 * 3 / 1e9:.2f} GB total)")
    out(f"Timing method       : {REPEATS} timed runs per point, median reported")
    out(f"Python / NumPy      : {platform.python_version()} / {np.__version__}")
    out()
    out("Kernel A (memory-bound) : B[i] = A[i]")
    out("Kernel B (compute-bound): B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])")
    out()

    out("-" * 78)
    out("ALLOCATION")
    out("-" * 78)
    t0 = time.perf_counter()
    a = np.random.rand(N).astype(np.float64)
    b = np.empty_like(a)
    scratch = np.empty_like(a)
    out(f"Allocated and filled in {(time.perf_counter() - t0) * 1000:.1f} ms")
    out()

    pool = ThreadPoolExecutor(max_workers=max_cores)
    results = {"A": {}, "B": {}}
    try:
        for name, kernel in (("A", kernel_memory), ("B", kernel_compute)):
            for cores in core_counts:
                results[name][cores] = measure(kernel, a, b, scratch, cores, pool)

        # correctness spot-check of the compute kernel
        kernel_compute(a, b, scratch, 0, N)
        sample = np.arange(0, N, N // 7)
        expected = np.sin(a[sample]) * np.cos(a[sample]) + np.tan(a[sample])
        matches = np.allclose(b[sample], expected, rtol=1e-12, atol=1e-12)
    finally:
        pool.shutdown(wait=True)

    # ------------------------------------------------------------- tables
    for name, title in (("A", "KERNEL A — MEMORY-BOUND   B[i] = A[i]"),
                        ("B", "KERNEL B — COMPUTE-BOUND  "
                              "B[i] = sin(A)*cos(A) + tan(A)")):
        out("=" * 78)
        out(title)
        out("=" * 78)
        header = (f"{'Cores':>6} | {'Time (ms)':>11} | {'Speedup':>9} | "
                  f"{'Efficiency':>11} | {'Bandwidth':>13}")
        out(header)
        out("-" * len(header))
        t1 = results[name][1]
        for cores in core_counts:
            t = results[name][cores]
            speedup = t1 / t
            eff = speedup / cores
            gbs = N * BYTES_PER_ELEMENT / t / 1e9
            out(f"{cores:>6} | {t * 1000:>11.3f} | {speedup:>8.4f}x | "
                f"{eff * 100:>10.2f}% | {gbs:>9.2f} GB/s")
        out("-" * len(header))
        out()

    # --------------------------------------------------------- the verdict
    eff_a = (results["A"][1] / results["A"][max_cores]) / max_cores
    eff_b = (results["B"][1] / results["B"][max_cores]) / max_cores
    bw_a = [N * BYTES_PER_ELEMENT / results["A"][c] / 1e9 for c in core_counts]
    bw_growth = bw_a[-1] / bw_a[0]

    out("=" * 78)
    out("VERDICT — WHICH KERNEL SATURATED THE MEMORY CHANNEL")
    out("=" * 78)
    out(f"Kernel A efficiency at {max_cores} cores : {eff_a * 100:.2f}%")
    out(f"Kernel B efficiency at {max_cores} cores : {eff_b * 100:.2f}%")
    out(f"Kernel A bandwidth, 1 core          : {bw_a[0]:.2f} GB/s")
    out(f"Kernel A bandwidth, {max_cores} cores{' ' * (10 - len(str(max_cores)))}: "
        f"{bw_a[-1]:.2f} GB/s")
    out(f"Kernel A bandwidth growth           : {bw_growth:.2f}x "
        f"for {max_cores}x the cores")
    out()

    if eff_a < eff_b:
        out(">>> KERNEL A (the vector copy) STALLED IN SCALING.")
        out(">>> Its bottleneck is memory channel saturation, not the CPU.")
        out()
        out(f"Kernel A does zero arithmetic per element, so every core spends")
        out(f"its entire time waiting on the memory bus. One core already pulls")
        out(f"{bw_a[0]:.2f} GB/s; adding {max_cores - 1} more only reaches "
            f"{bw_a[-1]:.2f} GB/s, because")
        out("they are all queueing at the same memory controller. The extra")
        out(f"cores are idle inside a stall, which is why efficiency fell to")
        out(f"{eff_a * 100:.2f}%.")
        out()
        out(f"Kernel B moves exactly the same 16 bytes per element, but each")
        out("element also costs three transcendental evaluations. That work")
        out("happens in the core's own execution units, which are private and")
        out(f"therefore scale: efficiency held at {eff_b * 100:.2f}%.")
    else:
        out(">>> Kernel B scaled no better than Kernel A on this run.")
        out(f"    Kernel A efficiency {eff_a * 100:.2f}%, "
            f"kernel B efficiency {eff_b * 100:.2f}%.")
        out("    On a machine with few cores both kernels can be limited by")
        out("    the same small amount of parallelism, and the two bottlenecks")
        out("    are not separable. The distinction needs more cores to show.")
    out()

    out("-" * 78)
    out("ARITHMETIC INTENSITY — WHY THE TWO DIFFER")
    out("-" * 78)
    out("Arithmetic intensity is flops performed per byte moved from memory.")
    out("It is the number that decides which ceiling a kernel hits.")
    out()
    out(f"  Kernel A : 0 flops / {BYTES_PER_ELEMENT} bytes = 0.00 flops per byte")
    out("             Nothing is computed. The kernel is a pure bandwidth test")
    out("             and sits on the sloped part of the roofline, where")
    out("             performance is bandwidth x intensity and nothing else.")
    out()
    out(f"  Kernel B : ~3 transcendental calls / {BYTES_PER_ELEMENT} bytes")
    out("             sin, cos and tan each cost on the order of tens to")
    out("             hundreds of cycles. Intensity is high enough that the")
    out("             kernel sits on the flat part of the roofline, limited by")
    out("             the cores' arithmetic throughput, which grows with core")
    out("             count because each core has its own execution units.")
    out()
    out("The memory controller is shared; execution units are not. That is the")
    out("whole reason the two kernels scale differently, and it is why Amdahl's")
    out("Law alone cannot predict parallel performance: it models serial")
    out("fractions, not contention for a shared resource.")
    out()
    out(f"Compute-kernel correctness spot-check: "
        f"{'PASSED' if matches else 'FAILED'}")
    out("=" * 78)

    with open("task7_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWritten: task7_output.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
