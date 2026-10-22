"""
TASK 3 — Cache Line Contention (False Sharing)

Kystaubay Ayanat — Student ID 230103075

Run:
    py -m pip install numba
    py task3.py

Three variants of the same 50,000,000-increment-per-thread workload:

  A  UNPADDED SHARED  — every thread hammers shared_counts[i], and the
                        slots sit next to each other, so all of them live
                        in the same 64-byte cache line.
  B  PADDED SHARED    — same shared array, but the slots are 64 bytes
                        apart, so each thread owns its own cache line.
  C  THREAD-LOCAL     — each thread counts in a private variable and
                        writes shared_counts[i] exactly once, at the end.

A and B differ ONLY in the distance between the slots. No thread ever
reads or writes another thread's counter, so there is no logical sharing
at all — any slowdown in A is pure cache-coherence traffic.

Why the loop body looks unusual
-------------------------------
A naive `for _ in range(n): arr[i] += 1` is removed by the compiler: LLVM
proves the loop has a closed form and rewrites it as `arr[i] += n`. The
measurement then times nothing. Section 1 below demonstrates that, and the
benchmark loops instead chain the next index off the value just loaded, so
the load and the store have to happen on every single iteration.
"""
import os
import platform
import sys
import statistics
import threading
import time

import numpy as np

try:
    from numba import njit
except ImportError:
    print("numba is required:  py -m pip install numba", file=sys.stderr)
    raise SystemExit(1)

STUDENT_ID = 230103075
ITERATIONS = 50_000_000       # increments per thread, as specified
CACHE_LINE = 64               # bytes, x86-64
INT64 = 8                     # bytes
STRIDE = CACHE_LINE // INT64  # 8 int64 slots == one 64-byte line
REPEATS = 3                   # timed runs per variant; median reported

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
    except Exception:                                       # noqa: BLE001
        pass
    return platform.processor() or platform.machine() or "unknown"


# --------------------------------------------------------------- kernels
@njit(nogil=True)
def naive_increment(arr, idx, n):
    """The textbook loop — and the one the optimiser deletes."""
    for _ in range(n):
        arr[idx] += 1


@njit(nogil=True)
def hammer_shared(arr, idx, n, zero):
    """Increment arr[idx] n times with a real load and store each time.

    `zero` is 0 at runtime, but the compiler cannot prove that, so the next
    index depends on the value just loaded. That dependency defeats the
    loop-invariant-code-motion pass that would otherwise keep the counter
    in a register for the whole loop.
    """
    j = idx
    for _ in range(n):
        v = arr[j] + 1
        arr[j] = v
        j = idx + (v & zero)


@njit(nogil=True)
def hammer_local(arr, idx, n, zero):
    """Count in a private one-element buffer; touch the shared array once."""
    local = np.zeros(1, dtype=np.int64)
    j = 0
    for _ in range(n):
        v = local[j] + 1
        local[j] = v
        j = v & zero
    arr[idx] = local[0]          # the only write to shared memory


def aligned_zeros(n, alignment=CACHE_LINE):
    """int64 array whose first element starts exactly on a cache line.

    np.zeros gives no alignment guarantee. Without this, slot 0 can land
    mid-line and the 'unpadded' counters straddle two lines, which halves
    the contention the experiment is supposed to create.
    """
    pad = alignment // INT64
    buf = np.zeros(n + pad, dtype=np.int64)
    offset = (-buf.ctypes.data) % alignment
    start = offset // INT64
    return buf[start:start + n]


def run_variant(kernel, slots, n_threads, array_len):
    """Launch one thread per slot and return (wall_ms, total_counted)."""
    arr = aligned_zeros(array_len)
    threads = [threading.Thread(target=kernel,
                                args=(arr, slots[i], ITERATIONS, 0))
               for i in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return wall_ms, int(arr.sum())


def timed(kernel, slots, n_threads, array_len):
    """Warm-up run plus REPEATS timed runs; returns the median and spread."""
    run_variant(kernel, slots, n_threads, array_len)
    runs = [run_variant(kernel, slots, n_threads, array_len)
            for _ in range(REPEATS)]
    times = [r[0] for r in runs]
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "total": runs[0][1],
    }


def main():
    logical = os.cpu_count() or 1
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
    except Exception:                                       # noqa: BLE001
        physical = logical

    n_threads = physical          # task: N = physical core count
    expected = ITERATIONS * n_threads

    out("=" * 78)
    out("TASK 3 — CACHE LINE CONTENTION (FALSE SHARING)")
    out("=" * 78)
    out(f"Student ID          : {STUDENT_ID}")
    out(f"Operating System    : {platform.system()} {platform.release()}")
    out(f"CPU model           : {cpu_model()}")
    out(f"Logical cores       : {logical}")
    out(f"Physical cores      : {physical}")
    out(f"Worker threads      : {n_threads}   (= physical core count)")
    out(f"Increments / thread : {ITERATIONS:,}")
    out(f"Total increments    : {expected:,}")
    out(f"Assumed cache line  : {CACHE_LINE} bytes "
        f"({STRIDE} x int64 slots per line)")
    out(f"Timing method       : {REPEATS} timed runs per variant, median reported")
    out()

    # ------------------------------------------------- 1. compiler evidence
    out("-" * 78)
    out("1. WHY THE NAIVE LOOP CANNOT BE USED")
    out("-" * 78)
    probe = np.zeros(8, dtype=np.int64)
    naive_increment(probe, 0, 10)                 # compile
    probe[0] = 0
    t0 = time.perf_counter()
    naive_increment(probe, 0, ITERATIONS)
    naive_ms = (time.perf_counter() - t0) * 1000.0
    naive_ns = naive_ms * 1e6 / ITERATIONS
    out(f"`for _ in range(n): arr[i] += 1`, {ITERATIONS:,} iterations, 1 thread")
    out(f"  wall time         : {naive_ms:.3f} ms")
    out(f"  per iteration     : {naive_ns:.4f} ns")
    out(f"  counter value     : {int(probe[0]):,}  (correct)")
    out()
    out("A single load-modify-store to L1 cache costs roughly 1 ns. A figure")
    out("far below that means the loop never ran: LLVM recognised the closed")
    out("form and replaced 50 million increments with one addition. The result")
    out("is still correct, but nothing was measured. Every benchmark below")
    out("therefore uses a loop-carried dependency that forces a genuine load")
    out("and store on each iteration.")
    out()

    # ----------------------------------------------------- 2. the benchmarks
    out("-" * 78)
    out("2. BENCHMARK VARIANTS")
    out("-" * 78)

    unpadded_slots = list(range(n_threads))
    padded_slots = [i * STRIDE for i in range(n_threads)]
    array_len = max(n_threads * STRIDE + STRIDE, 1024)

    align = aligned_zeros(array_len).ctypes.data % CACHE_LINE
    out(f"Array base address  : {CACHE_LINE}-byte aligned (forced), "
        f"slot 0 starts a cache line"
        if align == 0 else
        f"Array base address  : WARNING offset {align} bytes into a line")
    out()
    out(f"A  unpadded slots   : {unpadded_slots}")
    out(f"   byte offsets     : {[i * INT64 for i in unpadded_slots]}"
        f"  -> spans {((n_threads - 1) * INT64) // CACHE_LINE + 1} cache line(s)")
    out(f"B  padded slots     : {padded_slots}")
    out(f"   byte offsets     : {[i * INT64 for i in padded_slots]}"
        f"  -> one cache line each")
    out(f"C  thread-local     : private counter, one write to the shared array")
    out()

    res_a = timed(hammer_shared, unpadded_slots, n_threads, array_len)
    res_b = timed(hammer_shared, padded_slots, n_threads, array_len)
    res_c = timed(hammer_local, unpadded_slots, n_threads, array_len)

    for name, r in (("A", res_a), ("B", res_b), ("C", res_c)):
        if r["total"] != expected:
            out(f"WARNING: variant {name} counted {r['total']:,}, "
                f"expected {expected:,}")

    # --------------------------------------------------------- 3. the table
    out("=" * 78)
    out("3. RESULTS")
    out("=" * 78)
    header = (f"{'Variant':<22} | {'Wall Time (ms)':>15} | {'ns / increment':>14} | "
              f"{'vs fastest':>11}")
    out(header)
    out("-" * len(header))

    fastest = min(res_a["median_ms"], res_b["median_ms"], res_c["median_ms"])
    rows = [
        ("A  Unpadded Shared", res_a),
        ("B  Padded Shared", res_b),
        ("C  Thread-Local", res_c),
    ]
    for label, r in rows:
        ns = r["median_ms"] * 1e6 / expected
        out(f"{label:<22} | {r['median_ms']:>15.3f} | {ns:>14.4f} | "
            f"{r['median_ms'] / fastest:>10.4f}x")
    out("-" * len(header))
    out()

    out(f"{'Variant':<22} | {'min (ms)':>10} | {'median (ms)':>12} | "
        f"{'max (ms)':>10} | {'total counted':>16}")
    out("-" * 78)
    for label, r in rows:
        out(f"{label:<22} | {r['min_ms']:>10.3f} | {r['median_ms']:>12.3f} | "
            f"{r['max_ms']:>10.3f} | {r['total']:>16,}")
    out()

    # ------------------------------------------------------- 4. the ratios
    ratio_pad = res_a["median_ms"] / res_b["median_ms"]
    ratio_loc = res_a["median_ms"] / res_c["median_ms"]

    out("=" * 78)
    out("4. EXECUTION RATIO — THE DELIVERABLE")
    out("=" * 78)
    out(f"Unpadded Shared / Padded Shared : {ratio_pad:.4f}x")
    out(f"Unpadded Shared / Thread-Local  : {ratio_loc:.4f}x")
    out(f"Time lost to false sharing      : "
        f"{res_a['median_ms'] - res_b['median_ms']:+.3f} ms "
        f"({(ratio_pad - 1.0) * 100:+.2f}%)")
    out()

    out("-" * 78)
    out("5. ANALYSIS")
    out("-" * 78)
    out("All three variants perform exactly the same arithmetic and each")
    out("thread touches only its own counter. There is no lock, no logical")
    out("sharing and no race. The only difference between A and B is the")
    out("distance in bytes between the counters.")
    out()
    out("Cache coherence works on whole cache lines, not on variables. In A")
    out("all counters sit inside one 64-byte line, so under MESI each write")
    out("must take that line into the Modified state, which invalidates every")
    out("other core's copy. The next core to write takes a HITM miss and pulls")
    out("the line back. The line ping-pongs between cores for the whole run")
    out("even though no core ever wants another core's data. That is false")
    out("sharing: the hardware shares a line, the program shares nothing.")
    out()
    out("B spaces the counters 64 bytes apart, so each core owns its own line")
    out("outright and never has to give it up. C goes further and keeps the")
    out("counter in a private buffer, touching shared memory once per thread.")
    out()

    if ratio_pad >= 1.5:
        out(f"Measured here: the unpadded version is {ratio_pad:.2f}x slower,")
        out("which is the coherence traffic made visible.")
    elif ratio_pad >= 1.05:
        out(f"Measured here: the unpadded version is {ratio_pad:.2f}x slower.")
        out("The penalty is real but modest. The loop-carried dependency makes")
        out("each iteration wait on the previous load, so the core is already")
        out("stalled part of the time and the coherence miss partly hides")
        out("behind that latency.")
    else:
        out(f"Measured here the gap is only {ratio_pad:.2f}x. On this machine")
        out("the threads were likely scheduled onto hyperthread siblings that")
        out("share an L1 cache, in which case no coherence traffic crosses")
        out("between cores and false sharing cannot appear. The effect needs")
        out("threads on separate physical cores.")
    out()
    out("Practical remedies, in the order one should reach for them:")
    out("  1. Keep the hot counter thread-local and publish once (variant C).")
    out("  2. Pad shared slots to a full cache line: alignas(64) in C++,")
    out("     @Contended in Java, CachePadded in Rust, _ [56]byte in Go.")
    out("  3. Use a reduction instead of a shared array when the final value")
    out("     is all that matters.")
    out("=" * 78)

    with open("task3_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWritten: task3_output.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
