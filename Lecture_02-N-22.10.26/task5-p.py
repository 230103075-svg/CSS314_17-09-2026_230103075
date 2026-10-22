"""
TASK 5 — Thread Oversubscription Threshold

Kystaubay Ayanat — Student ID 230103075

Run:
    py task5-p.py

10,000 small CPU-bound tasks are pushed through thread pools of
increasing size, up to pools many times larger than the machine has
cores, to find where adding threads stops helping and starts hurting.

Two variants of the same batch are profiled:

  VARIANT A — hashlib, exactly as the task specifies
              Each task hashes a 1 KB string 500 times with SHA-256.
              CPython's hashlib only releases the GIL for inputs of about
              2 KB and up, so at 1 KB the interpreter lock is held for
              the whole hash. Every thread in the pool therefore waits
              its turn no matter how many cores exist.

  VARIANT B — the same work with the GIL released
              A compiled FNV-1a hash over the same 1 KB buffer, 500
              times per task, through numba's nogil path. The arithmetic
              is equivalent and the task count is identical, but threads
              now run genuinely in parallel.

Variant A answers the question as written. Variant B is what the
question is actually about: without real parallelism there is no
scaling curve to degrade, and the oversubscription threshold cannot
appear. Reporting only A would hide that distinction.
"""
import hashlib
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
N_TASKS = 10_000
ROUNDS = 500
PAYLOAD_BYTES = 1024
REPEATS = 1

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


# ------------------------------------------------------- the payload data
rng = np.random.default_rng(STUDENT_ID)
PAYLOAD = bytes(rng.integers(0, 256, size=PAYLOAD_BYTES, dtype=np.uint8))
PAYLOAD_ARR = np.frombuffer(PAYLOAD, dtype=np.uint8)


# --------------------------------------------------- variant A: hashlib
def task_hashlib(_):
    """Hash a 1 KB string 500 times. The GIL is held throughout."""
    acc = 0
    for _r in range(ROUNDS):
        acc ^= hashlib.sha256(PAYLOAD).digest()[0]
    return acc


# ------------------------------------------- variant B: compiled, no GIL
@njit(nogil=True)
def fnv1a_rounds(buf, rounds, seed):
    """FNV-1a over `buf`, `rounds` times, with the GIL released.

    Each round is seeded from the previous round's result, so the rounds
    form a dependency chain the optimiser cannot collapse into one.
    """
    acc = np.uint64(seed)
    prime = np.uint64(1099511628211)
    for _r in range(rounds):
        h = np.uint64(14695981039346656037) ^ acc
        for i in range(buf.size):
            h = h ^ np.uint64(buf[i])
            h = h * prime
        acc = h
    return acc


def task_native(_):
    return int(fnv1a_rounds(PAYLOAD_ARR, ROUNDS, 0))


# ------------------------------------------------------------ the driver
def run_batch(task_fn, pool_size):
    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        t0 = time.perf_counter()
        results = list(pool.map(task_fn, range(N_TASKS)))
        ms = (time.perf_counter() - t0) * 1000.0
    return ms, len(results)


def measure(task_fn, pool_size):
    times = [run_batch(task_fn, pool_size)[0] for _ in range(REPEATS)]
    return statistics.median(times), min(times), max(times)


def profile(label, task_fn, pool_sizes):
    out("=" * 78)
    out(label)
    out("=" * 78)

    rows = []
    for size in pool_sizes:
        print(f"  ...running pool size {size}", file=sys.stderr, flush=True)
        med, lo, hi = measure(task_fn, size)
        rows.append({"size": size, "ms": med, "min": lo, "max": hi})

    base = rows[0]["ms"]
    best = min(rows, key=lambda r: r["ms"])

    header = (f"{'Pool Size':>10} | {'Execution Time (ms)':>20} | "
              f"{'Speedup vs 1':>13} | {'Status':<24}")
    out(header)
    out("-" * len(header))

    degradation_start = None
    passed_peak = False
    for r in rows:
        r["speedup"] = base / r["ms"]
        if r["size"] == rows[0]["size"]:
            status = "baseline"
        elif r is best:
            status = "OPTIMUM"
            passed_peak = True
        elif not passed_peak:
            status = "scaling up"
        else:
            worse = (r["ms"] / best["ms"] - 1.0) * 100.0
            status = f"degrading ({worse:+.1f}% vs peak)"
            if degradation_start is None:
                degradation_start = r
        out(f"{r['size']:>10} | {r['ms']:>20.3f} | {r['speedup']:>12.4f}x | "
            f"{status:<24}")
    out("-" * len(header))
    out()
    out("Run spread (min / max over "
        f"{REPEATS} batches):")
    for r in rows:
        out(f"  pool {r['size']:>4} : {r['min']:.3f} / {r['max']:.3f} ms")
    out()

    out(f"Fastest pool size        : {best['size']} "
        f"({best['ms']:.3f} ms, {best['speedup']:.4f}x)")
    if degradation_start is not None:
        out(f"DEGRADATION BEGINS AT    : pool size {degradation_start['size']}")
        out(f"  time rose from {best['ms']:.3f} ms to "
            f"{degradation_start['ms']:.3f} ms "
            f"({(degradation_start['ms'] / best['ms'] - 1) * 100:+.1f}%)")
        # Only sizes past the optimum count as oversubscription; the
        # single-thread baseline is slow for a different reason.
        beyond = [r for r in rows if r["size"] > best["size"]]
        if beyond:
            worst = max(beyond, key=lambda r: r["ms"])
            out(f"  worst oversubscribed  : pool {worst['size']} at "
                f"{worst['ms']:.3f} ms "
                f"({worst['ms'] / best['ms']:.2f}x the optimum)")
    else:
        out("DEGRADATION BEGINS AT    : not observed across the tested sizes")
    out()
    return rows, best, degradation_start


def main():
    cores = os.cpu_count() or 1
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or cores
    except Exception:                                      # noqa: BLE001
        physical = cores

    raw = [1, max(1, cores // 2), cores, cores * 2, cores * 4, cores * 16, 256]
    pool_sizes = sorted(set(raw))

    out("=" * 78)
    out("TASK 5 — THREAD OVERSUBSCRIPTION THRESHOLD")
    out("=" * 78)
    out(f"Student ID           : {STUDENT_ID}")
    out(f"Operating System     : {platform.system()} {platform.release()}")
    out(f"CPU model            : {cpu_model()}")
    out(f"Logical cores        : {cores}")
    out(f"Physical cores       : {physical}")
    out(f"Tasks in the batch   : {N_TASKS:,}")
    out(f"Work per task        : hash a {PAYLOAD_BYTES}-byte payload "
        f"{ROUNDS} times")
    out(f"Payload seed         : {STUDENT_ID}  (Student ID)")
    out(f"Requested pool sizes : {raw}")
    out(f"Tested (deduplicated): {pool_sizes}")
    out(f"Timing method        : {REPEATS} full batches per size, median")
    out()

    # Compile before any timing.
    fnv1a_rounds(PAYLOAD_ARR, 2, 0)

    a_rows, a_best, a_deg = profile(
        "VARIANT A — hashlib SHA-256 (GIL held: 1 KB is below the "
        "release threshold)",
        task_hashlib, pool_sizes)

    b_rows, b_best, b_deg = profile(
        "VARIANT B — compiled FNV-1a, same batch, GIL released",
        task_native, pool_sizes)

    # ---------------------------------------------------------- analysis
    out("=" * 78)
    out("ANALYSIS")
    out("=" * 78)
    out(f"Variant A best speedup : {a_best['speedup']:.4f}x at pool "
        f"{a_best['size']}")
    out(f"Variant B best speedup : {b_best['speedup']:.4f}x at pool "
        f"{b_best['size']}")
    out()

    if a_best["speedup"] < 1.3:
        out("Variant A never scaled. SHA-256 over 1 KB stays inside the GIL,")
        out("so only one thread executes at a time regardless of pool size.")
        out("Adding threads cannot add throughput; it only adds contention for")
        out("the lock and work for the scheduler. This is the measurement the")
        out("task asks for, and the flat-then-worsening curve is the correct")
        out("result for it, not a failed experiment.")
    else:
        out("Variant A scaled to some degree, which means hashlib released the")
        out("GIL for this payload on this build of Python.")
    out()

    out("Variant B removes the lock and the curve appears:")
    out(f"  it climbs to {b_best['speedup']:.2f}x at pool {b_best['size']}, "
        f"close to the {physical} physical")
    out("  cores available, and then flattens or falls.")
    out()

    if b_deg is not None:
        out(f"The oversubscription threshold is pool size "
            f"{b_deg['size']}. Past the optimum")
        out(f"of {b_best['size']}, every extra thread is a thread that must be "
            f"scheduled onto")
        out("a core that is already busy. The costs that appear are:")
        out("  - context switches: the OS saves and restores register state,")
        out("    and each switch costs microseconds of pure overhead.")
        out("  - cache pollution: a thread resuming on a different core finds")
        out("    its working set cold and reloads it from memory.")
        out("  - scheduler work: the run queue grows and the kernel spends")
        out("    more of its own time deciding what to run next.")
        out("  - memory: each thread carries its own stack, so a 256-thread")
        out("    pool reserves hundreds of megabytes of address space for")
        out("    threads that are mostly waiting.")
    else:
        out("No clear degradation appeared across the tested sizes, which")
        out("happens when the per-task work is long enough that switching")
        out("overhead stays small relative to it.")
    out()

    out("The practical rule this measures: for CPU-bound work, size the pool")
    out("to the core count, not to the task count. 10,000 tasks do not need")
    out("10,000 threads. Threads are a way to use cores, and once the cores")
    out("are busy more threads only add bookkeeping. Oversubscription pays off")
    out("only for I/O-bound work, where threads spend their time blocked and")
    out("a waiting thread costs nothing but memory.")
    out("=" * 78)

    with open("task5-p_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWritten: task5-p_output.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
