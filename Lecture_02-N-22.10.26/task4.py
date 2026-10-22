"""
TASK 4 — Work Stealing & Scheduling Under Skew

Kystaubay Ayanat — Student ID 230103075

Run:
    py -m pip install numba
    py task4.py

1,000 items. Item i costs 0.0001 * (i % 20) seconds, so the cost sweeps
from 0 ms up to 1.9 ms and repeats every 20 items. Four workers run the
loop under two scheduling strategies:

  A  STATIC   — the range is cut into 4 equal contiguous blocks, one per
                worker, decided before any work starts.
  B  DYNAMIC  — workers pull the next chunk from a shared queue whenever
                they go idle. Tested at chunk = 1 and chunk = 5.

Reported per strategy: total wall time, each worker's finish time, and the
completion delta between the first and last worker to finish.

Why the cost is burned on the CPU instead of time.sleep
-------------------------------------------------------
The task allows a "mathematical equivalent", and here it is required.
Windows timer granularity is about 1 ms, which is ten times the smallest
unit of work in this workload (0.1 ms). Every short item would be rounded
up to the same floor, flattening the very skew the exercise is about, and
sleeping workers would not compete for a core at all. A calibrated
CPU-bound spin reproduces the intended cost ratio exactly and makes the
workers contend for real cores, which is what scheduling has to solve.
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
N_ITEMS = 1000
N_WORKERS = 4
UNIT_SECONDS = 0.0001          # cost(i) = UNIT_SECONDS * (i % 20)
MODULO = 20
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


@njit(nogil=True)
def spin(iterations):
    """Burn `iterations` of dependent float work, with the GIL released.

    The multiply-add chain is sequentially dependent and floating point is
    not associative, so the optimiser cannot collapse or vectorise it.
    """
    x = 1.0
    acc = 0.0
    for _ in range(iterations):
        x = x * 1.0000001 + 1e-9
        acc += x
    return acc


def calibrate():
    """Measure spin throughput so a cost in seconds maps to an iteration count."""
    spin(1000)                                   # compile
    samples = []
    probe = 2_000_000
    for _ in range(5):
        t0 = time.perf_counter()
        spin(probe)
        samples.append(probe / (time.perf_counter() - t0))
    return statistics.median(samples)            # iterations per second


def item_cost_seconds(i):
    return UNIT_SECONDS * (i % MODULO)


# ------------------------------------------------------------- strategies
def run_static(items, iters_for, n_workers):
    """Equal contiguous blocks, assigned before the run begins."""
    base, extra = divmod(len(items), n_workers)
    blocks, start = [], 0
    for w in range(n_workers):
        stop = start + base + (1 if w < extra else 0)
        blocks.append(items[start:stop])
        start = stop

    finish = [0.0] * n_workers
    counts = [0] * n_workers

    def worker(wid):
        for i in blocks[wid]:
            spin(iters_for[i])
        counts[wid] = len(blocks[wid])
        finish[wid] = time.perf_counter() - t0

    threads = [threading.Thread(target=worker, args=(w,))
               for w in range(n_workers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0
    return total, finish, counts


def run_dynamic(items, iters_for, n_workers, chunk):
    """Shared queue: an idle worker claims the next chunk of indices."""
    cursor = [0]
    lock = threading.Lock()
    finish = [0.0] * n_workers
    counts = [0] * n_workers

    def worker(wid):
        done = 0
        while True:
            with lock:
                start = cursor[0]
                cursor[0] += chunk
            if start >= len(items):
                break
            for i in items[start:start + chunk]:
                spin(iters_for[i])
                done += 1
        counts[wid] = done
        finish[wid] = time.perf_counter() - t0

    threads = [threading.Thread(target=worker, args=(w,))
               for w in range(n_workers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0
    return total, finish, counts


def measure(fn, *args):
    """Warm-up plus REPEATS timed runs; the run with the median total is kept."""
    fn(*args)
    runs = [fn(*args) for _ in range(REPEATS)]
    runs.sort(key=lambda r: r[0])
    return runs[len(runs) // 2]


def report(label, total, finish, counts):
    delta = max(finish) - min(finish)
    out(f"{label}")
    out(f"  total wall time   : {total * 1000:.3f} ms")
    out(f"  worker finish (ms): " +
        "  ".join(f"W{w}={f * 1000:.2f}" for w, f in enumerate(finish)))
    out(f"  items per worker  : " +
        "  ".join(f"W{w}={c}" for w, c in enumerate(counts)))
    out(f"  first finisher    : {min(finish) * 1000:.3f} ms")
    out(f"  last finisher     : {max(finish) * 1000:.3f} ms")
    out(f"  COMPLETION DELTA  : {delta * 1000:.3f} ms")
    out()
    return delta


def main():
    items = list(range(N_ITEMS))
    costs = [item_cost_seconds(i) for i in items]
    total_work = sum(costs)

    out("=" * 78)
    out("TASK 4 — WORK STEALING & SCHEDULING UNDER SKEW")
    out("=" * 78)
    out(f"Student ID          : {STUDENT_ID}")
    out(f"Operating System    : {platform.system()} {platform.release()}")
    out(f"CPU model           : {cpu_model()}")
    out(f"Logical cores       : {os.cpu_count()}")
    out(f"Workers             : {N_WORKERS}")
    out(f"Items               : {N_ITEMS}  (indices 0..{N_ITEMS - 1})")
    out(f"Cost model          : {UNIT_SECONDS} * (i % {MODULO}) seconds")
    out(f"Cost range          : {min(costs) * 1000:.2f} ms .. "
        f"{max(costs) * 1000:.2f} ms  (ratio "
        f"{max(costs) / min(c for c in costs if c > 0):.0f}:1)")
    out(f"Total serial work   : {total_work * 1000:.3f} ms")
    out(f"Perfect split / {N_WORKERS}  : {total_work / N_WORKERS * 1000:.3f} ms "
        f"(the theoretical floor)")
    out(f"Timing method       : {REPEATS} timed runs per strategy, median kept")
    out()

    out("-" * 78)
    out("CALIBRATION")
    out("-" * 78)
    rate = calibrate()
    out(f"Spin throughput     : {rate / 1e6:.2f} M iterations/sec "
        f"({1e9 / rate:.3f} ns per iteration)")
    iters_for = [max(1, int(round(c * rate))) for c in costs]
    out(f"Iterations for 0.1ms: {int(round(UNIT_SECONDS * rate)):,}")
    out(f"Iterations for 1.9ms: {int(round(19 * UNIT_SECONDS * rate)):,}")

    # Prove the calibration holds before trusting any timing below.
    check_target = 10 * UNIT_SECONDS
    t0 = time.perf_counter()
    spin(int(round(check_target * rate)))
    check_actual = time.perf_counter() - t0
    out(f"Calibration check   : asked {check_target * 1000:.3f} ms, "
        f"got {check_actual * 1000:.3f} ms "
        f"({(check_actual / check_target - 1) * 100:+.2f}%)")
    out()

    # ---------------------------------------------- how static splits the work
    base = N_ITEMS // N_WORKERS
    out("-" * 78)
    out("PREDICTED STATIC IMBALANCE (from the cost model alone)")
    out("-" * 78)
    block_work = []
    for w in range(N_WORKERS):
        lo, hi = w * base, (w + 1) * base
        block_work.append(sum(costs[lo:hi]))
        out(f"  W{w}: items {lo:>4}..{hi - 1:<4} -> "
            f"{block_work[-1] * 1000:8.3f} ms of work")
    predicted_spread = (max(block_work) - min(block_work)) * 1000
    out(f"  spread between the heaviest and lightest block: "
        f"{predicted_spread:.3f} ms")
    out("  This is the imbalance the cost model alone predicts for static")
    out("  scheduling, before the operating system has any say.")
    out()

    # ------------------------------------------------------------ strategy A
    out("=" * 78)
    out("STRATEGY A — STATIC CHUNK DISTRIBUTION")
    out("=" * 78)
    a_total, a_finish, a_counts = measure(run_static, items, iters_for, N_WORKERS)
    a_delta = report("Static, 4 equal contiguous blocks", a_total, a_finish, a_counts)

    # ------------------------------------------------------------ strategy B
    out("=" * 78)
    out("STRATEGY B — DYNAMIC / WORK-STEALING QUEUE")
    out("=" * 78)
    b1_total, b1_finish, b1_counts = measure(run_dynamic, items, iters_for,
                                             N_WORKERS, 1)
    b1_delta = report("Dynamic, chunk = 1", b1_total, b1_finish, b1_counts)

    b5_total, b5_finish, b5_counts = measure(run_dynamic, items, iters_for,
                                             N_WORKERS, 5)
    b5_delta = report("Dynamic, chunk = 5", b5_total, b5_finish, b5_counts)

    # ---------------------------------------------------------- the summary
    floor_ms = total_work / N_WORKERS * 1000
    rows = [
        ("A  Static blocks", a_total, a_delta),
        ("B  Dynamic chunk=1", b1_total, b1_delta),
        ("B  Dynamic chunk=5", b5_total, b5_delta),
    ]

    out("=" * 78)
    out("SUMMARY")
    out("=" * 78)
    header = (f"{'Strategy':<20} | {'Total (ms)':>11} | {'Delta (ms)':>11} | "
              f"{'vs Static':>10} | {'over floor':>11}")
    out(header)
    out("-" * len(header))
    for label, total, delta in rows:
        out(f"{label:<20} | {total * 1000:>11.3f} | {delta * 1000:>11.3f} | "
            f"{a_total / total:>9.4f}x | "
            f"{(total * 1000 / floor_ms - 1) * 100:>10.2f}%")
    out("-" * len(header))
    out(f"Theoretical floor (total work / {N_WORKERS}) = {floor_ms:.3f} ms")
    out()

    best_dyn_delta = min(b1_delta, b5_delta)
    out("=" * 78)
    out("ANALYSIS")
    out("=" * 78)
    out("Completion delta is the honest measure here. It is the time cores")
    out("spend idle at the barrier waiting for the slowest worker, and it is")
    out("exactly what a scheduler is supposed to drive to zero.")
    out()
    out(f"  Static  delta : {a_delta * 1000:.3f} ms")
    out(f"  Dynamic delta : {best_dyn_delta * 1000:.3f} ms (best of chunk 1 and 5)")
    if a_delta > 0:
        out(f"  Reduction     : {(1 - best_dyn_delta / a_delta) * 100:.1f}%")
    out()
    cores = os.cpu_count() or 1
    out(f"Predicted static delta from the cost model : {predicted_spread:.3f} ms")
    out(f"Measured static delta                      : {a_delta * 1000:.3f} ms")
    if a_delta * 1000 > predicted_spread * 3:
        out()
        out("The measured delta is far larger than the cost model predicts.")
        if N_WORKERS > cores:
            out(f"This machine reports {cores} logical core(s) but the task asks for")
            out(f"{N_WORKERS} workers, so the workers are oversubscribed and have to")
            out("time-share cores. The finish spread then reflects how the OS")
            out("scheduler interleaved them, not how the work was divided. On a")
            out(f"machine with at least {N_WORKERS} free cores the measured delta")
            out("lands close to the predicted one.")
        else:
            out("The cores were not exclusively available: other load on the")
            out("machine, frequency scaling, or migration between cores adds")
            out("spread the cost model does not account for.")
    else:
        out("The two agree closely, which confirms the measured imbalance comes")
        out("from how the work was divided and not from OS scheduling noise.")
    out()
    out("Static scheduling fixes the assignment before the first item runs, so")
    out("it can only be as good as its guess about cost. Here the cost repeats")
    out("every 20 items and each block of 250 covers 12 and a half full cycles,")
    out("so the blocks come out nearly, but not exactly, equal. Whatever")
    out("imbalance remains is paid in full: a worker that finishes early cannot")
    out("help, it just waits.")
    out()
    out("Dynamic scheduling makes no guess. A worker that finishes early comes")
    out("back for more, so the imbalance can only ever be as large as the last")
    out("chunk still in flight. That is why chunk = 1 gives the tightest finish")
    out("spread: the worst case is one item, 1.9 ms. Chunk = 5 claims five items")
    out("per lock acquisition, trading a slightly looser spread for fewer trips")
    out("through the shared queue.")
    out()
    out("The cost of dynamic scheduling is that shared queue. Every chunk claim")
    out("takes a lock, so chunk = 1 pays 1,000 lock acquisitions against chunk")
    out("= 5's 200. When the items are cheap relative to that overhead, dynamic")
    out("can lose on total time even while winning on the delta, which is the")
    out("trade-off behind OpenMP's schedule(static) / schedule(dynamic, n) /")
    out("schedule(guided) choice.")
    out("=" * 78)

    with open("task4_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWritten: task4_output.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
