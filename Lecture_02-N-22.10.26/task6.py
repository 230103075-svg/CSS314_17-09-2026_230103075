"""
TASK 6 — Synchronization Cost & Data Loss

Kystaubay Ayanat — Student ID 230103075

Run:
    py -m pip install numba
    py task6.py

A shared counter is incremented 2,000,000 times by each of 4 threads.
Expected final value: 8,000,000.

Five runs, each isolating one thing:

  A  UNSAFE, default Python   — no lock, interpreter left alone
  B  UNSAFE, forced preemption — no lock, GIL switch interval minimised
  C  UNSAFE, no GIL at all     — no lock, compiled kernel, real hardware race
  D  LOCKED                    — threading.Lock around every increment
  E  LOCK-FREE REDUCTION       — private counters, one merge at the end

Why A and B are both needed
---------------------------
`counter.value += 1` is three bytecodes: LOAD_ATTR, BINARY_OP, STORE_ATTR.
A thread switch between the load and the store loses an update. CPython
only considers switching every `sys.getswitchinterval()` seconds, so with
the default 5 ms the window is rarely hit and the damage looks negligible.
That is the GIL hiding the bug, not the bug being absent. B shrinks the
interval to make the same broken code fail visibly, and C removes the GIL
entirely to show what the hardware does when nothing is protecting it.

The timing penalty in section 5 compares D against A, both at the default
switch interval, so the comparison measures the lock and nothing else.
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
N_THREADS = 4
PER_THREAD = 2_000_000
EXPECTED = N_THREADS * PER_THREAD          # 8,000,000
TIMING_REPEATS = 3

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


class Counter:
    """A plain shared counter. No protection of any kind."""
    __slots__ = ("value",)

    def __init__(self):
        self.value = 0


# ------------------------------------------------------- Python-level runs
def run_unsafe():
    c = Counter()

    def worker():
        for _ in range(PER_THREAD):
            c.value += 1                   # LOAD_ATTR / BINARY_OP / STORE_ATTR

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return c.value, (time.perf_counter() - t0) * 1000.0


def run_locked():
    c = Counter()
    lock = threading.Lock()

    def worker():
        for _ in range(PER_THREAD):
            with lock:
                c.value += 1

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return c.value, (time.perf_counter() - t0) * 1000.0


def run_reduction():
    """Each thread owns a private counter; one merge at the very end."""
    partials = [0] * N_THREADS

    def worker(wid):
        local = 0                          # thread-local, never shared
        for _ in range(PER_THREAD):
            local += 1
        partials[wid] = local              # one write, to its own slot

    threads = [threading.Thread(target=worker, args=(w,))
               for w in range(N_THREADS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sum(partials), (time.perf_counter() - t0) * 1000.0


# ----------------------------------------------------- compiled, GIL-free
@njit(nogil=True)
def unsafe_native(arr, n, zero):
    """Increment arr[0] n times with no GIL and no synchronisation.

    The next index is derived from the value just loaded, so the optimiser
    cannot keep the counter in a register: every iteration performs a real
    load and a real store, which is what makes the race observable.
    """
    j = 0
    for _ in range(n):
        v = arr[j] + 1
        arr[j] = v
        j = v & zero


def run_unsafe_native():
    arr = np.zeros(1, dtype=np.int64)
    threads = [threading.Thread(target=unsafe_native, args=(arr, PER_THREAD, 0))
               for _ in range(N_THREADS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return int(arr[0]), (time.perf_counter() - t0) * 1000.0


def report(label, actual, ms):
    lost = EXPECTED - actual
    pct = lost / EXPECTED * 100.0
    out(f"{label}")
    out(f"  expected final value : {EXPECTED:,}")
    out(f"  actual final value   : {actual:,}")
    out(f"  updates lost         : {lost:,}")
    out(f"  percentage lost      : {pct:.6f}%")
    out(f"  wall-clock time      : {ms:.3f} ms")
    out()
    return lost, pct, ms


def median_time(fn):
    """Median wall time over TIMING_REPEATS runs; last result value kept."""
    times, value = [], None
    for _ in range(TIMING_REPEATS):
        value, ms = fn()
        times.append(ms)
    return value, statistics.median(times)


def main():
    out("=" * 78)
    out("TASK 6 — SYNCHRONIZATION COST & DATA LOSS")
    out("=" * 78)
    out(f"Student ID          : {STUDENT_ID}")
    out(f"Operating System    : {platform.system()} {platform.release()}")
    out(f"CPU model           : {cpu_model()}")
    out(f"Logical cores       : {os.cpu_count()}")
    out(f"Threads             : {N_THREADS}")
    out(f"Increments / thread : {PER_THREAD:,}")
    out(f"EXPECTED FINAL VALUE: {EXPECTED:,}")
    out(f"Python              : {platform.python_version()}")
    out(f"GIL switch interval : {sys.getswitchinterval() * 1000:.3f} ms (default)")
    out()

    # ------------------------------------------------------------------- A
    out("=" * 78)
    out("A. UNSAFE — no lock, default interpreter settings")
    out("=" * 78)
    a_val, a_ms = run_unsafe()
    a_lost, a_pct, _ = report("4 threads, counter.value += 1, no protection",
                              a_val, a_ms)

    # ------------------------------------------------------------------- B
    out("=" * 78)
    out("B. UNSAFE — same code, GIL forced to switch aggressively")
    out("=" * 78)
    original = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    out(f"sys.setswitchinterval({1e-6}) — the interpreter now looks for a")
    out("thread switch roughly 5000x more often than by default.")
    out()
    b_val, b_ms = run_unsafe()
    sys.setswitchinterval(original)
    b_lost, b_pct, _ = report("4 threads, same unprotected increment",
                              b_val, b_ms)

    # ------------------------------------------------------------------- C
    out("=" * 78)
    out("C. UNSAFE — compiled kernel, GIL released, true hardware race")
    out("=" * 78)
    arr = np.zeros(1, dtype=np.int64)
    unsafe_native(arr, 10, 0)                       # compile before timing
    c_val, c_ms = run_unsafe_native()
    c_lost, c_pct, _ = report("4 threads, read-modify-write on shared memory",
                              c_val, c_ms)

    # ------------------------------------------------------------------- D
    out("=" * 78)
    out("D. CORRECT — threading.Lock around every increment")
    out("=" * 78)
    d_val, d_ms = median_time(run_locked)
    d_lost, d_pct, _ = report("4 threads, with lock: counter.value += 1",
                              d_val, d_ms)

    # ------------------------------------------------------------------- E
    out("=" * 78)
    out("E. CORRECT — lock-free: private counters, merged once")
    out("=" * 78)
    e_val, e_ms = median_time(run_reduction)
    e_lost, e_pct, _ = report("4 threads, thread-local accumulation + reduction",
                              e_val, e_ms)

    # ------------------------------------------------- the timing penalty
    a_val_t, a_ms_t = median_time(run_unsafe)

    out("=" * 78)
    out("SUMMARY")
    out("=" * 78)
    header = (f"{'Run':<34} | {'Final value':>13} | {'Lost %':>10} | "
              f"{'Time (ms)':>10}")
    out(header)
    out("-" * len(header))
    rows = [
        ("A  unsafe, default GIL", a_val, a_pct, a_ms),
        ("B  unsafe, forced preemption", b_val, b_pct, b_ms),
        ("C  unsafe, no GIL (native)", c_val, c_pct, c_ms),
        ("D  locked (threading.Lock)", d_val, d_pct, d_ms),
        ("E  lock-free reduction", e_val, e_pct, e_ms),
    ]
    for label, val, pct, ms in rows:
        out(f"{label:<34} | {val:>13,} | {pct:>9.4f}% | {ms:>10.3f}")
    out("-" * len(header))
    out()

    # The task asks for three specific numbers. State them outright, from
    # the run where the race is actually able to happen.
    if a_lost > 0:
        src, src_val, src_lost, src_pct = "A (unsafe, default GIL)", a_val, a_lost, a_pct
    elif b_lost > 0:
        src, src_val, src_lost, src_pct = "B (unsafe, forced preemption)", b_val, b_lost, b_pct
    else:
        src, src_val, src_lost, src_pct = "C (unsafe, no GIL)", c_val, c_lost, c_pct

    out("=" * 78)
    out("DELIVERABLE — DATA LOSS FROM THE UNSYNCHRONISED COUNTER")
    out("=" * 78)
    out(f"Measured on run {src}")
    out(f"  Expected final value       : {EXPECTED:,}")
    out(f"  Actual final value         : {src_val:,}")
    out(f"  Updates lost to races      : {src_lost:,}")
    out(f"  PERCENTAGE OF UPDATES LOST : {src_pct:.4f}%")
    if a_lost == 0 and b_lost == 0:
        out()
        out("  Runs A and B lost nothing: CPython's GIL serialises bytecode and")
        out("  masks the race entirely. Run C removes the GIL, which is the only")
        out("  configuration in which this hardware race can be observed from")
        out("  Python. The percentage above is therefore taken from run C.")
    out("=" * 78)
    out()

    out("=" * 78)
    out("SYNCHRONIZATION PENALTY")
    out("=" * 78)
    a_verdict = ("result correct (the GIL hid the race)"
                 if a_val_t == EXPECTED
                 else f"result WRONG by {EXPECTED - a_val_t:,}")
    out(f"Unsafe  (A, median of {TIMING_REPEATS}) : {a_ms_t:.3f} ms  "
        f"-> {a_verdict}")
    out(f"Locked  (D, median of {TIMING_REPEATS}) : {d_ms:.3f} ms  "
        f"-> result correct")
    out(f"Lock-free (E, median of {TIMING_REPEATS}) : {e_ms:.3f} ms  "
        f"-> result correct")
    out()
    penalty_ms = d_ms - a_ms_t
    penalty_x = d_ms / a_ms_t if a_ms_t > 0 else float("nan")
    out(f"TIME PENALTY OF THE LOCK : {penalty_ms:+.3f} ms "
        f"({penalty_x:.4f}x slower than unsafe)")
    out(f"Cost per increment       : {penalty_ms * 1e6 / EXPECTED:+.2f} ns")
    if e_ms > 0:
        out(f"Lock vs lock-free        : {d_ms / e_ms:.4f}x")
    out("=" * 78)
    out()

    out("-" * 78)
    out("ANALYSIS")
    out("-" * 78)
    out("The increment is not one operation. It is read the value, add one,")
    out("write it back. If a second thread reads between another thread's read")
    out("and its write, both compute the same new value and one increment")
    out("vanishes. Nothing crashes and no error is raised; the number is")
    out("simply wrong, which is what makes this class of bug dangerous.")
    out()
    if a_lost == 0 and b_lost == 0:
        out(f"Runs A and B both finished at exactly {EXPECTED:,}, even with the")
        out("switch interval cut by a factor of 5000. That is not because the")
        out("code is correct. It is because CPython's GIL lets only one thread")
        out(f"execute bytecode at a time, and on Python "
            f"{platform.python_version()} the interpreter")
        out("does not take a switch in the middle of this particular")
        out("read-modify-write. The bug is fully present in the source; the")
        out("interpreter is simply standing in front of it.")
        out()
        out("This is worth stating plainly because it is the trap: the same")
        out("logic ported to a language without a GIL, or run on a free-")
        out("threaded Python build, starts losing data immediately. Relying on")
        out("the GIL for correctness means the code is one runtime change away")
        out("from silently producing wrong numbers.")
    else:
        out(f"Run A lost {a_pct:.6f}% and run B lost {b_pct:.4f}% running the")
        out("exact same source code. The only difference is how often the")
        out("interpreter was willing to switch threads. A race condition is not")
        out("a property of the code alone: it is the code plus a timing window,")
        out("and that window depends on the scheduler, the machine and the")
        out("load. Code that passes on a laptop can lose data in production.")
    out()
    out(f"Run C lost {c_pct:.4f}%. With the GIL gone, four cores perform")
    out("genuine simultaneous read-modify-write on one memory location and")
    out("most of the work is overwritten. This is the unmasked hardware")
    out("behaviour that the GIL was concealing in A.")
    out()
    out("D fixes it with a mutex: one thread at a time holds the lock, so the")
    out("read-modify-write becomes indivisible and the count is exact. The")
    out("price is in the summary above. The lock serialises the hot path, so")
    out("the four threads no longer overlap on it at all.")
    out()
    out("E is the better fix where it applies. Each thread counts in a private")
    out("variable with no contention, and the four partial results are merged")
    out("once at the end. Correct like D, but without serialising anything.")
    out("This is what OpenMP's reduction(+:x) does, and it is why a reduction")
    out("should be preferred over a shared counter whenever the intermediate")
    out("value is not actually needed.")
    out("=" * 78)

    with open("task6_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWritten: task6_output.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
