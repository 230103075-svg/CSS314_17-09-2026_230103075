"""
TASK 6 — Atomic CAS vs. OS Mutex Under Heavy Contention

Kystaubay Ayanat — Student ID 230103075

Run:
    py task6-p.py

8 threads x 1,000,000 increments of one shared counter, two ways:

  Primitive 1  OS MUTEX     threading.Lock() around every increment.
  Primitive 2  ATOMIC CAS   a compare-and-swap retry loop built on the
                            CPU's cmpxchg instruction (LOCK CMPXCHG on
                            x86), emitted through numba's LLVM layer,
                            run with the GIL released.

Python has no atomic CAS in its standard library, so the CAS loop is
compiled. That means the two runs also differ in interpreted vs compiled
execution; section 3 measures both primitives single-threaded so that
part of the gap can be separated from the contention effect.
"""
import os
import platform
import sys
import threading
import time

import numpy as np

try:
    from numba import njit, types
    from numba.core import cgutils
    from numba.extending import intrinsic
except ImportError:
    print("numba is required:  py -m pip install numba", file=sys.stderr)
    raise SystemExit(1)

STUDENT_ID = 230103075
N_THREADS = 8
PER_THREAD = 1_000_000
EXPECTED = N_THREADS * PER_THREAD

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
    except Exception:                                      # noqa: BLE001
        pass
    return platform.processor() or platform.machine() or "unknown"


# --------------------------------------------- hardware CAS via LLVM cmpxchg
@intrinsic
def atomic_cas(typingctx, arr, idx, expected, new):
    """Atomically: if arr[idx] == expected, set it to new. Returns old value."""
    sig = types.int64(arr, idx, expected, new)

    def codegen(context, builder, signature, args):
        arr_t = signature.args[0]
        aryv, idxv, exp, nv = args
        ary = context.make_array(arr_t)(context, builder, aryv)
        ptr = cgutils.get_item_pointer(context, builder, arr_t, ary, [idxv])
        res = builder.cmpxchg(ptr, exp, nv, "seq_cst", "seq_cst")
        return builder.extract_value(res, 0)

    return sig, codegen


@njit(nogil=True)
def cas_increment(counter, n):
    """Increment counter[0] n times with a lock-free CAS retry loop."""
    retries = 0
    cur = counter[0]
    for _ in range(n):
        while True:
            prev = atomic_cas(counter, 0, cur, cur + 1)
            if prev == cur:          # our swap won
                cur = cur + 1
                break
            cur = prev               # another core got there first; retry
            retries += 1
    return retries


# ------------------------------------------------------------- the runs
class Counter:
    __slots__ = ("value",)

    def __init__(self):
        self.value = 0


def run_mutex(n_threads, per_thread):
    c = Counter()
    lock = threading.Lock()

    def worker():
        for _ in range(per_thread):
            with lock:
                c.value += 1

    ts = [threading.Thread(target=worker) for _ in range(n_threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return (time.perf_counter() - t0) * 1000.0, c.value


def run_cas(n_threads, per_thread):
    counter = np.zeros(1, dtype=np.int64)
    retries = [0] * n_threads

    def worker(w):
        retries[w] = cas_increment(counter, per_thread)

    ts = [threading.Thread(target=worker, args=(w,)) for w in range(n_threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return (time.perf_counter() - t0) * 1000.0, int(counter[0]), sum(retries)


def main():
    out("=" * 78)
    out("TASK 6 — ATOMIC CAS vs. OS MUTEX UNDER HEAVY CONTENTION")
    out("=" * 78)
    out(f"Student ID          : {STUDENT_ID}")
    out(f"Operating System    : {platform.system()} {platform.release()}")
    out(f"CPU model           : {cpu_model()}")
    out(f"Logical cores       : {os.cpu_count()}")
    out(f"Threads             : {N_THREADS}")
    out(f"Increments / thread : {PER_THREAD:,}")
    out(f"Expected final value: {EXPECTED:,}")
    out()

    cas_increment(np.zeros(1, dtype=np.int64), 10)       # compile first

    # ---------------------------------------------------- 1. OS mutex
    m_ms, m_val = run_mutex(N_THREADS, PER_THREAD)
    m_tput = EXPECTED / (m_ms / 1000.0)
    out("-" * 78)
    out("1. PRIMITIVE 1 — OS MUTEX (threading.Lock)")
    out("-" * 78)
    out(f"Final value         : {m_val:,}  "
        f"{'CORRECT' if m_val == EXPECTED else 'WRONG'}")
    out(f"Execution time      : {m_ms:.3f} ms")
    out(f"Throughput          : {m_tput:,.0f} ops/sec")
    out()

    # ---------------------------------------------------- 2. atomic CAS
    c_ms, c_val, c_retries = run_cas(N_THREADS, PER_THREAD)
    c_tput = EXPECTED / (c_ms / 1000.0)
    out("-" * 78)
    out("2. PRIMITIVE 2 — HARDWARE ATOMIC CAS (LOCK CMPXCHG loop)")
    out("-" * 78)
    out(f"Final value         : {c_val:,}  "
        f"{'CORRECT' if c_val == EXPECTED else 'WRONG'}")
    out(f"Execution time      : {c_ms:.3f} ms")
    out(f"Throughput          : {c_tput:,.0f} ops/sec")
    out(f"Failed CAS retries  : {c_retries:,} "
        f"({c_retries / EXPECTED * 100:.1f}% of successful ops)")
    out()

    # -------------------------------------------- 3. uncontended baseline
    m1_ms, _ = run_mutex(1, PER_THREAD)
    c1_ms, _, _ = run_cas(1, PER_THREAD)
    out("-" * 78)
    out("3. SAME PRIMITIVES, ONE THREAD, NO CONTENTION")
    out("-" * 78)
    out(f"Mutex, 1 thread     : {m1_ms:.3f} ms for {PER_THREAD:,} ops "
        f"({m1_ms * 1e6 / PER_THREAD:.1f} ns/op)")
    out(f"CAS,   1 thread     : {c1_ms:.3f} ms for {PER_THREAD:,} ops "
        f"({c1_ms * 1e6 / PER_THREAD:.1f} ns/op)")
    out(f"Uncontended ratio   : {m1_ms / c1_ms:.2f}x  "
        f"(interpreter + lock cost, before contention)")
    out()

    # --------------------------------------------------- the deliverable
    ratio = c_tput / m_tput
    out("=" * 78)
    out("DELIVERABLE")
    out("=" * 78)
    out(f"{'Primitive':<22} | {'Time (ms)':>12} | {'Throughput (ops/sec)':>22}")
    out("-" * 62)
    out(f"{'Mutex (OS lock)':<22} | {m_ms:>12.3f} | {m_tput:>22,.0f}")
    out(f"{'Atomic CAS':<22} | {c_ms:>12.3f} | {c_tput:>22,.0f}")
    out("-" * 62)
    out(f"THROUGHPUT RATIO (CAS / Mutex) : {ratio:.4f}x")
    out("=" * 78)
    out()

    out("-" * 78)
    out("ANALYSIS")
    out("-" * 78)
    out("A mutex is a contract with the operating system. When the lock is")
    out("taken, a contending thread is put to sleep and later woken, which")
    out("costs a system call and a context switch, microseconds per handoff.")
    out("Under heavy contention almost every acquisition is a handoff.")
    out()
    out("CAS never sleeps. A single LOCK CMPXCHG instruction compares and")
    out("swaps the value in one indivisible step; if another core won the")
    out("race, the loop simply retries with the fresh value. The cost moves")
    out(f"from the kernel into the cache: the {c_retries:,} failed retries")
    out("above are cores losing the race for the same cache line.")
    out()
    out(f"Contended ratio {ratio:.1f}x vs uncontended ratio "
        f"{m1_ms / c1_ms:.1f}x. The uncontended")
    out("figure includes the Python interpreter overhead on the mutex side,")
    out("since the CAS loop is compiled. What contention adds on top is the")
    out("mutex's sleep/wake handoffs against CAS's cache-line retries.")
    out()
    out("CAS is not free: under extreme contention the retry loop burns CPU")
    out("while making no progress, and a sleeping mutex can be kinder to the")
    out("rest of the system. For a single hot counter, however, an atomic")
    out("is the right tool, and it is what AtomicInteger in Java uses.")
    out("=" * 78)

    with open("task6-p_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWritten: task6-p_output.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
