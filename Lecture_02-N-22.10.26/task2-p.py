"""
TASK 2 — Pseudo-Random Generator State Bottleneck

Kystaubay Ayanat — Student ID 230103075

Run:
    py task2-p.py

Monte Carlo estimate of pi from 40,000,000 sample points, computed twice
on 4 threads:

  SHARED       one PRNG instance for all four threads, every draw taken
               while holding a mutex.
  INDEPENDENT  one PRNG per thread, seeded (Student_ID * 10) + thread_index,
               no lock anywhere.

The point of the exercise is that a PRNG is not a pure function. It owns
mutable state that every draw advances, so two threads cannot draw from
one instance at the same time without corrupting the stream. Guarding it
with a mutex keeps it correct and makes generation strictly sequential:
four threads, one generator, one stream at a time.

Both runs draw in batches of the same size, so what is being compared is
the serialisation of the generator itself, not Python loop overhead.
Section 3 measures the per-draw lock cost separately to show why nobody
takes the lock one sample at a time.
"""
import math
import os
import platform
import statistics
import sys
import threading
import time

import numpy as np

STUDENT_ID = 230103075
N_SAMPLES = 40_000_000
N_THREADS = 4
BATCH = 500_000                 # samples drawn per lock acquisition
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


def count_inside(x, y):
    """Points falling inside the unit quarter-circle."""
    return int(np.count_nonzero(x * x + y * y <= 1.0))


# ------------------------------------------------------- shared generator
def run_shared():
    """One PRNG, one mutex. Every draw is serialised behind the lock."""
    rng = np.random.default_rng(STUDENT_ID * 10)
    lock = threading.Lock()
    remaining = [N_SAMPLES]
    partial = [0] * N_THREADS

    def worker(wid):
        hits = 0
        while True:
            with lock:
                take = min(BATCH, remaining[0])
                if take <= 0:
                    break
                remaining[0] -= take
                # Drawing happens INSIDE the lock: this is the shared state.
                x = rng.random(take)
                y = rng.random(take)
            hits += count_inside(x, y)          # arithmetic outside the lock
        partial[wid] = hits

    threads = [threading.Thread(target=worker, args=(w,))
               for w in range(N_THREADS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ms = (time.perf_counter() - t0) * 1000.0
    return 4.0 * sum(partial) / N_SAMPLES, ms, sum(partial)


# -------------------------------------------------- independent generators
def split_counts(total, parts):
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def run_independent():
    """One PRNG per thread, seeded (Student_ID * 10) + thread_index."""
    shares = split_counts(N_SAMPLES, N_THREADS)
    partial = [0] * N_THREADS

    def worker(wid):
        seed = STUDENT_ID * 10 + wid
        rng = np.random.default_rng(seed)       # private state, no sharing
        left = shares[wid]
        hits = 0
        while left > 0:
            take = min(BATCH, left)
            left -= take
            x = rng.random(take)
            y = rng.random(take)
            hits += count_inside(x, y)
        partial[wid] = hits

    threads = [threading.Thread(target=worker, args=(w,))
               for w in range(N_THREADS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ms = (time.perf_counter() - t0) * 1000.0
    return 4.0 * sum(partial) / N_SAMPLES, ms, sum(partial)


def measure(fn):
    fn()                                        # warm-up
    runs = [fn() for _ in range(REPEATS)]
    times = [r[1] for r in runs]
    med = statistics.median(times)
    # keep the run whose time is the median
    chosen = min(runs, key=lambda r: abs(r[1] - med))
    return chosen[0], med, chosen[2], min(times), max(times)


def main():
    out("=" * 78)
    out("TASK 2 — PSEUDO-RANDOM GENERATOR STATE BOTTLENECK")
    out("=" * 78)
    out(f"Student ID           : {STUDENT_ID}")
    out(f"Operating System     : {platform.system()} {platform.release()}")
    out(f"CPU model            : {cpu_model()}")
    out(f"Logical cores        : {os.cpu_count()}")
    out(f"Threads              : {N_THREADS}")
    out(f"Sample points        : {N_SAMPLES:,}")
    out(f"Batch per draw       : {BATCH:,} samples")
    out(f"Shared PRNG seed     : {STUDENT_ID * 10:,}")
    out(f"Independent seeds    : " +
        ", ".join(f"T{w}={STUDENT_ID * 10 + w:,}" for w in range(N_THREADS)))
    out(f"Reference            : math.pi = {math.pi:.17g}")
    out(f"Timing method        : {REPEATS} timed runs each, median reported")
    out()

    # ------------------------------------------------------------ run them
    out("-" * 78)
    out("1. SHARED PRNG, GUARDED BY A MUTEX")
    out("-" * 78)
    s_pi, s_ms, s_hits, s_min, s_max = measure(run_shared)
    s_err = abs(s_pi - math.pi)
    out(f"Computed pi          : {s_pi:.17g}")
    out(f"Absolute error       : {s_err:.6e}")
    out(f"Points inside circle : {s_hits:,} of {N_SAMPLES:,}")
    out(f"Wall time            : {s_ms:.3f} ms")
    out(f"Run spread           : min {s_min:.3f} ms, max {s_max:.3f} ms")
    out()

    out("-" * 78)
    out("2. INDEPENDENT PRNG PER THREAD, NO LOCK")
    out("-" * 78)
    i_pi, i_ms, i_hits, i_min, i_max = measure(run_independent)
    i_err = abs(i_pi - math.pi)
    out(f"Computed pi          : {i_pi:.17g}")
    out(f"Absolute error       : {i_err:.6e}")
    out(f"Points inside circle : {i_hits:,} of {N_SAMPLES:,}")
    out(f"Wall time            : {i_ms:.3f} ms")
    out(f"Run spread           : min {i_min:.3f} ms, max {i_max:.3f} ms")
    out()

    # ------------------------------------------- per-draw lock cost probe
    out("-" * 78)
    out("3. WHY THE DRAWS ARE BATCHED")
    out("-" * 78)
    probe_n = 200_000
    rng = np.random.default_rng(STUDENT_ID * 10)
    lock = threading.Lock()
    t0 = time.perf_counter()
    for _ in range(probe_n):
        with lock:
            rng.random()
            rng.random()
    per_draw_ms = (time.perf_counter() - t0) * 1000.0
    extrapolated = per_draw_ms * (N_SAMPLES / probe_n)
    out(f"Taking the lock once per sample, measured over {probe_n:,} samples:")
    out(f"  measured            : {per_draw_ms:.3f} ms")
    out(f"  per sample          : {per_draw_ms * 1e3 / probe_n:.3f} us")
    out(f"  extrapolated to {N_SAMPLES:,} samples : "
        f"{extrapolated:.0f} ms ({extrapolated / 1000:.1f} s)")
    out()
    out(f"That is {extrapolated / s_ms:.0f}x slower than the batched shared run")
    out(f"above. Both versions serialise the generator equally; the difference")
    out(f"is {N_SAMPLES:,} lock acquisitions instead of "
        f"{math.ceil(N_SAMPLES / BATCH):,}. Batching")
    out("hides the lock overhead but cannot hide the serialisation, which is")
    out("why section 1 is still the slower of the two real runs.")
    out()

    # ----------------------------------------------------- the deliverable
    speedup = s_ms / i_ms
    out("=" * 78)
    out("DELIVERABLE")
    out("=" * 78)
    out(f"{'':22} | {'Computed pi':>20} | {'Abs. error':>13} | {'Time (ms)':>11}")
    out("-" * 78)
    out(f"{'Shared PRNG':22} | {s_pi:>20.15f} | {s_err:>13.6e} | {s_ms:>11.3f}")
    out(f"{'Independent PRNG':22} | {i_pi:>20.15f} | {i_err:>13.6e} | {i_ms:>11.3f}")
    out("-" * 78)
    out()
    out(f"SPEEDUP FROM ELIMINATING GENERATOR STATE LOCKING : {speedup:.4f}x")
    out(f"  (shared {s_ms:.3f} ms / independent {i_ms:.3f} ms, "
        f"{N_THREADS} threads both)")
    out("=" * 78)
    out()

    # ---------------------------------------------------------- analysis
    out("-" * 78)
    out("ANALYSIS")
    out("-" * 78)
    out("A pseudo-random generator is a state machine. Each draw reads the")
    out("current state, transforms it and writes it back, and the next draw")
    out("depends on that write. Two threads drawing from one instance at the")
    out("same time would interleave those read-modify-writes and corrupt the")
    out("stream: repeated values, broken statistical properties, and a stream")
    out("that is no longer reproducible from its seed.")
    out()
    out("The mutex prevents that, correctly. What it cannot do is make the")
    out("generator parallel. While one thread holds the lock, the other three")
    out("are blocked, so generation proceeds at exactly single-thread speed no")
    out(f"matter how many cores are available. Only the hit test runs in")
    out("parallel, and that is the cheaper half of the work.")
    out()
    out("Giving each thread its own generator removes the shared state")
    out("entirely. There is nothing to lock because there is nothing shared:")
    out("four generators advance four independent streams on four cores.")
    out(f"Measured gain here: {speedup:.2f}x.")
    out()
    if speedup < N_THREADS * 0.6:
        out(f"The gain is below the {N_THREADS}x that {N_THREADS} threads might "
            f"suggest. Generation is")
        out("not the only cost: the hit test was already running in parallel in")
        out("both versions, so only the generation half was ever serialised.")
        out("Amdahl's Law applies directly here, with the lock defining the")
        out("serial fraction.")
    out()
    out("The seeds matter as much as the speed. Each thread is seeded")
    out(f"(Student_ID * 10) + thread_index, so the four streams are distinct")
    out("and the whole run is still reproducible: the same seeds give the same")
    out("estimate on any machine. Seeding every thread identically would be")
    out("the classic error, producing four copies of one stream and an answer")
    out("with a quarter of the independent samples it appears to have.")
    out()
    out(f"Both estimates land within {max(s_err, i_err):.2e} of math.pi. Monte")
    out(f"Carlo error falls as 1/sqrt(N), so {N_SAMPLES:,} samples buy roughly")
    out(f"{1 / math.sqrt(N_SAMPLES):.2e} relative accuracy. The two differ from")
    out("each other only because they consumed different random streams, which")
    out("is expected and is not an error in either one.")
    out("=" * 78)

    with open("task2-p_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWritten: task2-p_output.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
