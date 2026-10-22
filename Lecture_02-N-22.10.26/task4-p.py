"""
TASK 4 — IPC Overhead (CPU path)

Kystaubay Ayanat — Student ID 230103075

Run:
    py task4-p.py

The GPU branch of this task is not applicable: this machine has no CUDA
device, so the CPU branch is used and the PCIe bus is replaced by the
process boundary, which is the same problem in a different place.

An array of N float64 is handed from a parent process to a worker
process by two routes:

  PATH A  QUEUE / PICKLE      multiprocessing.Queue. The array is
                              serialised by pickle, written through a
                              pipe, and rebuilt in the child. Every byte
                              is touched at least three times.

  PATH B  SHARED MEMORY       multiprocessing.shared_memory. The bytes
                              live in a region both processes map. Only
                              the name of that region travels through the
                              queue, which is a few dozen bytes whatever
                              N is.

Path B is measured twice, because the honest answer depends on where the
data starts:

  B1  including the copy into the shared block, which is the fair
      comparison when the producer already holds an ordinary array.
  B2  handing over data that is already resident in shared memory, which
      is the true zero-copy case and what a pipeline would actually do.

The worker process is started once and reused. Process creation costs
tens of milliseconds and would otherwise swamp the thing being measured.
"""
import os
import platform
import statistics
import sys
import time
from multiprocessing import Process, Queue
from multiprocessing import shared_memory

import numpy as np

STUDENT_ID = 230103075
SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 5
DTYPE = np.float64

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


def checksum(a):
    """Cheap end-to-end check that the bytes really arrived."""
    return float(a[0]) + float(a[-1]) + float(a[len(a) // 2])


def worker_main(cmd_q, ack_q):
    """Runs in the child process for the whole benchmark."""
    while True:
        msg = cmd_q.get()
        if msg is None:
            break
        kind = msg[0]
        if kind == "pickle":
            arr = msg[1]                       # pickle already rebuilt it
            ack_q.put(checksum(arr))
        elif kind == "shm":
            name, n = msg[1], msg[2]
            block = shared_memory.SharedMemory(name=name)
            arr = np.ndarray((n,), dtype=DTYPE, buffer=block.buf)
            value = checksum(arr)              # read straight out of the map
            block.close()
            ack_q.put(value)
        else:
            ack_q.put(None)


def time_path_a(arr, cmd_q, ack_q):
    t0 = time.perf_counter()
    cmd_q.put(("pickle", arr))
    got = ack_q.get()
    return (time.perf_counter() - t0) * 1000.0, got


def time_path_b_with_copy(arr, block_view, name, n, cmd_q, ack_q):
    t0 = time.perf_counter()
    block_view[:] = arr                        # one memcpy into the shared map
    cmd_q.put(("shm", name, n))
    got = ack_q.get()
    return (time.perf_counter() - t0) * 1000.0, got


def time_path_b_resident(name, n, cmd_q, ack_q):
    t0 = time.perf_counter()
    cmd_q.put(("shm", name, n))                # only the name travels
    got = ack_q.get()
    return (time.perf_counter() - t0) * 1000.0, got


def median_of(fn, *args):
    fn(*args)                                   # warm-up
    samples = [fn(*args) for _ in range(REPEATS)]
    times = [s[0] for s in samples]
    return statistics.median(times), min(times), max(times), samples[-1][1]


def main():
    out("=" * 78)
    out("TASK 4 — HOST-TO-DEVICE TRANSFER PENALTY (CPU / IPC BRANCH)")
    out("=" * 78)
    out(f"Student ID           : {STUDENT_ID}")
    out(f"Operating System     : {platform.system()} {platform.release()}")
    out(f"CPU model            : {cpu_model()}")
    out(f"Logical cores        : {os.cpu_count()}")
    out(f"Python               : {platform.python_version()}")
    out(f"Start method         : {__import__('multiprocessing').get_start_method()}")
    out(f"Sizes                : {SIZES}  float64")
    out(f"Timing method        : {REPEATS} timed transfers each, median reported")
    out()
    out("GPU branch not applicable: no CUDA device on this machine. The CPU")
    out("branch measures the process boundary instead of the PCIe bus.")
    out()

    cmd_q, ack_q = Queue(), Queue()
    child = Process(target=worker_main, args=(cmd_q, ack_q), daemon=True)
    child.start()

    # Wake the child and let it finish importing before anything is timed.
    probe = np.zeros(8, dtype=DTYPE)
    cmd_q.put(("pickle", probe))
    ack_q.get()
    out("Worker process started and warmed up.")
    out()

    rows = []
    try:
        for n in SIZES:
            nbytes = n * np.dtype(DTYPE).itemsize
            rng = np.random.default_rng(STUDENT_ID + n)
            arr = rng.standard_normal(n, dtype=DTYPE)
            expected = checksum(arr)

            a_ms, a_min, a_max, a_got = median_of(time_path_a, arr, cmd_q, ack_q)

            block = shared_memory.SharedMemory(create=True, size=nbytes)
            try:
                view = np.ndarray((n,), dtype=DTYPE, buffer=block.buf)
                b1_ms, b1_min, b1_max, b1_got = median_of(
                    time_path_b_with_copy, arr, view, block.name, n, cmd_q, ack_q)
                view[:] = arr
                b2_ms, b2_min, b2_max, b2_got = median_of(
                    time_path_b_resident, block.name, n, cmd_q, ack_q)
            finally:
                block.close()
                block.unlink()

            ok = all(abs(g - expected) < 1e-12
                     for g in (a_got, b1_got, b2_got))
            rows.append({
                "n": n, "bytes": nbytes,
                "a": a_ms, "a_min": a_min, "a_max": a_max,
                "b1": b1_ms, "b1_min": b1_min, "b1_max": b1_max,
                "b2": b2_ms, "b2_min": b2_min, "b2_max": b2_max,
                "ok": ok,
            })
    finally:
        cmd_q.put(None)
        child.join(timeout=5)
        if child.is_alive():
            child.terminate()

    # --------------------------------------------------------- the tables
    out("=" * 78)
    out("TRANSFER OVERHEAD")
    out("=" * 78)
    header = (f"{'N':>12} | {'Size':>9} | {'Path A (ms)':>12} | "
              f"{'Path B1 (ms)':>13} | {'Path B2 (ms)':>13} | {'A/B1':>8} | "
              f"{'A/B2':>9}")
    out(header)
    out("-" * len(header))
    for r in rows:
        out(f"{r['n']:>12,} | {r['bytes'] / 1e6:>7.1f} MB | {r['a']:>12.3f} | "
            f"{r['b1']:>13.3f} | {r['b2']:>13.3f} | "
            f"{r['a'] / r['b1']:>7.2f}x | {r['a'] / r['b2']:>8.2f}x")
    out("-" * len(header))
    out("  Path A  = Queue + pickle serialisation")
    out("  Path B1 = shared memory, including the copy into the shared block")
    out("  Path B2 = shared memory, data already resident (true zero-copy)")
    out()

    out("-" * 78)
    out("EFFECTIVE THROUGHPUT")
    out("-" * 78)
    th = (f"{'N':>12} | {'Path A':>12} | {'Path B1':>12} | {'Path B2':>14}")
    out(th)
    out("-" * len(th))
    for r in rows:
        gb = r["bytes"] / 1e9
        out(f"{r['n']:>12,} | {gb / (r['a'] / 1000):>8.2f} GB/s | "
            f"{gb / (r['b1'] / 1000):>8.2f} GB/s | "
            f"{gb / (r['b2'] / 1000):>10.2f} GB/s")
    out("-" * len(th))
    out("  Path B2's figure is notional: it divides the payload size by a time")
    out("  in which no payload actually moved. A number far above what the")
    out("  memory bus can do is the proof that nothing was copied, not a")
    out("  measurement of copying done quickly.")
    out()

    out("-" * 78)
    out("RUN SPREAD AND CORRECTNESS")
    out("-" * 78)
    for r in rows:
        out(f"N = {r['n']:>12,}  "
            f"A [{r['a_min']:.3f}, {r['a_max']:.3f}]  "
            f"B1 [{r['b1_min']:.3f}, {r['b1_max']:.3f}]  "
            f"B2 [{r['b2_min']:.3f}, {r['b2_max']:.3f}]  "
            f"checksum {'OK' if r['ok'] else 'MISMATCH'}")
    out()

    # ---------------------------------------------------------- analysis
    big = rows[-1]
    small = rows[0]
    out("=" * 78)
    out("ANALYSIS")
    out("=" * 78)
    out("Path A moves data by rewriting it. pickle walks the array and")
    out("produces a byte stream, the operating system copies that stream")
    out("through a pipe, and the child allocates a fresh array and fills it.")
    out("The payload is touched at least three times and exists twice in RAM")
    out("at the peak. The cost therefore grows with N, and grows steeply:")
    out(f"at N = {small['n']:,} path A takes {small['a']:.3f} ms, "
        f"at N = {big['n']:,} it takes {big['a']:.3f} ms.")
    out()
    out("Path B moves nothing. The region is mapped into both address")
    out("spaces, so the child reads the very same physical pages the parent")
    out("wrote. What travels through the queue is the name of the region, a")
    out("handful of bytes, and that is true at every N. Hence B2 is roughly")
    out(f"flat: {small['b2']:.3f} ms at N = {small['n']:,} against "
        f"{big['b2']:.3f} ms at N = {big['n']:,}.")
    out()
    out("B1 sits between them. It still performs one memcpy into the shared")
    out("block, so it scales with N, but a single memcpy is far cheaper than")
    out("serialise-pipe-deserialise. B1 is the number to quote when the")
    out("producer hands over an ordinary array; B2 is what a real pipeline")
    out("achieves by allocating into shared memory in the first place and")
    out("never making the copy at all.")
    out()
    out(f"At the largest size, shared memory is {big['a'] / big['b1']:.1f}x "
        f"faster including the copy")
    out(f"and {big['a'] / big['b2']:.0f}x faster without it.")
    out()
    out("This is the same lesson as the GPU branch of the task. There the")
    out("bottleneck is the PCIe bus and the remedy is to keep data resident")
    out("on the device across several kernels instead of copying it back and")
    out("forth. Here the boundary is between processes and the remedy is to")
    out("keep data in a shared mapping instead of serialising it. In both")
    out("cases the arithmetic is not the problem: moving the bytes is.")
    out("=" * 78)

    with open("task4-p_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWritten: task4-p_output.txt", file=sys.stderr)

    if not all(r["ok"] for r in rows):
        print("CHECKSUM MISMATCH — data did not survive a transfer",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
