"""
TASK 1 — Hardware Probe & Serial Baseline

Kystaubay Ayanat — Student ID 230103075

Run:
    python task1.py

Prints the report to the console AND writes the identical text to
task1_output.txt, so the committed file is a byte-for-byte copy of what
the console showed.
"""
import os

# Pin every numeric backend to one thread BEFORE numpy is imported, so the
# baseline is provably serial and cannot silently use a multi-threaded BLAS.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import platform                                    # noqa: E402
import sys                                         # noqa: E402
import time                                        # noqa: E402

import numpy as np                                 # noqa: E402

STUDENT_ID = 230103075
N = 20_000_000

LINES = []


def out(text=""):
    """Print and capture, so console and task1_output.txt cannot diverge."""
    print(text, flush=True)
    LINES.append(str(text))


def cpu_model():
    """Best available CPU model string for the current platform."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Windows":
            name = os.environ.get("PROCESSOR_IDENTIFIER", "")
            if name:
                return name
        elif system == "Darwin":
            import subprocess
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).decode().strip()
    except Exception:                               # noqa: BLE001
        pass
    return platform.processor() or platform.machine() or "unknown"


def physical_cores():
    """Physical core count if psutil is installed, otherwise None."""
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except Exception:                               # noqa: BLE001
        return None


def main():
    # ---------------------------------------------------- 1. Hardware probe
    out("=" * 70)
    out("TASK 1 — HARDWARE PROBE & SERIAL BASELINE")
    out("=" * 70)
    out(f"Student ID          : {STUDENT_ID}")
    out(f"Operating System    : {platform.system()} {platform.release()}")
    out(f"Platform detail     : {platform.platform()}")
    out(f"Machine / arch      : {platform.machine()}")
    out(f"CPU model           : {cpu_model()}")
    out(f"Logical cores       : {os.cpu_count()}   (os.cpu_count())")
    phys = physical_cores()
    out(f"Physical cores      : {phys if phys is not None else 'n/a (pip install psutil)'}")
    out(f"Python              : {platform.python_implementation()} "
        f"{platform.python_version()}")
    out(f"NumPy               : {np.__version__}")
    out(f"Threading           : single thread (all BLAS/OpenMP vars pinned to 1)")
    out()

    # ------------------------------------------- 2. Deterministic generation
    out("-" * 70)
    out("ARRAY GENERATION")
    out("-" * 70)
    out(f"Elements            : {N:,}")
    out(f"dtype               : float64")
    out(f"Memory footprint    : {N * 8 / (1024 ** 2):.2f} MiB")
    out(f"Seed                : {STUDENT_ID}  (Student ID)")
    out(f"Distribution        : standard normal, mean 0.0, sigma 1.0")

    t0 = time.perf_counter()
    rng = np.random.default_rng(STUDENT_ID)
    data = rng.standard_normal(N, dtype=np.float64)
    gen_ms = (time.perf_counter() - t0) * 1000.0

    out(f"Generation time     : {gen_ms:.3f} ms")
    out()

    # --------------------------------------- 3 & 4. Serial compute + timing
    out("-" * 70)
    out("SERIAL COMPUTATION (single thread)")
    out("-" * 70)

    t0 = time.perf_counter()
    total = np.sum(data, dtype=np.float64)
    sum_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    std_pop = np.std(data, dtype=np.float64)          # population, ddof = 0
    std_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    std_sample = np.std(data, dtype=np.float64, ddof=1)   # sample, ddof = 1
    std_sample_ms = (time.perf_counter() - t0) * 1000.0

    compute_ms = sum_ms + std_ms
    mean = total / N

    # Full precision kept throughout; rounding happens only at print time.
    out(f"Sum                 : {float(total)!r}")
    out(f"Sum (17 s.f.)       : {total:.17g}")
    out(f"Mean                : {mean:.17g}")
    out(f"Std dev (ddof=0)    : {std_pop:.17g}")
    out(f"Std dev (ddof=1)    : {std_sample:.17g}")
    out()
    out(f"Sum time            : {sum_ms:.3f} ms")
    out(f"Std dev time        : {std_ms:.3f} ms   (ddof=0)")
    out(f"Std dev time        : {std_sample_ms:.3f} ms   (ddof=1)")
    out()
    out("=" * 70)
    out(f"SERIAL BASELINE WALL-CLOCK TIME : {compute_ms:.3f} ms")
    out(f"  (sum + standard deviation, generation excluded)")
    out(f"TOTAL INCLUDING GENERATION      : {gen_ms + compute_ms:.3f} ms")
    out(f"THROUGHPUT                      : "
        f"{N / (compute_ms / 1000.0) / 1e6:.2f} M elements/sec")
    out("=" * 70)

    with open("task1_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWritten: task1_output.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
