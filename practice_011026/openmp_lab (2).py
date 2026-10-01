"""
LAB PRACTICUM: OpenMP Multi-Core Scaling in Python (Numba / prange)
Kystaubay Ayanat - 230103075

Run:  pip install numpy numba matplotlib
      python openmp_lab.py            (also fine inside Colab: !python openmp_lab.py)

Produces:
  - console tables for Challenges 1, 2, 3
  - results.csv          (every number needed for Table 1 of the scorecard)
  - mandelbrot_output.png (Challenge 2, deliverable D)
"""
import time, csv, platform
import numpy as np
import numba
from numba import njit, prange

ROWS = []          # rows for results.csv


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------- Challenge 1
@njit(parallel=True)
def monte_carlo_pi(n_samples):
    inside_circle = 0
    for i in prange(n_samples):
        x = np.random.uniform(0.0, 1.0)
        y = np.random.uniform(0.0, 1.0)
        if x * x + y * y <= 1.0:
            inside_circle += 1          # Numba infers reduction(+:inside_circle)
    return (4.0 * inside_circle) / n_samples


def challenge1():
    log("\n=== CHALLENGE 1: Monte Carlo Pi - core speedometer & Amdahl's Law ===")
    _ = monte_carlo_pi(10_000)          # JIT warm-up, excluded from timings

    SAMPLES = 120_000_000
    maxthr = numba.config.NUMBA_NUM_THREADS
    counts = sorted(set(t for t in [1, 2, 4, 8, maxthr] if t <= maxthr))

    log(f"{'Threads':<10} | {'Time (s)':<12} | {'Pi':<12} | {'Speedup':<10} | {'Efficiency (%)':<15}")
    log("-" * 72)
    t1 = None
    for t in counts:
        numba.set_num_threads(t)
        start = time.perf_counter()
        pi_est = monte_carlo_pi(SAMPLES)
        elapsed = time.perf_counter() - start
        if t == 1:
            t1, speedup, eff = elapsed, 1.0, 100.0
        else:
            speedup = t1 / elapsed
            eff = speedup / t * 100.0
        log(f"{t:<10} | {elapsed:<12.4f} | {pi_est:<12.6f} | {speedup:<10.2f}x | {eff:<15.1f}")
        ROWS.append(["Challenge 1: Monte Carlo", t, SAMPLES, f"{elapsed:.4f}",
                     f"{speedup:.2f}x", f"{eff:.1f}%", f"pi={pi_est:.6f}"])
    numba.set_num_threads(maxthr)


# ----------------------------------------------------------- Challenge 2
@njit(parallel=True)
def render_mandelbrot_rows(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):                               # row-parallel
        cy = -1.2 + (r / h) * 2.4
        for c in range(w):
            cx = -2.0 + (c / w) * 2.5
            z_real, z_imag = 0.0, 0.0
            it = 0
            while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img


@njit(parallel=True)
def render_mandelbrot_cols(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for c in prange(w):                               # column-parallel
        cx = -2.0 + (c / w) * 2.5
        for r in range(h):
            cy = -1.2 + (r / h) * 2.4
            z_real, z_imag = 0.0, 0.0
            it = 0
            while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img


def challenge2():
    log("\n=== CHALLENGE 2: Mandelbrot - load imbalance & memory layout ===")
    _ = render_mandelbrot_rows(100, 100, 50)
    _ = render_mandelbrot_cols(100, 100, 50)

    H, W, MAX_IT = 2500, 2500, 1000
    maxthr = numba.config.NUMBA_NUM_THREADS

    t0 = time.perf_counter(); grid_rows = render_mandelbrot_rows(H, W, MAX_IT)
    t_rows = time.perf_counter() - t0
    t0 = time.perf_counter(); render_mandelbrot_cols(H, W, MAX_IT)
    t_cols = time.perf_counter() - t0

    log(f"Row-Parallel Render Time   : {t_rows:.3f} s")
    log(f"Column-Parallel Render Time: {t_cols:.3f} s")
    log(f"Rows / Cols ratio          : {t_rows / t_cols:.2f}x")
    ROWS.append(["Challenge 2: Mandelbrot (Rows)", maxthr, "2500 x 2500", f"{t_rows:.3f}", "N/A", "N/A", ""])
    ROWS.append(["Challenge 2: Mandelbrot (Cols)", maxthr, "2500 x 2500", f"{t_cols:.3f}", "N/A", "N/A",
                 f"rows/cols={t_rows / t_cols:.2f}"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_rows, cmap="magma", extent=[-2.0, 0.5, -1.2, 1.2])
    plt.title(f"Mandelbrot {H}x{W} (Render: {t_rows:.2f}s)")
    plt.axis("off")
    plt.savefig("mandelbrot_output.png", dpi=300, bbox_inches="tight")
    plt.close()
    log("Saved image: mandelbrot_output.png")


# ----------------------------------------------------------- Challenge 3
@njit(parallel=True)
def heat_step(u, u_next, alpha=0.20):
    rows, cols = u.shape
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            u_next[i, j] = u[i, j] + alpha * (
                u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4.0 * u[i, j]
            )


def run_heat(dtype, label):
    GRID_SIZE, STEPS = 1500, 300
    u = np.zeros((GRID_SIZE, GRID_SIZE), dtype=dtype)
    u_next = np.zeros_like(u)
    for arr in (u, u_next):                 # Dirichlet: top & left walls at 100 C
        arr[0, :] = 100.0
        arr[:, 0] = 100.0
    heat_step(u, u_next)                    # warm-up (JIT per dtype)

    start = time.perf_counter()
    for _ in range(STEPS):
        heat_step(u, u_next)
        u, u_next = u_next, u               # pointer swap
    elapsed = time.perf_counter() - start
    mcells = (GRID_SIZE * GRID_SIZE * STEPS) / elapsed / 1e6
    log(f"{label:<28}: {elapsed:7.3f} s | {mcells:8.2f} Megacells/sec")
    ROWS.append([f"Challenge 3: Heat Stencil ({label})", numba.config.NUMBA_NUM_THREADS,
                 "1500 x 1500 x 300", f"{elapsed:.3f}", "N/A", "N/A", f"{mcells:.2f} Mcells/s"])
    return elapsed


def challenge3():
    log("\n=== CHALLENGE 3: Heat stencil - memory bandwidth wall ===")
    t64 = run_heat(np.float64, "float64")
    t32 = run_heat(np.float32, "float32")
    log(f"float64 / float32 speedup   : {t64 / t32:.2f}x")

    # extra: thread scaling of the stencil, shows the bandwidth roofline
    maxthr = numba.config.NUMBA_NUM_THREADS
    log("\nStencil thread scaling (float64):")
    base = None
    for t in sorted(set(x for x in [1, 2, 4, maxthr] if x <= maxthr)):
        numba.set_num_threads(t)
        el = run_heat(np.float64, f"float64, {t} thread(s)")
        if base is None:
            base = el
        log(f"    speedup vs 1 thread     : {base / el:.2f}x")
    numba.set_num_threads(maxthr)


# ----------------------------------------------------------- main
if __name__ == "__main__":
    log("=== ENVIRONMENT ===")
    log(f"{platform.platform()}")
    log(f"Python {platform.python_version()} | NumPy {np.__version__} | Numba {numba.__version__}")
    log(f"Hardware Threads Detected (NUMBA_NUM_THREADS): {numba.config.NUMBA_NUM_THREADS}")
    log(f"Threading layer: {numba.config.THREADING_LAYER}")

    challenge1()
    challenge2()
    challenge3()

    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "active_threads", "size", "time_s", "speedup", "efficiency", "note"])
        w.writerows(ROWS)
    log("\nWritten: results.csv, mandelbrot_output.png")
