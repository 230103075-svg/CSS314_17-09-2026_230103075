"""
Task 1: Warp Divergence Microbenchmark
CUDA Lab 02 - Kystaubay Ayanat (230103075)

Run: python task1_divergence.py
Kernel-only timings (no H2D/D2H), warm-up launch + 10 averaged trials.
"""
import time
import numpy as np
from numba import cuda

N = 2 ** 20                 # 1,048,576 elements
ITERS = 1000
THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
TRIALS = 10


@cuda.jit
def kernel_a_uniform(y):
    """Every thread executes the identical arithmetic path."""
    idx = cuda.grid(1)
    if idx < y.shape[0]:
        for _ in range(ITERS):
            y[idx] = y[idx] * 1.0001 + 0.0001


@cuda.jit
def kernel_b_divergent(y):
    """Interleaved branching: adjacent threads in every warp diverge."""
    idx = cuda.grid(1)
    if idx < y.shape[0]:
        if idx % 2 == 0:
            for _ in range(ITERS):
                y[idx] = y[idx] * 1.0001 + 0.0001      # multiply-accumulate
        else:
            for _ in range(ITERS):
                y[idx] = (y[idx] - 0.0001) / 1.0001    # subtract-divide


@cuda.jit
def kernel_c_warp_aligned(y):
    """Whole warps take the same path - no intra-warp divergence."""
    idx = cuda.grid(1)
    if idx < y.shape[0]:
        warp_id = idx // 32
        if warp_id % 2 == 0:
            for _ in range(ITERS):
                y[idx] = y[idx] * 1.0001 + 0.0001      # Path 1
        else:
            for _ in range(ITERS):
                y[idx] = (y[idx] - 0.0001) / 1.0001    # Path 2


def time_kernel(kernel, d_y, trials=TRIALS):
    """Average kernel-only execution time in milliseconds."""
    kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](d_y)    # warm-up launch (JIT)
    cuda.synchronize()
    total = 0.0
    for _ in range(trials):
        t0 = time.perf_counter()
        kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](d_y)
        cuda.synchronize()
        total += time.perf_counter() - t0
    return total / trials * 1000.0


def run_benchmark():
    h_y = np.ones(N, dtype=np.float32)
    d_y = cuda.to_device(h_y)

    t_a = time_kernel(kernel_a_uniform, d_y)
    t_b = time_kernel(kernel_b_divergent, d_y)
    t_c = time_kernel(kernel_c_warp_aligned, d_y)
    return t_a, t_b, t_c


if __name__ == "__main__":
    dev = cuda.get_current_device()
    name = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
    print(f"GPU: {name} | Compute Capability: {dev.compute_capability} | Warp size: {dev.WARP_SIZE}")
    print(f"N = {N:,} elements | {ITERS} iterations per element | "
          f"grid {BLOCKS_PER_GRID} x {THREADS_PER_BLOCK} | {TRIALS} averaged trials\n")

    t_a, t_b, t_c = run_benchmark()

    print(f"{'Kernel':<34} | {'Time (ms)':>10} | {'vs Kernel A':>12}")
    print("-" * 64)
    print(f"{'A - Uniform path':<34} | {t_a:>10.4f} | {1.0:>11.2f}x")
    print(f"{'B - Full divergence (interleaved)':<34} | {t_b:>10.4f} | {t_b / t_a:>11.2f}x")
    print(f"{'C - Warp-aligned branching':<34} | {t_c:>10.4f} | {t_c / t_a:>11.2f}x")
    print(f"\nDivergence penalty (B / C) = {t_b / t_c:.2f}x")
