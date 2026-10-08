"""
Task 2: 1D Boundary Stencil & Halo Protection
CUDA Lab 02 - Kystaubay Ayanat (230103075)

Run: python task2_stencil_1d.py
3-point smoothing filter y[i] = 0.25*x[i-1] + 0.5*x[i] + 0.25*x[i+1]
with halo replication (clamping) at both ends.
"""
import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256


@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)
    if idx < N:                                   # boundary guard
        if idx == 0:
            left = d_in[0]                        # halo replication, left edge
        else:
            left = d_in[idx - 1]
        if idx == N - 1:
            right = d_in[N - 1]                   # halo replication, right edge
        else:
            right = d_in[idx + 1]
        d_out[idx] = 0.25 * left + 0.5 * d_in[idx] + 0.25 * right


def run_stencil(h_in):
    """Host wrapper: device transfers + launch with 256-thread blocks."""
    h_in = np.ascontiguousarray(h_in, dtype=np.float32)
    N = h_in.shape[0]
    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(d_in)
    blocks_per_grid = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    stencil_1d[blocks_per_grid, THREADS_PER_BLOCK](d_in, d_out, N)
    cuda.synchronize()
    return d_out.copy_to_host()


def cpu_stencil(arr):
    """Exact NumPy reference validator."""
    padded = np.pad(arr, (1, 1), mode='edge')
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


if __name__ == "__main__":
    N = 100_007                                   # odd, non-power-of-two
    h_in = np.sin(np.linspace(0, 10, N)).astype(np.float32)

    h_out_gpu = run_stencil(h_in)
    cpu_ref = cpu_stencil(h_in)

    delta = float(np.max(np.abs(h_out_gpu - cpu_ref)))
    assert np.allclose(h_out_gpu, cpu_ref, atol=1e-4), f"mismatch, max delta = {delta}"
    print(f"N = {N:,} | blocks = {(N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK} x {THREADS_PER_BLOCK} threads")
    print(f"TASK 2 PASSED: MAX DELTA = {delta}")
