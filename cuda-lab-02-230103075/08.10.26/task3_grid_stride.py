"""
Task 3: Arbitrary-Size Vector Scaling via Grid-Stride Loops
CUDA Lab 02 - Kystaubay Ayanat (230103075)

Run: python task3_grid_stride.py
16,384 hardware threads process 16,777,216 elements through a grid-stride loop.
"""
import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = 64                 # 64 * 256 = 16,384 threads only


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)             # global thread index
    stride = cuda.gridsize(1)        # total grid span = 16,384
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    """Host caller: transfers, launch with the fixed 64 x 256 grid, result back."""
    h_arr = np.ascontiguousarray(h_arr, dtype=np.float32)
    N = h_arr.shape[0]
    d_arr = cuda.to_device(h_arr)
    grid_stride_scale_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](d_arr, factor, N)
    cuda.synchronize()
    return d_arr.copy_to_host()


if __name__ == "__main__":
    N = 2 ** 24                       # 16,777,216 elements
    factor = 4.25
    h_arr = np.ones(N, dtype=np.float32)

    result = run_grid_stride(h_arr, factor)

    threads = BLOCKS_PER_GRID * THREADS_PER_BLOCK
    print(f"N = {N:,} elements | hardware threads launched = {threads:,} "
          f"({BLOCKS_PER_GRID} blocks x {THREADS_PER_BLOCK})")
    print(f"Elements per thread (grid-stride iterations) = {N // threads:,}")
    assert np.allclose(result, factor), "elements not uniformly scaled"
    print(f"All {N:,} elements equal {factor} -> TASK 3 PASSED")
