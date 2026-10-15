"""
Self-check for Module 1 (Tasks 01-08) — Kystaubay Ayanat, 230103075

Run in Colab with a GPU runtime:

    !python selftest_module1.py

Each check reproduces the exact assertion from the task sheet's
"Launch & Verification" line. Set NUMBA_ENABLE_CUDASIM=1 to validate the
indexing logic on a CPU-only machine (much slower, so sizes shrink).
"""
import os
import sys

import numpy as np
from numba import cuda

from module1_execution_model import (
    vector_add_1d,
    vector_scale_grid_stride,
    matrix_add_2d,
    matrix_transpose_naive,
    volume_init_3d,
    rgb_to_grayscale,
    vector_lerp,
    vector_threshold,
)

SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
PASSED, FAILED = [], []


def check(task, title, fn):
    try:
        fn()
        PASSED.append(task)
        print(f"  [PASS] Task {task:02d} — {title}")
    except Exception as exc:                      # noqa: BLE001
        FAILED.append((task, exc))
        print(f"  [FAIL] Task {task:02d} — {title}\n         {exc}")


# ---------------------------------------------------------------- Task 01
def t01():
    lengths = [1, 255, 256, 1024, 1003] if SIM else [1, 255, 256, 1024, 1000003]
    for n in lengths:
        a = np.random.rand(n).astype(np.float32)
        b = np.random.rand(n).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        d_a, d_b, d_out = cuda.to_device(a), cuda.to_device(b), cuda.to_device(out)
        threads = 256
        blocks = (n + 255) // 256
        vector_add_1d[blocks, threads](d_a, d_b, d_out)
        np.testing.assert_allclose(d_out.copy_to_host(), a + b, atol=1e-5)


# ---------------------------------------------------------------- Task 02
def t02():
    n = 50_000 if SIM else 10_000_000
    alpha = np.float32(2.5)
    x = np.random.rand(n).astype(np.float32)
    d_x = cuda.to_device(x)
    d_out = cuda.device_array(n, dtype=np.float32)
    # Fixed geometry, deliberately far smaller than N.
    blocks, threads = 64, 256
    assert blocks * threads < n, "grid must be strictly smaller than N"
    vector_scale_grid_stride[blocks, threads](d_x, alpha, d_out)
    np.testing.assert_allclose(d_out.copy_to_host(), x * alpha, atol=1e-5)


# ---------------------------------------------------------------- Task 03
def t03():
    M, N = (65, 97) if SIM else (513, 1021)       # non-square on purpose
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    d_A, d_B = cuda.to_device(A), cuda.to_device(B)
    d_C = cuda.device_array((M, N), dtype=np.float32)
    block = (16, 16)
    grid = ((N + 15) // 16, (M + 15) // 16)
    matrix_add_2d[grid, block](d_A, d_B, d_C)
    np.testing.assert_allclose(d_C.copy_to_host(), A + B, atol=1e-5)


# ---------------------------------------------------------------- Task 04
def t04():
    M, N = (70, 45) if SIM else (700, 453)
    A = np.random.rand(M, N).astype(np.float32)
    d_A = cuda.to_device(A)
    d_B = cuda.device_array((N, M), dtype=np.float32)
    block = (16, 16)
    grid = ((N + 15) // 16, (M + 15) // 16)
    matrix_transpose_naive[grid, block](d_A, d_B)
    np.testing.assert_allclose(d_B.copy_to_host(), A.T, atol=1e-5)


# ---------------------------------------------------------------- Task 05
def t05():
    D, H, W = (9, 11, 13) if SIM else (33, 47, 65)
    d_vol = cuda.device_array((D, H, W), dtype=np.float32)
    block = (8, 8, 8)
    grid = ((W + 7) // 8, (H + 7) // 8, (D + 7) // 8)
    volume_init_3d[grid, block](d_vol, D, H, W)

    z, y, x = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    expected = (z * 10000 + y * 100 + x).astype(np.float32)
    np.testing.assert_array_equal(d_vol.copy_to_host(), expected)


# ---------------------------------------------------------------- Task 06
def t06():
    H, W = (24, 32) if SIM else (480, 640)
    img = (np.random.rand(H, W, 3) * 255).astype(np.float32)
    d_img = cuda.to_device(img)
    d_gray = cuda.device_array((H, W), dtype=np.float32)
    block = (16, 16)
    grid = ((W + 15) // 16, (H + 15) // 16)
    rgb_to_grayscale[grid, block](d_img, d_gray)

    expected = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])
    np.testing.assert_allclose(d_gray.copy_to_host(), expected, atol=1e-4)


# ---------------------------------------------------------------- Task 07
def t07():
    n = 5_000 if SIM else 1_048_576
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    t = np.random.rand(n).astype(np.float32)
    d_a, d_b, d_t = cuda.to_device(a), cuda.to_device(b), cuda.to_device(t)
    d_out = cuda.device_array(n, dtype=np.float32)
    threads = 256
    blocks = (n + 255) // 256
    vector_lerp[blocks, threads](d_a, d_b, d_t, d_out)
    np.testing.assert_allclose(d_out.copy_to_host(), (1 - t) * a + t * b, atol=1e-5)


# ---------------------------------------------------------------- Task 08
def t08():
    n = 5_000 if SIM else 1_000_003
    x = np.random.rand(n).astype(np.float32)
    thresh = np.float32(0.5)
    d_x = cuda.to_device(x)
    d_mask = cuda.device_array(n, dtype=np.uint8)
    vector_threshold[64, 256](d_x, thresh, d_mask)   # grid-stride loop
    np.testing.assert_array_equal(d_mask.copy_to_host(),
                                  (x >= thresh).astype(np.uint8))


def main():
    print("=" * 66)
    print("Module 1 — CUDA Execution Model | Kystaubay Ayanat (230103075)")
    if SIM:
        print("Running under NUMBA_ENABLE_CUDASIM=1 (CPU simulator, small sizes)")
    else:
        gpu = cuda.get_current_device()
        name = gpu.name
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        print(f"Device: {name} | compute capability {gpu.compute_capability}")
    print("=" * 66)

    check(1, "1D Vector Addition with Boundary Checks", t01)
    check(2, "1D Grid-Stride Loop Element-wise Scaling", t02)
    check(3, "2D Matrix Element-wise Addition & Index Mapping", t03)
    check(4, "2D Matrix Transpose (Naive Global Memory)", t04)
    check(5, "3D Grid Volume Initialization & Index Flattening", t05)
    check(6, "Color Image to Grayscale Luminance Conversion", t06)
    check(7, "Vector Linear Interpolation (LERP)", t07)
    check(8, "Thresholding & Boolean Mask Generation", t08)

    print("=" * 66)
    print(f"RESULT: {len(PASSED)}/8 tasks passed")
    if FAILED:
        print("Failed tasks: " + ", ".join(str(t) for t, _ in FAILED))
        sys.exit(1)
    print("All Module 1 kernels verified.")


if __name__ == "__main__":
    main()
