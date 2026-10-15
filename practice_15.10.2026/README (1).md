# Advanced GPU Computing with CUDA Python — Module 1

**Student:** Kystaubay Ayanat
**Student ID:** 230103075
**Module:** 1 — CUDA Execution Model (Tasks 01–08)
**Platform:** Google Colab, T4 GPU runtime
**Stack:** Numba CUDA (Python 3.10+)

## Implemented tasks

| # | Task | Signature | Key concept |
|---|---|---|---|
| 01 | 1D Vector Addition with Boundary Checks | `vector_add_1d(a, b, out)` | Threads & Blocks |
| 02 | 1D Grid-Stride Loop Element-wise Scaling | `vector_scale_grid_stride(x, alpha, out)` | Grid-Stride Loops |
| 03 | 2D Matrix Element-wise Addition & Index Mapping | `matrix_add_2d(A, B, C)` | 2D Indexing |
| 04 | 2D Matrix Transpose (Naive Global Memory) | `matrix_transpose_naive(A, B)` | Non-Coalesced Access |
| 05 | 3D Grid Volume Initialization & Index Flattening | `volume_init_3d(vol, D, H, W)` | 3D Thread Geometry |
| 06 | Color Image to Grayscale Luminance Conversion | `rgb_to_grayscale(img_rgb, img_gray)` | Packed Tensor Processing |
| 07 | Vector Linear Interpolation (LERP) | `vector_lerp(a, b, t, out)` | Fused Math Ops |
| 08 | Thresholding & Boolean Mask Generation | `vector_threshold(x, thresh, mask)` | Bitmask Predication |

All eight live in `module1_execution_model.py` with the exact signatures
required by the task sheet.

## How to run

In Google Colab, set the runtime to **T4 GPU** (Runtime → Change runtime type),
then:

```bash
!pip install -q numba
!python selftest_module1.py
```

`selftest_module1.py` reproduces the exact assertion from each task's
"Launch & Verification" line, including the required edge-case sizes
(N ∈ {1, 255, 256, 1024, 1000003} for Task 01, the non-square 513×1021 matrix
for Task 03, and the 33×47×65 volume for Task 05).

## Verification run

Executed on a Google Colab **Tesla T4** runtime (compute capability 7.5):

```
==================================================================
Module 1 — CUDA Execution Model | Kystaubay Ayanat (230103075)
Device: Tesla T4 | compute capability ComputeCapability(major=7, minor=5)
==================================================================
  [PASS] Task 01 — 1D Vector Addition with Boundary Checks
  [PASS] Task 02 — 1D Grid-Stride Loop Element-wise Scaling
  [PASS] Task 03 — 2D Matrix Element-wise Addition & Index Mapping
  [PASS] Task 04 — 2D Matrix Transpose (Naive Global Memory)
  [PASS] Task 05 — 3D Grid Volume Initialization & Index Flattening
  [PASS] Task 06 — Color Image to Grayscale Luminance Conversion
  [PASS] Task 07 — Vector Linear Interpolation (LERP)
  [PASS] Task 08 — Thresholding & Boolean Mask Generation
==================================================================
RESULT: 8/8 tasks passed
All Module 1 kernels verified.
```

The run emits `NumbaPerformanceWarning: Grid size N will likely result in GPU
under-utilization` for Tasks 01, 02 and 08. These are expected, not defects:
Task 01 is deliberately tested at `N = 1` and `N = 255`, which round up to a
single block, and Tasks 02 and 08 are launched on a fixed 64 × 256 grid that
is *smaller* than the data on purpose — demonstrating the grid-stride loop is
the point of those tasks.

The kernel logic was first validated on CPU under `NUMBA_ENABLE_CUDASIM=1`
(8/8 passing) before the hardware run. All results come from a Google Colab
T4 runtime, not from a local machine.

## Files

| File | Purpose |
|---|---|
| `module1_execution_model.py` | The eight CUDA kernels |
| `selftest_module1.py` | Verification harness for Tasks 01–08 |
| `requirements.txt` | numba, cupy-cuda12x, pytest |
| `README.md` | This file |
