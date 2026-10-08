# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230103075
**Allocated GPU Node:** Tesla T4
**CUDA Compute Capability:** 7.5
**Official Verification Token:** D57A7ED28CB31C7FF567

---

## Environment

| Item | Value |
|---|---|
| Platform | Google Colab, T4 GPU runtime |
| GPU | Tesla T4 (Turing), 15360 MiB VRAM, 40 SMs, warp size 32 |
| Compute capability | 7.5 |
| Driver / CUDA version | 580.82.07 / CUDA 13.0 |
| Python / NumPy / Numba | 3.13.15 / 2.1.3 / 0.61.2 |
| Run date | 08.10.2026 |

## Task 1 — Warp Divergence Microbenchmark

`N = 2^20 = 1,048,576` float32 elements, 1,000 iterations per element, grid 4096 x 256,
kernel-only timing (no H2D/D2H transfers), warm-up launch followed by 10 averaged trials.

| Kernel | Branching pattern | Kernel time (ms) | Slowdown vs Kernel A |
|---|---|---|---|
| A — Uniform path | all threads execute identical arithmetic | **40.7665** | 1.00x |
| B — Full divergence | `idx % 2` — adjacent threads diverge inside every warp | **132.8725** | **3.26x** |
| C — Warp-aligned | `(idx // 32) % 2` — whole warps take one path | **67.0523** | **1.64x** |

**Divergence penalty (B / C) = 1.98x**

**Observation.** An SM issues instructions per warp of 32 threads, so a warp has a single program
counter. In Kernel B the predicate `idx % 2 == 0` splits every single warp into two halves: the
hardware must execute the multiply-accumulate path with 16 threads active and the subtract-divide
path with the other 16 masked off, then swap — the two paths run one after another rather than
together, and each instruction slot wastes half of its lanes. That is why B costs 132.87 ms against
40.77 ms for the uniform kernel.

Kernel C evaluates exactly the same branch condition and performs exactly the same amount of
arithmetic, but the predicate is computed from `warp_id = idx // 32`, so all 32 threads of a warp
agree and the warp takes a single path with every lane active. The result is 67.05 ms — the measured
1.98x gap between B and C is the cost of divergence alone, almost exactly the theoretical 2x for a
two-way split. The residual 1.64x of C over A comes from the extra division instruction on Path 2
(division has far lower throughput than multiply-add on a Turing SM), not from divergence.

**Takeaway:** branch granularity matters more than branch count. Aligning conditional work to
multiples of 32 keeps full warp occupancy even when the control flow is non-trivial.

## Task 2 — 1D Boundary Stencil & Halo Protection

3-point smoothing filter `y[i] = 0.25*x[i-1] + 0.5*x[i] + 0.25*x[i+1]` with halo replication
(clamping) at `idx == 0` and `idx == N-1`, launched with 256-thread blocks.

- Test size: `N = 100,007` (odd, non-power-of-two) → **391 blocks x 256 threads = 100,096 threads**
  (89 threads switched off by the `if idx < N` guard)
- Validated against the exact NumPy reference `cpu_stencil` using `np.allclose(..., atol=1e-4)`
- Result: `TASK 2 PASSED: MAX DELTA = 5.960464477539063e-08`

The maximum deviation is 5.96e-08, i.e. one unit in the last place of float32 — the GPU result is
bit-level equivalent to the CPU reference up to floating-point rounding order.

## Task 3 — Arbitrary-Size Vector Scaling via Grid-Stride Loops

- Vector size: `N = 2^24 = 16,777,216` elements, scale factor 4.25
- Launch geometry: **64 blocks x 256 threads = 16,384 hardware threads** — 1,024x fewer threads
  than elements, as required by the hardware constraint
- Each thread processes **1,024 elements** via `for i in range(start, N, stride)` where
  `start = cuda.grid(1)` and `stride = cuda.gridsize(1) = 16,384`
- Host check: all 16,777,216 elements equal 4.25 → `TASK 3 PASSED`

The stride equals the full grid span, so consecutive threads still touch consecutive addresses on
every iteration — memory access stays fully coalesced while the software problem size is decoupled
from the hardware thread count.

## Task 4 — 2D Spatial Convolution (Sobel Horizontal)

- Input: `2048 x 2048` float32 matrix
- `threads_2d = (16, 16)` = 256 threads per block; grid computed dynamically →
  **128 x 128 = 16,384 blocks** (4,194,304 threads, one per pixel)
- Interior pixels (`0 < row < rows-1` and `0 < col < cols-1`) get the full `Kx` convolution;
  all four outer borders are set to `0.0`
- Border check: `top=True bottom=True left=True right=True`
- Flat-field sanity check: a uniform image gives interior max |gradient| = `0.00e+00`
- Random-image interior gradient range: `[-3.6643, 3.8189]` → `TASK 4 PASSED`

## Verification

```bash
python verify_submission.py      # enter 230103075 when prompted
```

Output:

```
[PASS] Task 2 (1D Stencil & Clamping)
[PASS] Task 3 (Grid-Stride Scaling)
[PASS] Task 4 (2D Sobel Horizontal)
VERIFICATION SUCCESSFUL
OFFICIAL SUBMISSION TOKEN: D57A7ED28CB31C7FF567
```

## Files

| File | Purpose |
|---|---|
| `task1_divergence.py` | Warp divergence microbenchmark (kernels A / B / C) |
| `task2_stencil_1d.py` | 1D stencil kernel with boundary clamping + exact CPU validator |
| `task3_grid_stride.py` | Grid-stride loop scaling a 16.7M-element vector with 16,384 threads |
| `task4_sobel_2d.py` | 2D Sobel-X convolution with zeroed borders |
| `verify_submission.py` | Automated self-check and integrity token generator |

## How to run

```bash
python task1_divergence.py     # warp divergence table
python task2_stencil_1d.py     # stencil + CPU cross-check
python task3_grid_stride.py    # grid-stride scaling
python task4_sobel_2d.py       # 2D Sobel
python verify_submission.py    # all checks + token
```

Requires an NVIDIA GPU with CUDA, `numpy` and `numba`. All benchmarks above were executed on the
Tesla T4 node described in the Environment table.
