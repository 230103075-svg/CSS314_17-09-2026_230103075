"""
Module 1 — CUDA Execution Model (Tasks 01-08)

Kystaubay Ayanat — 230103075
Advanced GPU Computing with CUDA Python

Every public name below is a @cuda.jit kernel with exactly the signature
required by the task sheet. Launch geometry is documented above each kernel.
"""
import numpy as np
from numba import cuda, float32, uint8


# ---------------------------------------------------------------- Task 01
@cuda.jit
def vector_add_1d(a, b, out):
    """C[i] = A[i] + B[i] for float32 arrays of arbitrary length N.

    Launch: blockDim=(256,), gridDim=((N + 255) // 256,)
    Concept: Threads & Blocks — O(N)
    """
    idx = cuda.grid(1)
    # Grid is rounded UP, so the tail block always over-provisions threads.
    if idx < a.size:
        out[idx] = a[idx] + b[idx]


# ---------------------------------------------------------------- Task 02
@cuda.jit
def vector_scale_grid_stride(x, alpha, out):
    """out = x * alpha using a grid-stride loop.

    Launch: fixed geometry, e.g. 64 blocks x 256 threads, decoupled from N.
    stride = gridDim.x * blockDim.x == cuda.gridsize(1)
    Concept: Grid-Stride Loops — O(N)
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, x.size, stride):
        out[i] = x[i] * alpha


# ---------------------------------------------------------------- Task 03
@cuda.jit
def matrix_add_2d(A, B, C):
    """C = A + B for 2D matrices of shape (M, N).

    Launch: blockDim=(16, 16), gridDim=((N + 15) // 16, (M + 15) // 16)
    Concept: 2D Indexing — O(M*N)
    """
    # Explicit mapping as specified: row from the y axis, col from the x axis.
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    M, N = A.shape
    if row < M and col < N:
        C[row, col] = A[row, col] + B[row, col]


# ---------------------------------------------------------------- Task 04
@cuda.jit
def matrix_transpose_naive(A, B):
    """B = A.T — A is (M, N), B is (N, M).

    Reads are coalesced along rows, writes are strided down columns, which is
    exactly the non-coalesced write pattern this task is meant to expose.

    Launch: blockDim=(16, 16), gridDim=((N + 15) // 16, (M + 15) // 16)
    Concept: Non-Coalesced Access — O(M*N)
    """
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    M, N = A.shape
    if row < M and col < N:
        B[col, row] = A[row, col]


# ---------------------------------------------------------------- Task 05
@cuda.jit
def volume_init_3d(vol, D, H, W):
    """vol[z, y, x] = z * 10000 + y * 100 + x for a (D, H, W) tensor.

    Launch: blockDim=(8, 8, 8),
            gridDim=((W + 7) // 8, (H + 7) // 8, (D + 7) // 8)
    Concept: 3D Thread Geometry — O(D*H*W)
    """
    # cuda.grid(3) yields (x, y, z) in that order.
    x, y, z = cuda.grid(3)
    if z < D and y < H and x < W:
        vol[z, y, x] = z * 10000 + y * 100 + x


# ---------------------------------------------------------------- Task 06
@cuda.jit
def rgb_to_grayscale(img_rgb, img_gray):
    """ITU-R BT.601 luminance: Y = 0.299*R + 0.587*G + 0.114*B.

    img_rgb is (H, W, 3); img_gray is (H, W).
    Launch: blockDim=(16, 16), gridDim=((W + 15) // 16, (H + 15) // 16)
    Concept: Packed Tensor Processing — O(H*W)
    """
    x, y = cuda.grid(2)
    H = img_gray.shape[0]
    W = img_gray.shape[1]
    if y < H and x < W:
        r = float32(img_rgb[y, x, 0])
        g = float32(img_rgb[y, x, 1])
        b = float32(img_rgb[y, x, 2])
        img_gray[y, x] = 0.299 * r + 0.587 * g + 0.114 * b


# ---------------------------------------------------------------- Task 07
@cuda.jit
def vector_lerp(a, b, t, out):
    """Element-wise linear interpolation out[i] = (1 - t[i])*a[i] + t[i]*b[i].

    Written in fused multiply-add form so the compiler emits FFMA instead of
    a separate multiply and add.

    Launch: blockDim=(256,), gridDim=((N + 255) // 256,)
    Concept: Fused Math Ops — O(N)
    """
    idx = cuda.grid(1)
    if idx < out.size:
        ti = t[idx]
        ai = a[idx]
        # a + t*(b - a) is algebraically (1-t)*a + t*b, one FFMA chain.
        out[idx] = ai + ti * (b[idx] - ai)


# ---------------------------------------------------------------- Task 08
@cuda.jit
def vector_threshold(x, thresh, mask):
    """mask[i] = 1 if x[i] >= thresh else 0, as uint8.

    The ternary compiles to a predicated select, so the whole warp retires in
    one pass with no divergent branch.

    Launch: grid-stride loop, blockDim=(256,)
    Concept: Bitmask Predication — O(N)
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, x.size, stride):
        mask[i] = uint8(1) if x[i] >= thresh else uint8(0)
