"""
Task 4: 2D Spatial Convolution - Sobel Horizontal Filter
CUDA Lab 02 - Kystaubay Ayanat (230103075)

Run: python task4_sobel_2d.py
Kx = [[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]] over a 2048 x 2048 float32 matrix.
"""
import numpy as np
from numba import cuda

THREADS_2D = (16, 16)


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)                       # 2D coordinates
    if row < rows and col < cols:
        if 0 < row < rows - 1 and 0 < col < cols - 1:
            d_out[row, col] = (
                -1.0 * d_in[row - 1, col - 1] + 1.0 * d_in[row - 1, col + 1]
                - 2.0 * d_in[row,     col - 1] + 2.0 * d_in[row,     col + 1]
                - 1.0 * d_in[row + 1, col - 1] + 1.0 * d_in[row + 1, col + 1]
            )
        else:
            d_out[row, col] = 0.0                 # outer border pixels


def run_sobel(h_img):
    """Host wrapper: returns the filtered matrix."""
    h_img = np.ascontiguousarray(h_img, dtype=np.float32)
    rows, cols = h_img.shape
    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array_like(d_in)
    blocks_2d = ((cols + THREADS_2D[0] - 1) // THREADS_2D[0],
                 (rows + THREADS_2D[1] - 1) // THREADS_2D[1])
    sobel_x_kernel[blocks_2d, THREADS_2D](d_in, d_out, rows, cols)
    cuda.synchronize()
    return d_out.copy_to_host()


if __name__ == "__main__":
    ROWS = COLS = 2048
    rng = np.random.default_rng(230103075)
    h_img = rng.random((ROWS, COLS), dtype=np.float32)

    out = run_sobel(h_img)

    blocks_2d = ((COLS + THREADS_2D[0] - 1) // THREADS_2D[0],
                 (ROWS + THREADS_2D[1] - 1) // THREADS_2D[1])
    print(f"Image {ROWS} x {COLS} | threads_2d = {THREADS_2D} | "
          f"grid = {blocks_2d[0]} x {blocks_2d[1]} = {blocks_2d[0]*blocks_2d[1]:,} blocks")
    print(f"Border zeroed: top={np.all(out[0, :] == 0.0)} bottom={np.all(out[-1, :] == 0.0)} "
          f"left={np.all(out[:, 0] == 0.0)} right={np.all(out[:, -1] == 0.0)}")

    # flat-field sanity check: a uniform image must give a zero interior gradient
    flat = run_sobel(np.ones((64, 64), dtype=np.float32))
    print(f"Flat-field interior max |gradient| = {np.max(np.abs(flat[1:-1, 1:-1])):.2e}")
    print(f"Random-image interior gradient range: [{out[1:-1,1:-1].min():.4f}, {out[1:-1,1:-1].max():.4f}]")
    print("TASK 4 PASSED")
