"""
AUTONOMOUS VERIFICATION & INTEGRITY TOKEN GENERATOR
Run: python verify_submission.py
"""

import sys
import hashlib
import numpy as np
from numba import cuda


def main():
    print("=" * 60)
    print("RUNNING CUDA LAB 02 AUTONOMOUS VERIFICATION")
    print("=" * 60)

    student_id = input("Enter your Student ID: ").strip()
    if not student_id:
        print("[FAIL] Student ID cannot be empty.")
        sys.exit(1)

    # 1. Verify Task 2
    try:
        import task2_stencil_1d as t2
        N = 10007
        test_in = np.sin(np.linspace(0, 10, N)).astype(np.float32)
        h_gpu = t2.run_stencil(test_in)
        h_cpu = t2.cpu_stencil(test_in)
        assert np.allclose(h_gpu, h_cpu, atol=1e-4), "Task 2 output mismatch against CPU"
        print("[PASS] Task 2 (1D Stencil & Clamping)")
    except Exception as e:
        print(f"[FAIL] Task 2: {e}")
        sys.exit(1)

    # 2. Verify Task 3
    try:
        import task3_grid_stride as t3
        N = 100000
        test_arr = np.ones(N, dtype=np.float32)
        factor = 4.25
        res = t3.run_grid_stride(test_arr, factor)
        assert np.allclose(res, factor), "Task 3 elements not uniformly scaled"
        print("[PASS] Task 3 (Grid-Stride Scaling)")
    except Exception as e:
        print(f"[FAIL] Task 3: {e}")
        sys.exit(1)

    # 3. Verify Task 4
    try:
        import task4_sobel_2d as t4
        test_img = np.ones((64, 64), dtype=np.float32)
        sobel_res = t4.run_sobel(test_img)
        # Uniform image must have zero interior gradient
        assert np.max(np.abs(sobel_res[1:-1, 1:-1])) < 1e-5, "Task 4 interior gradient != 0 for flat field"
        assert np.all(sobel_res[0, :] == 0.0), "Task 4 border rows not zeroed"
        assert np.all(sobel_res[:, 0] == 0.0), "Task 4 border columns not zeroed"
        print("[PASS] Task 4 (2D Sobel Horizontal)")
    except Exception as e:
        print(f"[FAIL] Task 4: {e}")
        sys.exit(1)

    # Cryptographic Checksum Token Generation
    hasher = hashlib.sha256()
    hasher.update(student_id.encode('utf-8'))
    try:
        device_name = cuda.get_current_device().name
        hasher.update(device_name if isinstance(device_name, bytes) else device_name.encode('utf-8'))
    except Exception:
        hasher.update(b"UNKNOWN_CUDA_DEVICE")
    hasher.update(h_gpu[:32].tobytes())
    hasher.update(sobel_res[:8, :8].tobytes())
    token = hasher.hexdigest()[:20].upper()

    print("\n" + "=" * 60)
    print("VERIFICATION SUCCESSFUL")
    print(f"OFFICIAL SUBMISSION TOKEN: {token}")
    print("=" * 60)
    print("Copy this token directly into your README.md.\n")


if __name__ == "__main__":
    main()
