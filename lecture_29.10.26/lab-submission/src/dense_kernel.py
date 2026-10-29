"""
Milestone 1, Task 1.2 — Loop interchange: i-j-k vs i-k-j.

Usage:
    py src/dense_kernel.py

C = A @ B on an M x M sub-block, M = N_salted / 2, written two ways:

  Algorithm A  naive i-j-k  : innermost loop runs over k, so B[k, j]
                              jumps a whole row (M * 8 bytes) every step.
  Algorithm B  interchanged i-k-j : A[i, k] is hoisted into a register and
                              row C[i, :] is updated from row B[k, :] with
                              stride-1, contiguous access.

Pure Python would take hours at this size, so both loops are compiled
with Numba (the handout allows Cython/Numba/ctypes). The arithmetic is
identical in both: each C[i, j] accumulates over k in the same order, so
the results must agree to within 1e-12.

Writes results/loop_interchange.json (+ the two result matrices as .npy)
"""
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:
    raise SystemExit("numba is required:  py -m pip install numba")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RUNS = 3


@njit(cache=False)
def matmul_ijk(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            acc = 0.0
            for k in range(n):
                acc += A[i, k] * B[k, j]       # B strides by n elements
            C[i, j] = acc


@njit(cache=False)
def matmul_ikj(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for k in range(n):
            a = A[i, k]                        # hoisted into a register
            for j in range(n):
                C[i, j] += a * B[k, j]         # stride-1 over B and C rows


def digest(M):
    """SHA-256 of the values rounded to 6 decimal places."""
    return hashlib.sha256(np.round(M, 6).tobytes()).hexdigest()


def main():
    p = RESULTS / "salted_params.json"
    if not p.exists():
        raise SystemExit("results/salted_params.json missing — run src/profile_env.py first")
    params = json.loads(p.read_text(encoding="utf-8"))
    N = int(params["N"])
    M = N // 2
    rng = np.random.default_rng(int(params["seed"]))
    A = rng.random((M, M))
    B = rng.random((M, M))

    print("=" * 70)
    print("MILESTONE 1.2 — LOOP INTERCHANGE (i-j-k vs i-k-j)")
    print("=" * 70)
    print(f"Salted N    : {N}    sub-block M = N/2 = {M}")
    print(f"Seed        : {params['seed']}")
    print(f"Inner FMAs  : {M ** 3:,} per algorithm")
    print(f"Runs        : {RUNS}, median reported (compiled with Numba)")

    tiny = np.ones((4, 4))
    matmul_ijk(tiny, tiny, np.zeros((4, 4)))
    matmul_ikj(tiny, tiny, np.zeros((4, 4)))

    def bench(fn):
        times, C = [], None
        for _ in range(RUNS):
            C = np.zeros((M, M))
            t0 = time.perf_counter()
            fn(A, B, C)
            times.append(time.perf_counter() - t0)
        return statistics.median(times), times, C

    t_naive, runs_naive, C_naive = bench(matmul_ijk)
    print(f"\ni-j-k (naive)        : {t_naive * 1000:10.3f} ms   runs {[round(t * 1000, 2) for t in runs_naive]}")
    t_inter, runs_inter, C_inter = bench(matmul_ikj)
    print(f"i-k-j (interchanged) : {t_inter * 1000:10.3f} ms   runs {[round(t * 1000, 2) for t in runs_inter]}")

    speedup = t_naive / t_inter
    max_diff = float(np.max(np.abs(C_naive - C_inter)))
    ref_diff = float(np.max(np.abs(C_inter - A @ B)))
    ok = max_diff <= 1e-12
    gflops = lambda t: 2 * M ** 3 / t / 1e9                    # noqa: E731

    print(f"\nSPEEDUP S = T_naive / T_interchanged = {speedup:.4f}x")
    print(f"GFLOP/s : naive {gflops(t_naive):.3f}  |  interchanged {gflops(t_inter):.3f}")
    print(f"max |C_ijk - C_ikj|  = {max_diff:.3e}  -> {'PASS' if ok else 'FAIL'} (eps 1e-12)")
    print(f"max |C_ikj - A @ B|  = {ref_diff:.3e}  (NumPy/BLAS reference)")
    print(f"\nWhy: in i-j-k the inner loop reads B[k, j] down a column, {M * 8:,} bytes apart,")
    print("so every multiply pulls a fresh cache line and uses 8 of its 64 bytes.")
    print("In i-k-j the inner loop walks rows of B and C contiguously: every byte of")
    print("each line is used, the prefetcher keeps up, and the loop vectorises.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS / "dense_C_naive.npy", C_naive)
    np.save(RESULTS / "dense_C_interchanged.npy", C_inter)
    out = {
        "student_id": params["student_id"],
        "N_salted": N, "M_subblock": M, "seed": params["seed"],
        "compiler": "numba njit",
        "runs": RUNS,
        "t_naive_ijk_s": t_naive, "t_naive_runs_s": runs_naive,
        "t_interchanged_ikj_s": t_inter, "t_interchanged_runs_s": runs_inter,
        "speedup": speedup,
        "gflops_naive": gflops(t_naive), "gflops_interchanged": gflops(t_inter),
        "max_abs_diff_ijk_vs_ikj": max_diff,
        "max_abs_diff_vs_numpy": ref_diff,
        "epsilon": 1e-12, "numerically_equivalent": ok,
        "checksum_naive": float(C_naive.sum()),
        "checksum_interchanged": float(C_inter.sum()),
        "sha256_round6_naive": digest(C_naive),
        "sha256_round6_interchanged": digest(C_inter),
    }
    (RESULTS / "loop_interchange.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWritten: {RESULTS / 'loop_interchange.json'}")


if __name__ == "__main__":
    main()
