import multiprocessing as mp
from multiprocessing import shared_memory
from harness import run_benchmark, emit_verify_tag
import labutil as L

SZ = 8 * 10**6 if L.FAST else 64 * 10**6              # bytes per buffer
FLOPS_W = 6_000_000 if L.FAST else 60_000_000         # FLOPs per worker
N = 64 if L.FAST else 256
BS_LIST = [8, 16, 32, 64, 96, 128]


def _close(shm):
    try:
        shm.close()
    except BufferError:
        pass


# ---------------- 3A: roofline ceilings
def copy_task(a):
    na, nb, s, e = a
    sa, sb = shared_memory.SharedMemory(name=na), shared_memory.SharedMemory(name=nb)
    mv_a, mv_b = sa.buf, sb.buf
    mv_a[s:e] = mv_b[s:e]
    mv_a = mv_b = None
    _close(sa)
    _close(sb)


def fma_task(_):                                       # 16 independent chains, 32 FLOPs / iteration
    m, c = 1.0000001, 1e-7
    x0 = x1 = x2 = x3 = x4 = x5 = x6 = x7 = 1.0
    x8 = x9 = x10 = x11 = x12 = x13 = x14 = x15 = 1.0
    for _ in range(FLOPS_W // 32):
        x0 = x0 * m + c
        x1 = x1 * m + c
        x2 = x2 * m + c
        x3 = x3 * m + c
        x4 = x4 * m + c
        x5 = x5 * m + c
        x6 = x6 * m + c
        x7 = x7 * m + c
        x8 = x8 * m + c
        x9 = x9 * m + c
        x10 = x10 * m + c
        x11 = x11 * m + c
        x12 = x12 * m + c
        x13 = x13 * m + c
        x14 = x14 * m + c
        x15 = x15 * m + c
    return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15


# ---------------- 3B: GEMM variants on flat double buffers
def gemm_task(a):
    names, variant, r0, r1, BS = a
    shs = [shared_memory.SharedMemory(name=n) for n in names]
    A, B, C = [s.buf[:N * N * 8].cast("d") for s in shs]
    if variant == 1:                                   # naive i-j-k, strided B access
        for i in range(r0, r1):
            ai = i * N
            for j in range(N):
                s = 0.0
                for k in range(N):
                    s += A[ai + k] * B[k * N + j]
                C[ai + j] = s
    elif variant == 2:                                 # i-k-j, hoisted a_ik, unit-stride j
        for i in range(r0, r1):
            ci = i * N
            for j in range(N):
                C[ci + j] = 0.0
            for k in range(N):
                a_ik = A[ci + k]
                bk = k * N
                for j in range(N):
                    C[ci + j] += a_ik * B[bk + j]
    else:                                              # 6-loop blocked, (i,k,j) micro-kernel
        for i in range(r0, r1):
            for j in range(N):
                C[i * N + j] = 0.0
        for ii in range(r0, r1, BS):
            i_max = min(ii + BS, r1)
            for jj in range(0, N, BS):
                j_max = min(jj + BS, N)
                for kk in range(0, N, BS):
                    k_max = min(kk + BS, N)
                    for i in range(ii, i_max):
                        ci = i * N
                        for k in range(kk, k_max):
                            a_ik = A[ci + k]
                            bk = k * N
                            for j in range(jj, j_max):
                                C[ci + j] += a_ik * B[bk + j]
    A.release()
    B.release()
    C.release()
    for s in shs:
        _close(s)


if __name__ == "__main__":
    seed = L.seed_of(L.get_id())
    plog, pphys = mp.cpu_count(), L.detect_pphys()
    ci = L.cache_info()
    W, R = L.WARMUP, L.RUNS
    pool = mp.Pool(processes=pphys)
    shms = []
    try:
        # ---- 3A bandwidth
        sa = shared_memory.SharedMemory(create=True, size=SZ)
        sb = shared_memory.SharedMemory(create=True, size=SZ)
        shms += [sa, sb]
        tasks = [(sa.name, sb.name, SZ * t // pphys, SZ * (t + 1) // pphys) for t in range(pphys)]
        t_bw = run_benchmark(lambda: pool.map(copy_task, tasks, chunksize=1), W, R).median_ms
        BW = 2 * SZ / ((t_bw / 1e3) * 1e9)
        # ---- 3A compute
        t_fma = run_benchmark(lambda: pool.map(fma_task, range(pphys), chunksize=1), W, R).median_ms
        PEAK = pphys * FLOPS_W / ((t_fma / 1e3) * 1e9)
        ridge = PEAK / BW

        # ---- 3B GEMM
        gs = [shared_memory.SharedMemory(create=True, size=N * N * 8) for _ in range(3)]
        shms += gs
        names = [g.name for g in gs]
        Av, Bv, Cv = [g.buf[:N * N * 8].cast("d") for g in gs]
        for idx in range(N * N):
            Av[idx] = ((idx * 7 + 3) % 17) / 17.0
            Bv[idx] = ((idx * 5 + 1) % 13) / 13.0
        rows_r = [(N * t // pphys, N * (t + 1) // pphys) for t in range(pphys)]

        def gemm_bench(v, BS=0):
            tk = [(names, v, r0, r1, BS) for r0, r1 in rows_r]
            return run_benchmark(lambda: pool.map(gemm_task, tk, chunksize=1), W, R).median_ms

        gf = lambda ms: 2 * N**3 / (ms * 1e6)
        t1 = gemm_bench(1)
        t2 = gemm_bench(2)
        c2 = list(Cv)
        sweep, maxdiff = [], 0.0
        for BS in BS_LIST:
            t3 = gemm_bench(3, BS)
            d = max(abs(x - y) for x, y in zip(list(Cv), c2))
            maxdiff = max(maxdiff, d)
            sweep.append((BS, t3, gf(t3), 3 * BS * BS * 8, BS / 12))
        best = max(sweep, key=lambda r: r[2])
        BSs = best[0]

        g1, g2, g3 = gf(t1), gf(t2), best[2]
        tag = emit_verify_tag("PHASE3", seed, g1, g3)

        def fits(w):
            out = []
            for nm, sz in (("L1d", ci["l1d"]), ("L2", ci["l2"])):
                if sz:
                    out.append(f"{'fits' if w <= sz else 'exceeds'} {nm} ({L.fmt_bytes(sz)})")
            return ", ".join(out) or "cache sizes unknown"

        t_roof = L.md_table(["Ceiling", "Value"], [
            ["BW_sustained (copy, Pphys workers)", f"{BW:.2f} GB/s (median {t_bw:.1f} ms)"],
            ["P_sustained (16-var scalar FMA, Pphys workers)", f"{PEAK:.4f} GFLOP/s (median {t_fma:.1f} ms)"],
            ["Ridge point I*", f"{ridge:.4f} FLOP/B"]])
        I_assumed = {1: 0.125, 2: 0.25, 3: best[4]}
        t_gemm = L.md_table(["Variant", "median ms", "GFLOP/s", "I (FLOP/B, assumed)", "Roofline bound", "% of bound"], [
            ["1 naive i-j-k", f"{t1:.1f}", f"{g1:.4f}", I_assumed[1], f"{min(PEAK, BW * 0.125):.2f}", f"{g1 / min(PEAK, BW * 0.125) * 100:.2f}%"],
            ["2 reordered i-k-j", f"{t2:.1f}", f"{g2:.4f}", I_assumed[2], f"{min(PEAK, BW * 0.25):.2f}", f"{g2 / min(PEAK, BW * 0.25) * 100:.2f}%"],
            [f"3 blocked BS*={BSs}", f"{best[1]:.1f}", f"{g3:.4f}", f"{best[4]:.2f}", f"{min(PEAK, BW * best[4]):.2f}", f"{g3 / min(PEAK, BW * best[4]) * 100:.2f}%"]])
        t_sweep = L.md_table(["BS", "median ms", "GFLOP/s", "W_tile (B)", "I_tiled = BS/12"],
                             [[b, f"{t:.1f}", f"{g:.4f}", w, f"{i:.2f}"] for b, t, g, w, i in sweep])
        md = (f"## Phase 3: Roofline & Cache-Blocked GEMM (N={N}, p=Pphys={pphys})\n\n### 3A Ceilings\n\n{t_roof}\n\n"
              f"### 3B GEMM variants\n\n{t_gemm}\n\n### Tile sweep (Variant 3)\n\n{t_sweep}\n\n"
              f"max |C3 - C2| over all BS = {maxdiff:.2e} ({'PASS' if maxdiff < 1e-9 else 'FAIL'}, threshold 1e-9)\n\n"
              f"### Analysis\n\n"
              f"- BS* = {BSs}: W_tile = {best[3]} B ({best[3] / 1024:.1f} KiB), {fits(best[3])}. "
              f"The 3 tiles (A, B, C) should live in L1d/L2 so the inner j-loop reuses B and C rows from cache.\n"
              f"- Variant 1 walks B with stride N*8 = {N * 8} B (one new cache line per k step); Variant 2 makes the inner loop unit-stride "
              f"and hoists a_ik; Variant 3 adds reuse of tiles, raising operational intensity to ~BS/12.\n"
              f"- All GEMM points sit far below the roofline: CPython interpreter overhead, not memory or FPU, is the bottleneck "
              f"(P_sustained itself is measured through the same interpreter, so it is an interpreter-limited ceiling). "
              f"Cache blocking gains therefore look smaller than in C. Assumed I for Variants 1-2 is a rough estimate.\n\n```\n{tag}\n```\n")
        L.save_phase(3, md, {"hash": tag, "BW": BW, "peak": PEAK, "g1": g1, "g3": g3, "BS": BSs})
        Av.release(); Bv.release(); Cv.release()
    finally:
        pool.close()
        pool.join()
        for s in shms:
            _close(s)
            try:
                s.unlink()
            except Exception:
                pass
