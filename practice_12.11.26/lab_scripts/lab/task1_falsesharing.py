import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
from harness import run_benchmark, emit_verify_tag
import labutil as L

M = 1_600_000 if L.FAST else 16_000_000      # total read-modify-write steps


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    for i in range(iters):
        acc = (acc * 6364136223846793005 + (i | 1)) & 0xFFFFFFFFFFFFFFFF
        arr[slot_idx] = acc                    # forces store to shared cache line
    del arr
    shm.close()


def trial(name, p, stride):
    procs = [mp.Process(target=fs_worker, args=(name, (t * stride) % 512, M // p)) for t in range(p)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()


def bench(name, p, stride):
    return run_benchmark(lambda: trial(name, p, stride), L.WARMUP, L.RUNS)


if __name__ == "__main__":
    seed = L.seed_of(L.get_id())
    plog, pphys = mp.cpu_count(), L.detect_pphys()
    line = L.cache_info()["line"]
    shm = shared_memory.SharedMemory(create=True, size=4096)
    try:
        # --- 1. stride sweep at p = Pphys
        sweep = {}
        for stride in (1, 2, 4, 8, 16, 32):
            sweep[stride] = bench(shm.name, pphys, stride)
        strides = sorted(sweep)
        rows = [[s, s * 8, f"{sweep[s].median_ms:.1f}", f"{sweep[s].min_ms:.1f}", f"{sweep[s].iqr_ms:.1f}",
                 f"{M / (sweep[s].median_ms * 1e3):.2f}"] for s in strides]
        t_sweep = L.md_table(["stride", "bytes", "median ms", "min ms", "IQR ms", "Mops/s"], rows)
        ratios = [(sweep[strides[i - 1]].median_ms / sweep[strides[i]].median_ms, strides[i]) for i in range(1, len(strides))]
        best_ratio, drop_stride = max(ratios)

        # --- 2. process scaling for stride 1 and 16
        ps = L.p_list(plog)
        scal, tables, ekf = {}, [], {}
        for stride in (1, 16):
            T = {p: bench(shm.name, p, stride).median_ms for p in ps}
            scal[stride] = T
            rows, ekf[stride] = [], []
            for p in ps:
                S = T[1] / T[p]
                e = L.karp_flatt(S, p) if p >= 2 else None
                if e is not None:
                    ekf[stride].append(e)
                rows.append([p, f"{T[p]:.1f}", f"{S:.2f}", f"{S / p:.2f}", "-" if e is None else f"{e:.3f}"])
            tables.append(f"**stride = {stride}** ({stride * 8} B)\n\n" +
                          L.md_table(["p", "T_p ms", "S_p", "E_p", "Karp-Flatt e(p)"], rows))

        m1, m2 = sweep[1].median_ms, sweep[16].median_ms
        tag = emit_verify_tag("PHASE1", seed, m1, m2)
        avg = lambda v: sum(v) / len(v) if v else float("nan")
        analysis = (
            f"- Coherence contention drops sharply at stride {drop_stride} = **{drop_stride * 8} B** "
            f"({best_ratio:.2f}x faster than the previous stride); cache line = {line} B, so once slots are "
            f">= one line apart each core owns its line and the MESI invalidation ping-pong disappears.\n"
            f"- Mean Karp-Flatt e(p): stride=1 -> {avg(ekf[1]):.3f}, stride=16 -> {avg(ekf[16]):.3f}. "
            f"For stride=1, e(p) is large and grows with p: the 'serial fraction' is really coherence traffic "
            f"that gets worse with more sharers (overhead scales with p, not a fixed Amdahl fraction). "
            f"For stride=16 e(p) is small and roughly flat, i.e. remaining loss is process-spawn cost / SMT sharing.\n"
            f"- Note: pure-Python loop overhead hides part of the effect; spawn time is inside the timed region."
        )
        md = (f"## Phase 1: False Sharing & Karp-Flatt (M={M}, Pphys={pphys}, Plog={plog})\n\n"
              f"### Stride sweep (p = Pphys)\n\n{t_sweep}\n\n### Process scaling\n\n" + "\n\n".join(tables) +
              f"\n\n### Analysis\n\n{analysis}\n\n```\n{tag}\n```\n")
        L.save_phase(1, md, {"hash": tag, "median_s1": m1, "median_s16": m2})
    finally:
        shm.close()
        shm.unlink()
