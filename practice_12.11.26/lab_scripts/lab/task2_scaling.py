import math
import multiprocessing as mp
from harness import run_benchmark, emit_verify_tag
import labutil as L

N_STRONG = 60_000 if L.FAST else 1_200_000
N0 = 10_000 if L.FAST else 200_000


def stage_a(n_el):                      # serial: 10 sin/sqrt ops per element
    acc = 0.0
    for i in range(n_el):
        v = 0.1 + i * 1e-6
        for _ in range(5):
            v = math.sqrt(1.0 + math.sin(v))
        acc += v
    return acc


def stage_b(rng):                       # parallel: 30 FMA/trig ops per element, no shared writes
    s, e = rng
    acc = 0.0
    for i in range(s, e):
        v = 0.1 + i * 1e-6
        for _ in range(10):
            v = v * 0.9999991 + 3e-7    # FMA
            w = math.sin(v)             # trig
            v = w * 0.5 + v * 0.5       # FMA
        acc += v
    return acc


def chunks(n, p):
    k = 4 * p
    return [(n * c // k, n * (c + 1) // k) for c in range(k)]


def measure(p, n):
    """Median ms of full pipeline (Stage A serial + Stage B on persistent Pool)."""
    pool = mp.Pool(processes=p)         # created OUTSIDE the timed loop
    try:
        def trial():
            stage_a(int(0.05 * n))
            pool.map(stage_b, chunks(n, p), chunksize=1)
        return run_benchmark(trial, L.WARMUP, L.RUNS).median_ms
    finally:
        pool.close()
        pool.join()


def t_stage_a(n):
    return run_benchmark(lambda: stage_a(int(0.05 * n)), L.WARMUP, L.RUNS).median_ms


if __name__ == "__main__":
    seed = L.seed_of(L.get_id())
    plog, pphys = mp.cpu_count(), L.detect_pphys()
    ps = list(range(1, plog + 1))

    # ---- strong scaling
    ta1 = t_stage_a(N_STRONG)
    T = {p: measure(p, N_STRONG) for p in ps}
    S = {p: T[1] / T[p] for p in ps}
    s_frac = ta1 / T[1]
    cap = 1 / s_frac
    rows = [[p, f"{T[p]:.1f}", f"{S[p]:.2f}", f"{S[p] / p:.2f}"] for p in ps]
    t_strong = L.md_table(["p", "T_p ms", "S_p", "E_p"], rows)

    # ---- weak scaling
    TW, TA = {}, {}
    for p in ps:
        n = p * N0
        TW[p] = measure(p, n)
        TA[p] = t_stage_a(n)
    rows, SG = [], {}
    for p in ps:
        alpha = TA[p] / TW[p]
        SG[p] = p - alpha * (p - 1)
        rows.append([p, p * N0, f"{TW[p]:.1f}", f"{TW[1] / TW[p]:.2f}", f"{alpha:.4f}", f"{SG[p]:.2f}"])
    t_weak = L.md_table(["p", "N(p)", "T_weak ms", "E_weak", "alpha", "S_Gustafson"], rows)

    # ---- inflection Pphys -> Pphys+1
    if pphys < plog and pphys >= 2:
        d_after = S[pphys + 1] - S[pphys]
        d_before = (S[pphys] - S[1]) / (pphys - 1)
        infl = (f"Marginal speedup S({pphys + 1}) - S({pphys}) = {d_after:.2f} versus an average of "
                f"{d_before:.2f} per added process below Pphys ({d_after / d_before * 100:.0f}% of earlier gain): "
                f"beyond {pphys} physical cores extra processes land on SMT siblings (or E-cores) that share "
                f"execution units and L1/L2, so returns diminish.")
    else:
        infl = "Pphys == Plog (or Pphys < 2) on this machine: no SMT boundary to measure."

    tag = emit_verify_tag("PHASE2", seed, S[plog], SG[plog])
    md = (f"## Phase 2: Strong vs Weak Scaling\n\n### Strong (N={N_STRONG})\n\n{t_strong}\n\n"
          f"Stage A(1) = {ta1:.1f} ms, s = {s_frac:.4f}, Amdahl cap S_inf = 1/s = {cap:.1f}; "
          f"observed S_Plog = {S[plog]:.2f} ({S[plog] / cap * 100:.0f}% of the cap). "
          f"At this p the cap is not the limit; hardware (SMT sharing, turbo drop, pool overhead) is.\n\n"
          f"### Weak (N0={N0} per worker)\n\n{t_weak}\n\n### Architectural inflection\n\n{infl}\n\n```\n{tag}\n```\n")
    L.save_phase(2, md, {"hash": tag, "S_strong_plog": S[plog], "S_gust_plog": SG[plog]})
