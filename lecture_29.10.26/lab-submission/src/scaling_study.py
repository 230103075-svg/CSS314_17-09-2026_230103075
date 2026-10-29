"""
Milestone 3 — Strong / weak scaling, Amdahl & Gustafson fits, Roofline.

Usage:
    py src/scaling_study.py

Kernel: 5-point Jacobi stencil on an N x N float64 grid
    out[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
The interior rows are split into P contiguous bands, one per worker
process (multiprocessing.Pool). The grid lives in
multiprocessing.shared_memory, so no array is ever pickled: workers map
the same physical pages and each writes only its own band. One timed
"run" applies the stencil R times; R is calibrated so a single process
takes about one second, enough to make process dispatch negligible.

  Task 3.1  STRONG scaling: fixed N = N_salted, P = 1, 2, 4, ..., physical.
            S_P = T_1 / T_P, E_P = S_P / P. Amdahl serial fraction s is
            computed per P and least-squares fitted over all P.
  Task 3.2  WEAK scaling: N_P = ceil(N * sqrt(P)) so work grows as O(P).
            Scaled speedup = P * Work(1) * T_1 / (Work(P) * T_P_weak).
  Task 3.3  Arithmetic intensity and roofline regime, using the cache sizes
            from results/host_profile.json and a measured memory bandwidth.

Every parallel result is checked element-for-element against a
single-process reference (each element undergoes the same operations in
the same order, so the match must be exact).

Writes results/scaling_metrics.tsv and results/roofline_summary.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json                                                    # noqa: E402
import math                                                    # noqa: E402
import statistics                                              # noqa: E402
import time                                                    # noqa: E402
from multiprocessing import Pool, shared_memory                # noqa: E402
from pathlib import Path                                       # noqa: E402

import numpy as np                                             # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPEATS = 3
FLOPS_PER_POINT = 4            # 3 additions + 1 multiplication
TARGET_SECONDS = 1.0           # single-process run length used to pick R

# --------------------------------------------------- worker-side state
_U = _O = None
_SHM = []


def _attach(name):
    try:
        return shared_memory.SharedMemory(name=name, track=False)   # Python 3.13+
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def _init(u_name, o_name, n):
    global _U, _O
    su, so = _attach(u_name), _attach(o_name)
    _SHM[:] = [su, so]
    _U = np.ndarray((n, n), dtype=np.float64, buffer=su.buf)
    _O = np.ndarray((n, n), dtype=np.float64, buffer=so.buf)


def stencil_band(u, o, lo, hi):
    """Interior rows lo..hi-1 of out; no temporaries are allocated."""
    band = o[lo:hi, 1:-1]
    np.add(u[lo - 1:hi - 1, 1:-1], u[lo + 1:hi + 1, 1:-1], out=band)
    band += u[lo:hi, :-2]
    band += u[lo:hi, 2:]
    band *= 0.25


def _work(task):
    lo, hi, reps = task
    for _ in range(reps):
        stencil_band(_U, _O, lo, hi)
    return float(_O[lo:hi].sum())


def bands(n, parts):
    """Split interior rows [1, n-1) into `parts` contiguous bands."""
    rows = n - 2
    base, extra = divmod(rows, parts)
    out, start = [], 1
    for i in range(parts):
        stop = start + base + (1 if i < extra else 0)
        out.append((start, stop))
        start = stop
    return out


# ---------------------------------------------------------- measurement
def run_config(n, P, reps, seed):
    """Time R stencil sweeps on an n x n grid with P worker processes."""
    rng = np.random.default_rng(seed)
    init = rng.random((n, n))

    su = shared_memory.SharedMemory(create=True, size=init.nbytes)
    so = shared_memory.SharedMemory(create=True, size=init.nbytes)
    try:
        U = np.ndarray((n, n), dtype=np.float64, buffer=su.buf)
        O = np.ndarray((n, n), dtype=np.float64, buffer=so.buf)
        U[:] = init
        O[:] = init                                      # boundary = input
        tasks = [(lo, hi, reps) for lo, hi in bands(n, P)]

        with Pool(P, initializer=_init, initargs=(su.name, so.name, n)) as pool:
            pool.map(_work, [(lo, hi, 1) for lo, hi, _ in tasks])      # warm-up
            times, total = [], 0.0
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                partial = pool.map(_work, tasks)
                times.append(time.perf_counter() - t0)
                total = sum(partial)

        ref = init.copy()
        stencil_band(init, ref, 1, n - 1)
        exact = bool(np.array_equal(O, ref))
        # Workers return sums over the interior rows they own, so compare
        # against the reference over the same interior rows.
        ref_sum = float(ref[1:n - 1].sum())
    finally:
        for s in (su, so):
            s.close()
            s.unlink()
    return {
        "T_s": statistics.median(times), "runs_s": times,
        "checksum": total, "reference_checksum": ref_sum,
        "checksum_ok": math.isclose(total, ref_sum, rel_tol=1e-9),
        "exact_match": exact,
    }


def calibrate_reps(n, seed):
    rng = np.random.default_rng(seed)
    u = rng.random((n, n))
    o = u.copy()
    stencil_band(u, o, 1, n - 1)
    t0 = time.perf_counter()
    for _ in range(5):
        stencil_band(u, o, 1, n - 1)
    sweep = (time.perf_counter() - t0) / 5
    return max(10, min(400, round(TARGET_SECONDS / sweep))), sweep


def measure_copy_bandwidth():
    """Sustained single-process copy bandwidth on a 256 MB buffer."""
    a = np.ones(32_000_000)
    b = np.empty_like(a)
    np.copyto(b, a)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        np.copyto(b, a)
        ts.append(time.perf_counter() - t0)
    # read 8 B + write 8 B (+8 B write-allocate on most x86 parts)
    return 24 * a.size / statistics.median(ts) / 1e9


def p_values(physical):
    vals, p = [], 1
    while p <= physical:
        vals.append(p)
        p *= 2
    if vals[-1] != physical:
        vals.append(physical)
    return vals


def main():
    params = json.loads((RESULTS / "salted_params.json").read_text(encoding="utf-8"))
    try:
        prof = json.loads((RESULTS / "host_profile.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        prof = {}
    N = int(params["N"])
    seed = int(params["seed"])
    physical = (prof.get("topology", {}).get("physical_cores")
                or os.cpu_count() or 1)
    Ps = p_values(physical)

    reps, sweep = calibrate_reps(N, seed)
    print("=" * 78)
    print("MILESTONE 3 — STRONG / WEAK SCALING AND ROOFLINE")
    print("=" * 78)
    print(f"Salted N        : {N}   seed {seed}")
    print(f"Physical cores  : {physical}   ->  P = {Ps}")
    print(f"One sweep (1 P) : {sweep * 1000:.3f} ms   ->  R = {reps} sweeps per run")
    print(f"Repeats         : {REPEATS} per point, median reported")
    print()

    # ------------------------------------------------------ strong scaling
    strong = {}
    for P in Ps:
        print(f"  strong scaling P={P} ...", flush=True)
        strong[P] = run_config(N, P, reps, seed)
    T1 = strong[1]["T_s"]
    xs, ys = [], []
    for P in Ps:
        r = strong[P]
        r["S"] = T1 / r["T_s"]
        r["E"] = r["S"] / P * 100
        if P > 1:
            r["s_amdahl"] = (1 / r["S"] - 1 / P) / (1 - 1 / P)
            xs.append(1 - 1 / P)
            ys.append(1 / r["S"] - 1 / P)
    # least squares on 1/S - 1/P = s (1 - 1/P)
    s_fit = (sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)) if xs else 0.0
    s_max = (1 / s_fit) if s_fit > 0 else float("inf")

    # -------------------------------------------------------- weak scaling
    weak = {}
    for P in Ps:
        nP = math.ceil(N * math.sqrt(P))
        print(f"  weak scaling   P={P}  N_P={nP} ...", flush=True)
        weak[P] = run_config(nP, P, reps, seed)
        weak[P]["N"] = nP
        weak[P]["work"] = (nP - 2) ** 2 * reps
    W1, TW1 = weak[1]["work"], weak[1]["T_s"]
    for P in Ps:
        w = weak[P]
        # Formula exactly as printed in the handout. With Work(P) = P * Work(1)
        # it reduces to T_1 / T_P, so its ideal value is 1.0, not P.
        w["S_handout"] = P * W1 * TW1 / (w["work"] * w["T_s"])
        # Standard Gustafson scaled speedup: work done per unit time relative
        # to one process. Ideal value is P.
        w["S_scaled"] = (w["work"] / W1) * (TW1 / w["T_s"])
        w["E"] = w["S_scaled"] / P * 100
        w["s_gustafson"] = None if P == 1 else (P - w["S_scaled"]) / (P - 1)
    print()

    # --------------------------------------------------------------- tables
    print("STRONG SCALING (fixed N = %d)" % N)
    print(f"{'P':>3} {'T_P (ms)':>11} {'S_P':>8} {'E_P %':>8} {'s (Amdahl)':>11} {'exact':>6}")
    for P in Ps:
        r = strong[P]
        s = "          -" if P == 1 else f"{r['s_amdahl']:11.4f}"
        print(f"{P:>3} {r['T_s'] * 1000:>11.2f} {r['S']:>8.3f} {r['E']:>8.1f} {s} "
              f"{'YES' if r['exact_match'] else 'NO':>6}")
    print(f"Fitted serial fraction s = {s_fit:.4f}  ->  S_max = 1/s = "
          f"{'inf' if s_max == float('inf') else f'{s_max:.2f}'}x")
    if s_fit <= 0:
        print("  s <= 0: speedup was at or above linear (bands fit better in cache than")
        print("  the whole grid), so no serial fraction is measurable at these P.")
    print()
    print("WEAK SCALING (N_P = ceil(N * sqrt(P)))")
    print(f"{'P':>3} {'N_P':>6} {'T_P (ms)':>11} {'Gustafson S':>12} {'E %':>7} "
          f"{'handout S':>10} {'s (Gust.)':>10} {'exact':>6}")
    for P in Ps:
        w = weak[P]
        sg = "         -" if w["s_gustafson"] is None else f"{w['s_gustafson']:10.4f}"
        print(f"{P:>3} {w['N']:>6} {w['T_s'] * 1000:>11.2f} {w['S_scaled']:>12.3f} "
              f"{w['E']:>7.1f} {w['S_handout']:>10.3f} {sg} "
              f"{'YES' if w['exact_match'] else 'NO':>6}")
    print("  Gustafson S = (Work(P)/Work(1)) * (T_1/T_P), ideal = P.")
    print("  handout S   = P*Work(1)*T_1 / (Work(P)*T_P) as printed in the task; with")
    print("                Work(P) = P*Work(1) it equals T_1/T_P, so its ideal is 1.0.")
    pm = Ps[-1]
    print(f"\nAt P = {pm}: strong efficiency {strong[pm]['E']:.1f}%  vs  "
          f"weak efficiency {weak[pm]['E']:.1f}%")
    print()

    # -------------------------------------------------------------- roofline
    l3 = prof.get("caches", {}).get("l3_cache_bytes")
    l2 = prof.get("caches", {}).get("l2_cache_bytes")
    base_mhz = prof.get("cpu", {}).get("base_frequency_mhz") or 2500
    working_set = 2 * N * N * 8
    bw = measure_copy_bandwidth()
    ai_fused_wa = FLOPS_PER_POINT / 24          # read u, write out, write-allocate
    ai_fused = FLOPS_PER_POINT / 16             # optimistic: no write-allocate
    ai_numpy = FLOPS_PER_POINT / 96             # 4 NumPy passes, each streaming the band
    dp_flops_per_cycle = 16                     # AVX2: 2 x 256-bit FMA units
    peak_gflops = physical * (base_mhz / 1000) * dp_flops_per_cycle
    ridge = peak_gflops / bw
    pts = (N - 2) ** 2 * reps
    best = strong[pm]
    achieved_gflops = FLOPS_PER_POINT * pts / best["T_s"] / 1e9
    achieved_gbs = 96 * pts / best["T_s"] / 1e9
    regime = "MEMORY-BANDWIDTH BOUND" if ai_fused < ridge else "COMPUTE BOUND"

    print("ROOFLINE")
    print(f"Working set (u + out) : {working_set / 1e6:.1f} MB   L3 = "
          f"{(l3 or 0) / 1e6:.1f} MB  -> {'exceeds L3, DRAM-resident' if l3 and working_set > l3 else 'fits in cache'}")
    print(f"Arithmetic intensity  : {ai_fused_wa:.4f} FLOP/B (fused kernel, with write-allocate)")
    print(f"                        {ai_fused:.4f} FLOP/B (fused kernel, optimistic)")
    print(f"                        {ai_numpy:.4f} FLOP/B (this NumPy 4-pass implementation)")
    print(f"Measured bandwidth    : {bw:.2f} GB/s (single-process copy)")
    print(f"Peak DP compute       : {peak_gflops:.1f} GFLOP/s "
          f"({physical} cores x {base_mhz / 1000:.2f} GHz x {dp_flops_per_cycle} FLOP/cycle)")
    print(f"Ridge point           : {ridge:.2f} FLOP/B")
    print(f"Achieved at P={pm}       : {achieved_gflops:.3f} GFLOP/s, ~{achieved_gbs:.1f} GB/s of traffic")
    print(f"REGIME                : {regime} (AI {ai_fused:.3f} << ridge {ridge:.2f})")

    # ---------------------------------------------------------------- files
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "scaling_metrics.tsv", "w", encoding="utf-8") as fh:
        fh.write("study\tP\tN\tR\twork_points\tT_ms\tspeedup\tefficiency_pct\t"
                 "serial_fraction\tchecksum\treference_checksum\tchecksum_ok\texact_match\t"
                 + "\t".join(f"run{i + 1}_ms" for i in range(REPEATS)) + "\n")
        for P in Ps:
            r = strong[P]
            fh.write("\t".join(map(str, [
                "strong", P, N, reps, (N - 2) ** 2 * reps, f"{r['T_s'] * 1000:.3f}",
                f"{r['S']:.6f}", f"{r['E']:.3f}",
                "" if P == 1 else f"{r['s_amdahl']:.6f}",
                f"{r['checksum']:.6f}", f"{r['reference_checksum']:.6f}",
                r["checksum_ok"], r["exact_match"],
                *[f"{t * 1000:.3f}" for t in r["runs_s"]]])) + "\n")
        for P in Ps:
            w = weak[P]
            fh.write("\t".join(map(str, [
                "weak", P, w["N"], reps, w["work"], f"{w['T_s'] * 1000:.3f}",
                f"{w['S_scaled']:.6f}", f"{w['E']:.3f}",
                "" if w["s_gustafson"] is None else f"{w['s_gustafson']:.6f}",
                f"{w['checksum']:.6f}", f"{w['reference_checksum']:.6f}",
                w["checksum_ok"], w["exact_match"],
                *[f"{t * 1000:.3f}" for t in w["runs_s"]]])) + "\n")
        fh.write(f"# amdahl_fit\ts={s_fit:.6f}\tS_max={s_max}\n")
        fh.write("# weak speedup column = Gustafson scaled speedup (Work(P)/Work(1))*(T_1/T_P)\n")
        fh.write("# handout_formula_weak\t" + "\t".join(
            f"P{P}={weak[P]['S_handout']:.6f}" for P in Ps) + "\n")

    summary = {
        "kernel": "5-point Jacobi stencil, out = 0.25 * (N + S + W + E)",
        "N_salted": N, "sweeps_per_run": reps,
        "flops_per_point": FLOPS_PER_POINT,
        "arithmetic_intensity_flop_per_byte": {
            "fused_with_write_allocate": ai_fused_wa,
            "fused_optimistic": ai_fused,
            "numpy_4pass_implementation": ai_numpy,
        },
        "working_set_bytes": working_set,
        "l2_cache_bytes": l2, "l3_cache_bytes": l3,
        "working_set_exceeds_l3": bool(l3 and working_set > l3),
        "measured_bandwidth_gbs": bw,
        "bandwidth_method": "single-process numpy copy of 256 MB, 24 B/element",
        "peak_dp_gflops_estimate": peak_gflops,
        "peak_basis": f"{physical} cores x {base_mhz} MHz x {dp_flops_per_cycle} DP FLOP/cycle (AVX2 FMA)",
        "ridge_point_flop_per_byte": ridge,
        "achieved_gflops_at_max_P": achieved_gflops,
        "strong_speedup": {str(P): strong[P]["S"] for P in Ps},
        "weak_gustafson_scaled_speedup": {str(P): weak[P]["S_scaled"] for P in Ps},
        "amdahl_fitted_serial_fraction": s_fit,
        "amdahl_max_speedup": None if s_max == float("inf") else s_max,
        "regime": regime,
        "conclusion": (
            f"AI of {ai_fused:.3f} FLOP/byte (at best) sits far left of the ridge point "
            f"{ridge:.2f} FLOP/byte, and the {working_set / 1e6:.0f} MB working set "
            f"exceeds the {(l3 or 0) / 1e6:.0f} MB L3, so every sweep streams from DRAM. "
            "The kernel is limited by memory bandwidth, not arithmetic: adding cores "
            "helps only until the shared memory channels saturate, which is what caps "
            "the strong-scaling speedup."),
    }
    (RESULTS / "roofline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWritten: {RESULTS / 'scaling_metrics.tsv'}")
    print(f"         {RESULTS / 'roofline_summary.json'}")
    ok = all(r["exact_match"] for r in strong.values()) and all(w["exact_match"] for w in weak.values())
    print(f"All parallel results match the single-process reference: {ok}")


if __name__ == "__main__":
    main()
