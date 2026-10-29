"""
Milestone 2 — False sharing collapse, remediation and Karp-Flatt diagnosis.

Usage:
    py src/false_sharing.py

P processes (P in 1, 2, 4, 8) each increment their own counter K = 5,000,000
times in a shared multiprocessing array of 64-bit integers:

  UNPADDED    worker p writes arr[p]. Counters 0..7 occupy one 64-byte line,
              so every write by one core invalidates that line in every
              other core (MESI) and the line ping-pongs between them.
  PADDED      worker p writes arr[p * 8]: 8 x 8 bytes = 64 bytes apart, so
              each counter owns a physically separate cache line.
  PRIVATIZED  worker p counts in a local variable and performs one single
              synchronized write to the shared array at the end.

Measurement details that matter:
  * mp.Array is created with lock=False. With the default lock=True every
    `arr[i] += 1` takes one shared lock, and the experiment would measure
    lock contention instead of cache-line contention.
  * The array base is aligned to a 64-byte boundary, so the 8 unpadded
    counters really are on ONE line rather than straddling two.
  * Process start-up is excluded: all workers wait on a barrier, then each
    records its own loop start/end. T_P = last end - first start.
  * Work grows with P (each worker always does K increments), so the
    serial time for the same total work is P * T_1 and
        S_P = P * T_1 / T_P,
        e(P) = (1/S_P - 1/P) / (1 - 1/P)          (Karp-Flatt)
    Ideal scaling gives T_P = T_1, S_P = P and e = 0.

Writes results/false_sharing_karp_flatt.tsv
"""
import ctypes
import json
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
K = int(os.environ.get("LAB_K", 5_000_000))
PROCS = [1, 2, 4, 8]
REPEATS = 3
LINE = 64
INT64 = 8
STRIDE = LINE // INT64              # 8 int64 per cache line


# ------------------------------------------------------------- the workers
def worker_counter(arr, idx, iterations, barrier, out_q):
    """Unpadded and padded both use this; only `idx` differs."""
    barrier.wait()
    t0 = time.perf_counter()
    for _ in range(iterations):
        arr[idx] += 1
    t1 = time.perf_counter()
    out_q.put((t0, t1))


def worker_private(arr, idx, iterations, barrier, out_q, lock):
    barrier.wait()
    t0 = time.perf_counter()
    local = 0                        # lives in this process's own frame
    for _ in range(iterations):
        local += 1
    with lock:                       # the single synchronized write
        arr[idx] += local
    t1 = time.perf_counter()
    out_q.put((t0, t1))


# -------------------------------------------------------------- one run
def aligned_base(arr):
    """First index whose address sits exactly on a cache-line boundary."""
    addr = ctypes.addressof(arr)
    for i in range(STRIDE):
        if (addr + i * INT64) % LINE == 0:
            return i
    return 0


def run_once(variant, P):
    arr = mp.Array(ctypes.c_int64, STRIDE * 8 + 2 * STRIDE, lock=False)
    base = aligned_base(arr)
    barrier = mp.Barrier(P + 1)
    out_q = mp.Queue()
    lock = mp.Lock()

    procs = []
    for p in range(P):
        if variant == "unpadded":
            idx, target, extra = base + p, worker_counter, ()
        elif variant == "padded":
            idx, target, extra = base + p * STRIDE, worker_counter, ()
        else:
            idx, target, extra = base + p, worker_private, (lock,)
        procs.append(mp.Process(target=target,
                                args=(arr, idx, K, barrier, out_q) + extra))
    for pr in procs:
        pr.start()
    barrier.wait()                                   # everyone starts together
    stamps = [out_q.get() for _ in range(P)]
    for pr in procs:
        pr.join()

    t_ms = (max(s[1] for s in stamps) - min(s[0] for s in stamps)) * 1000.0
    if variant == "padded":
        total = sum(arr[base + p * STRIDE] for p in range(P))
    else:
        total = sum(arr[base + p] for p in range(P))
    addr_ok = (ctypes.addressof(arr) + base * INT64) % LINE == 0
    return t_ms, total, addr_ok


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    variants = ["unpadded", "padded", "privatized"]

    print("=" * 78)
    print("MILESTONE 2 — FALSE SHARING, REMEDIATION AND KARP-FLATT")
    print("=" * 78)
    print(f"K per worker : {K:,}")
    print(f"Processes    : {PROCS}   (logical CPUs on this host: {os.cpu_count()})")
    print(f"Repeats      : {REPEATS} per point, median reported")
    print(f"Shared array : mp.Array(c_int64, lock=False), base aligned to {LINE} bytes")
    print()

    data = {}
    for v in variants:
        for P in PROCS:
            print(f"  running {v:<10} P={P} ...", flush=True)
            runs = [run_once(v, P) for _ in range(REPEATS)]
            times = [r[0] for r in runs]
            data[(v, P)] = {
                "T_ms": statistics.median(times),
                "runs_ms": times,
                "total": runs[-1][1],
                "aligned": all(r[2] for r in runs),
            }
    print()

    rows = []
    for v in variants:
        T1 = data[(v, 1)]["T_ms"]
        for P in PROCS:
            d = data[(v, P)]
            S = P * T1 / d["T_ms"]
            e = None if P == 1 else (1 / S - 1 / P) / (1 - 1 / P)
            rows.append({
                "variant": v, "P": P, "K": K,
                "T_P_ms": d["T_ms"], "T_1_ms": T1,
                "slowdown_TP_over_T1": d["T_ms"] / T1,
                "speedup_S_P": S, "efficiency_pct": S / P * 100,
                "karp_flatt_e": e,
                "total_count": d["total"], "expected": P * K,
                "exact": d["total"] == P * K,
                "base_aligned_64B": d["aligned"],
                "runs_ms": d["runs_ms"],
            })

    pad = {r["P"]: r["T_P_ms"] for r in rows if r["variant"] == "padded"}
    for r in rows:
        r["ratio_vs_padded_same_P"] = r["T_P_ms"] / pad[r["P"]]

    hdr = (f"{'variant':<11}{'P':>3} {'T_P (ms)':>11} {'T_P/T_1':>8} {'S_P':>7} "
           f"{'eff %':>7} {'e(P)':>9} {'vs padded':>10} {'count ok':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        e = "      -" if r["karp_flatt_e"] is None else f"{r['karp_flatt_e']:9.4f}"
        print(f"{r['variant']:<11}{r['P']:>3} {r['T_P_ms']:>11.2f} "
              f"{r['slowdown_TP_over_T1']:>7.2f}x {r['speedup_S_P']:>7.3f} "
              f"{r['efficiency_pct']:>6.1f}% {e} {r['ratio_vs_padded_same_P']:>9.2f}x "
              f"{'YES' if r['exact'] else 'NO':>9}")
    print("-" * len(hdr))

    pmax = PROCS[-1]
    get = lambda v, P: next(r for r in rows if r["variant"] == v and r["P"] == P)  # noqa: E731
    u, p_, pr = get("unpadded", pmax), get("padded", pmax), get("privatized", pmax)
    e_u = [get("unpadded", P)["karp_flatt_e"] for P in PROCS[1:]]
    e_p = [get("padded", P)["karp_flatt_e"] for P in PROCS[1:]]

    print(f"\nAt P = {pmax}:")
    print(f"  unpadded   runs {u['slowdown_TP_over_T1']:.2f}x slower than its own P=1, "
          f"e = {u['karp_flatt_e']:.4f}")
    print(f"  padded     runs {p_['slowdown_TP_over_T1']:.2f}x its P=1 time,          "
          f"e = {p_['karp_flatt_e']:.4f}")
    print(f"  privatized runs {pr['slowdown_TP_over_T1']:.2f}x its P=1 time,          "
          f"e = {pr['karp_flatt_e']:.4f}")
    print(f"  FALSE-SHARING PENALTY (unpadded / padded, same P): "
          f"{u['ratio_vs_padded_same_P']:.2f}x")
    print(f"\nKarp-Flatt trend over P = {PROCS[1:]}:")
    print(f"  unpadded : {[round(x, 4) for x in e_u]}")
    print(f"  padded   : {[round(x, 4) for x in e_p]}")
    print("A rising e(P) means the serial / contended fraction GROWS as cores are")
    print("added: that is coherence traffic, not a fixed serial section.")
    gaps = [a - b for a, b in zip(e_u, e_p)]
    if all(g > 0 for g in gaps):
        print(f"Unpadded e(P) is above padded e(P) at every P (gap "
              f"{min(gaps):.3f}..{max(gaps):.3f}): the extra contended fraction is the")
        print("cache line ping-ponging between cores, and padding removes it.")
    else:
        print("Unpadded and padded e(P) are not separated at every P here; use the")
        print("'vs padded' column, which compares both layouts at the same P.")
    try:
        prof = json.loads((RESULTS / "host_profile.json").read_text(encoding="utf-8"))
        hybrid = prof.get("topology", {}).get("hybrid_note")
    except Exception:                                           # noqa: BLE001
        hybrid = None
    if hybrid:
        print("Note: this CPU mixes P-cores and slower E-cores and lowers its turbo")
        print("clock as more cores wake, so even padded T_P drifts up with P. The")
        print("'vs padded' column compares both layouts at the SAME P, which cancels")
        print("those effects and isolates the cache-line contention alone.")
    print(f"\nAll counters exact (P x K): {all(r['exact'] for r in rows)}")

    cols = ["variant", "P", "K", "T_P_ms", "T_1_ms", "slowdown_TP_over_T1",
            "speedup_S_P", "efficiency_pct", "karp_flatt_e",
            "ratio_vs_padded_same_P", "total_count", "expected", "exact",
            "base_aligned_64B"]
    with open(RESULTS / "false_sharing_karp_flatt.tsv", "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols + [f"run{i + 1}_ms" for i in range(REPEATS)]) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                x = r[c]
                vals.append("" if x is None else (f"{x:.6f}" if isinstance(x, float) else str(x)))
            fh.write("\t".join(vals + [f"{t:.3f}" for t in r["runs_ms"]]) + "\n")
    summary = {"K": K, "P_values": PROCS,
               "false_sharing_penalty_at_Pmax": u["ratio_vs_padded_same_P"],
               "karp_flatt_unpadded": e_u, "karp_flatt_padded": e_p,
               "all_counts_exact": all(r["exact"] for r in rows)}
    (RESULTS / "false_sharing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWritten: {RESULTS / 'false_sharing_karp_flatt.tsv'}")


if __name__ == "__main__":
    main()
