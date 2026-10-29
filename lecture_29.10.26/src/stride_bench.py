"""
Milestone 1, Task 1.1 — Stride latency cliff.

Usage:
    py src/stride_bench.py

A flat float64 buffer of 2^24 elements (~134 MB, far beyond L3) is read
with stride s in {1, 2, 4, 8, 16, 32, 64, 128}, accumulating
total += data[i] for i = 0, s, 2s, ...

5 warm-up iterations, then 5 recorded iterations; the median time_ns is
kept. Latency per element = elapsed ns / elements touched.

A cache line is 64 bytes = 8 doubles. At stride 1 one line fill serves 8
reads; from stride 8 upward every read needs its own line, so the cost
per element jumps. That jump is the cliff this file has to show.

Writes results/stride_results.tsv
"""
import json
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
STRIDES = [1, 2, 4, 8, 16, 32, 64, 128]
WARMUP = 5
RECORDED = 5
ITEM = 8  # bytes per float64


def load_params():
    p = RESULTS / "salted_params.json"
    if not p.exists():
        raise SystemExit("results/salted_params.json missing — run src/profile_env.py first")
    return json.loads(p.read_text(encoding="utf-8"))


def cache_line_bytes():
    try:
        prof = json.loads((RESULTS / "host_profile.json").read_text(encoding="utf-8"))
        return int(prof["caches"]["cache_line_bytes"]) or 64
    except Exception:                                       # noqa: BLE001
        return 64


def main():
    params = load_params()
    n = int(params["buffer_elements"])
    line = cache_line_bytes()
    per_line = line // ITEM

    # Allocate flat array in physical memory
    data = np.ones(n, dtype=np.float64)

    print("=" * 78)
    print("MILESTONE 1.1 — STRIDE LATENCY CLIFF")
    print("=" * 78)
    print(f"Buffer      : {n:,} float64 = {n * ITEM / 1e6:.1f} MB")
    print(f"Cache line  : {line} bytes = {per_line} doubles")
    print(f"Iterations  : {WARMUP} warm-up + {RECORDED} recorded, median reported")
    print()

    rows = []
    for s in STRIDES:
        view = data[::s]
        touched = view.size
        for _ in range(WARMUP):
            view.sum()
        times = []
        total = 0.0
        for _ in range(RECORDED):
            t0 = time.perf_counter_ns()
            total = view.sum()                 # total += data[i], step s
            times.append(time.perf_counter_ns() - t0)
        med = statistics.median(times)
        lines_touched = (n * ITEM // line) if s <= per_line else touched
        rows.append({
            "stride": s,
            "stride_bytes": s * ITEM,
            "elements_touched": touched,
            "cache_lines_touched": lines_touched,
            "useful_bytes_per_line": min(per_line, max(1, per_line // s)) * ITEM,
            "median_time_ns": int(med),
            "latency_ns_per_element": med / touched,
            "ns_per_cache_line": med / lines_touched,
            "runs_ns": times,
            "checksum": float(total),
            "checksum_ok": float(total) == float(touched),
        })

    base = rows[0]["latency_ns_per_element"]
    header = (f"{'stride':>6} {'bytes':>6} {'elements':>12} {'median ms':>10} "
              f"{'ns/elem':>9} {'vs s=1':>8} {'useful B/line':>14}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['stride']:>6} {r['stride_bytes']:>6} {r['elements_touched']:>12,} "
              f"{r['median_time_ns'] / 1e6:>10.3f} {r['latency_ns_per_element']:>9.4f} "
              f"{r['latency_ns_per_element'] / base:>7.2f}x {r['useful_bytes_per_line']:>10}/{line}")
    print("-" * len(header))

    cliff = next(r for r in rows if r["stride"] == per_line)
    print(f"\nCACHE-LINE CLIFF: stride 1 -> stride {per_line}: "
          f"{base:.4f} -> {cliff['latency_ns_per_element']:.4f} ns/element "
          f"({cliff['latency_ns_per_element'] / base:.2f}x)")
    print(f"At stride >= {per_line} every read needs its own {line}-byte line, so "
          f"{line - ITEM} of every {line} bytes fetched are wasted.")
    print(f"All checksums exact: {all(r['checksum_ok'] for r in rows)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    cols = ["stride", "stride_bytes", "elements_touched", "cache_lines_touched",
            "useful_bytes_per_line", "median_time_ns", "latency_ns_per_element",
            "ns_per_cache_line", "checksum", "checksum_ok"]
    with open(RESULTS / "stride_results.tsv", "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols + [f"run{i + 1}_ns" for i in range(RECORDED)]) + "\n")
        for r in rows:
            vals = [r[c] if not isinstance(r[c], float) else f"{r[c]:.6f}" for c in cols]
            fh.write("\t".join(map(str, vals + r["runs_ns"])) + "\n")
    print(f"\nWritten: {RESULTS / 'stride_results.tsv'}")


if __name__ == "__main__":
    main()
