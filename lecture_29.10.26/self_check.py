"""
Self-check of every milestone's numerical results.

Usage (from the lab-submission folder):
    py self_check.py

This is NOT the instructor's verify_outputs.py and it emits no receipt or
signature. It re-checks the same conditions the handout lists for
Milestone 4 so problems are caught before submission:

  M0  host_profile.json and salted_params.json exist; params match salt.py
  M1  stride results present with exact checksums; i-k-j matches i-j-k
      within 1e-12
  M2  every unpadded / padded / privatized run ended with exactly P x K
  M3  every parallel stencil result matched the single-process reference
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

ok_all = True


def report(name, ok, detail):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def tsv(path):
    with open(path, encoding="utf-8") as fh:
        return [r for r in csv.DictReader((l for l in fh if not l.startswith("#")), delimiter="\t")]


# ---------------------------------------------------------------- M0
try:
    from salt import get_student_params
    prof = json.loads((R / "host_profile.json").read_text(encoding="utf-8"))
    params = json.loads((R / "salted_params.json").read_text(encoding="utf-8"))
    expect = get_student_params(params["student_id"])
    c = prof["caches"]
    report("Milestone 0", params == expect and c["cache_line_bytes"] and c["l3_cache_bytes"],
           f"{prof['cpu']['model_name']}, N={params['N']}, seed={params['seed']}, "
           f"line={c['cache_line_bytes']} B")
except Exception as exc:                                        # noqa: BLE001
    report("Milestone 0", False, exc)

# ---------------------------------------------------------------- M1
try:
    rows = tsv(R / "stride_results.tsv")
    chk = all(r["checksum_ok"] == "True" for r in rows)
    li = json.loads((R / "loop_interchange.json").read_text(encoding="utf-8"))
    A = np.load(R / "dense_C_naive.npy")
    B = np.load(R / "dense_C_interchanged.npy")
    diff = float(np.max(np.abs(A - B)))
    report("Milestone 1", chk and diff <= 1e-12 and len(rows) == 8,
           f"{len(rows)} strides, checksums exact={chk}; max|ijk-ikj|={diff:.3e}, "
           f"speedup {li['speedup']:.2f}x")
except Exception as exc:                                        # noqa: BLE001
    report("Milestone 1", False, exc)

# ---------------------------------------------------------------- M2
try:
    rows = tsv(R / "false_sharing_karp_flatt.tsv")
    exact = all(int(r["total_count"]) == int(r["expected"]) for r in rows)
    biggest = max(rows, key=lambda r: int(r["expected"]))
    report("Milestone 2", exact and len(rows) == 12,
           f"{len(rows)} runs, all counters == P x K: {exact} "
           f"(largest {int(biggest['total_count']):,} == {int(biggest['expected']):,})")
except Exception as exc:                                        # noqa: BLE001
    report("Milestone 2", False, exc)

# ---------------------------------------------------------------- M3
try:
    rows = tsv(R / "scaling_metrics.tsv")
    exact = all(r["exact_match"] == "True" and r["checksum_ok"] == "True" for r in rows)
    roof = json.loads((R / "roofline_summary.json").read_text(encoding="utf-8"))
    report("Milestone 3", exact and roof.get("regime"),
           f"{len(rows)} strong+weak runs match reference: {exact}; regime: {roof.get('regime')}")
except Exception as exc:                                        # noqa: BLE001
    report("Milestone 3", False, exc)

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
sys.exit(0 if ok_all else 1)
