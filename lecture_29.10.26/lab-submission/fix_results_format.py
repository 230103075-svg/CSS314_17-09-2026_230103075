"""
One-time format fix for results/ — makes existing results readable by the
instructor's verify_outputs.py WITHOUT re-running any benchmark.

Usage (from the lab-submission folder):
    py fix_results_format.py

Nothing is re-measured and no value is changed. The script only adds the
field names verify_outputs.py looks for, copied from values already in
the files, and reshapes the Milestone 2 table to one row per P:

  host_profile.json            + top-level cpu_model, physical_cores,
                                 logical_threads, cache_line_bytes, ...
  stride_results.tsv           + latency_ns column (= latency per element)
  loop_interchange.json        + naive_time_s, interchanged_time_s
  false_sharing_karp_flatt.tsv   one row per P with e_unpadded / e_padded /
                                 e_privatized; the original per-run table
                                 is kept as false_sharing_detail.tsv
  roofline_summary.json        + arithmetic_intensity

Safe to run more than once: anything already converted is left alone.
"""
import csv
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def save(name, obj):
    (R / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_tsv(name):
    with open(R / name, encoding="utf-8") as fh:
        lines = [l for l in fh if not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def write_tsv(name, cols, rows):
    with open(R / name, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])


# ------------------------------------------------------------------- M0
prof = load("host_profile.json")
if "cpu_model" not in prof:
    top = {
        "cpu_model": prof["cpu"]["model_name"],
        "microarchitecture": prof["cpu"].get("microarchitecture"),
        "base_frequency_mhz": prof["cpu"].get("base_frequency_mhz"),
        "physical_cores": prof["topology"]["physical_cores"],
        "logical_threads": prof["topology"]["logical_threads"],
        "smt_enabled": prof["topology"].get("smt_enabled"),
        "l1d_cache_bytes": prof["caches"].get("l1d_cache_bytes"),
        "l2_cache_bytes": prof["caches"].get("l2_cache_bytes"),
        "l3_cache_bytes": prof["caches"].get("l3_cache_bytes"),
        "cache_line_bytes": int(prof["caches"]["cache_line_bytes"]),
    }
    save("host_profile.json", {**top, **prof})
    print("host_profile.json        : added top-level topology keys")
else:
    print("host_profile.json        : already in verifier format")

# ------------------------------------------------------------------- M1
with open(R / "stride_results.tsv", encoding="utf-8") as fh:
    header = fh.readline().rstrip("\n").split("\t")
if "latency_ns" not in header:
    rows = read_tsv("stride_results.tsv")
    for r in rows:
        r["latency_ns"] = r["latency_ns_per_element"]
    i = header.index("latency_ns_per_element") + 1
    write_tsv("stride_results.tsv", header[:i] + ["latency_ns"] + header[i:], rows)
    print("stride_results.tsv       : added latency_ns column")
else:
    print("stride_results.tsv       : already in verifier format")

li = load("loop_interchange.json")
if "naive_time_s" not in li:
    li["naive_time_s"] = li["t_naive_ijk_s"]
    li["interchanged_time_s"] = li["t_interchanged_ikj_s"]
    save("loop_interchange.json", li)
    print("loop_interchange.json    : added naive_time_s / interchanged_time_s")
else:
    print("loop_interchange.json    : already in verifier format")

# ------------------------------------------------------------------- M2
with open(R / "false_sharing_karp_flatt.tsv", encoding="utf-8") as fh:
    header = fh.readline().rstrip("\n").split("\t")
if "variant" in header:
    detail = read_tsv("false_sharing_karp_flatt.tsv")
    write_tsv("false_sharing_detail.tsv", header, detail)
    by = {(r["variant"], int(r["P"])): r for r in detail}
    ps = sorted({int(r["P"]) for r in detail})
    wide = []
    for P in ps:
        u, p, v = by[("unpadded", P)], by[("padded", P)], by[("privatized", P)]
        wide.append({
            "P": P, "K": u["K"],
            "T_unpadded_ms": u["T_P_ms"], "T_padded_ms": p["T_P_ms"],
            "T_privatized_ms": v["T_P_ms"],
            "S_unpadded": u["speedup_S_P"], "S_padded": p["speedup_S_P"],
            "S_privatized": v["speedup_S_P"],
            "e_unpadded": u["karp_flatt_e"], "e_padded": p["karp_flatt_e"],
            "e_privatized": v["karp_flatt_e"],
            "penalty_unpadded_vs_padded": u["ratio_vs_padded_same_P"],
            "count_unpadded": u["total_count"], "count_padded": p["total_count"],
            "count_privatized": v["total_count"], "expected": u["expected"],
            "all_exact": all(x["exact"] == "True" for x in (u, p, v)),
        })
    cols = ["P", "K", "T_unpadded_ms", "T_padded_ms", "T_privatized_ms",
            "S_unpadded", "S_padded", "S_privatized",
            "e_unpadded", "e_padded", "e_privatized",
            "penalty_unpadded_vs_padded", "count_unpadded", "count_padded",
            "count_privatized", "expected", "all_exact"]
    write_tsv("false_sharing_karp_flatt.tsv", cols, wide)
    print("false_sharing_karp_flatt : reshaped to one row per P "
          "(per-run detail kept in false_sharing_detail.tsv)")
else:
    print("false_sharing_karp_flatt : already in verifier format")

# ------------------------------------------------------------------- M3
roof = load("roofline_summary.json")
if "arithmetic_intensity" not in roof:
    ai = roof["arithmetic_intensity_flop_per_byte"]["fused_optimistic"]
    roof = {"arithmetic_intensity": ai,
            "arithmetic_intensity_basis": (
                "4 FLOPs per point / 16 DRAM bytes (one 8-byte read + one 8-byte "
                "write, ideal reuse). This is the upper bound; with write-allocate "
                "it is 0.167 and for the 4-pass NumPy implementation about 0.042, "
                "see arithmetic_intensity_flop_per_byte."),
            **roof}
    save("roofline_summary.json", roof)
    print("roofline_summary.json    : added arithmetic_intensity")
else:
    print("roofline_summary.json    : already in verifier format")

print("\nDone. Now run:  py verify_outputs.py")
