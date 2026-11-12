import multiprocessing as mp
import platform
import sys
import time
from harness import emit_verify_tag
import labutil as L

if __name__ == "__main__":
    sid = L.get_id()
    seed = L.seed_of(sid)
    plog, pphys = mp.cpu_count(), L.detect_pphys()
    ci, model = L.cache_info(), L.cpu_model()

    n = 10**6
    pc = time.perf_counter_ns
    pc()
    t0 = pc()
    for _ in range(n):
        pc()
    avg_ns = (pc() - t0) / n

    tag = emit_verify_tag("PHASE0", seed, avg_ns, float(plog))
    hw = L.md_table(["Item", "Value"], [
        ["CPU Model", model], ["Physical cores (Pphys)", pphys], ["Logical threads (Plog)", plog],
        ["Cache line (Lbytes)", ci["line"]], ["L1d", L.fmt_bytes(ci["l1d"])],
        ["L2", L.fmt_bytes(ci["l2"])], ["L3", L.fmt_bytes(ci["l3"])],
        ["OS / Python", f"{platform.platform()} / {platform.python_version()}"],
        ["perf_counter_ns call latency", f"{avg_ns:.1f} ns (avg over 10^6 calls)"],
        ["Student ID / seed", f"{sid} / {seed}"],
    ])
    md = f"## Phase 0: Hardware Topology & Telemetry\n\n{hw}\n\n```\n{tag}\n```\n"
    L.save_phase(0, md, {"hash": tag, "pphys": pphys, "plog": plog, "avg_ns": avg_ns})
