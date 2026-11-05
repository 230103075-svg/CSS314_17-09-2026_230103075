"""
Task 22/23 — Empirical weak-scaling benchmark (Python multiprocessing).

Usage (from the repository root):
    py benchmarks/weak_scale_bench/bench.py

Workload: midpoint-rule numerical integration of
    f(x) = sin(x) * exp(-x / 3) + sqrt(x + 1)
Each worker integrates its own slice of K = 6,000,000 points, processed in
blocks of 65,536 so the working set stays in cache (a compute-bound kernel).
Weak scaling: K per worker is constant, so P workers do P times the work.

P runs from 1 to the host's hardware concurrency (os.cpu_count()).
Each P is timed over RUNS independent runs on a pool that was spawned and
warmed beforehand. Each worker also times its own computation, to expose
frequency drop / contention separately from dispatch overhead.

Writes:
    benchmarks/results.csv       one row per (P, run) with raw timings
    benchmarks/telemetry.json    CPU, topology, caches, memory modules, OS
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import csv                          # noqa: E402
import ctypes                       # noqa: E402
import json                         # noqa: E402
import multiprocessing as mp        # noqa: E402
import platform                     # noqa: E402
import re                           # noqa: E402
import subprocess                   # noqa: E402
import time                         # noqa: E402
from pathlib import Path            # noqa: E402

import numpy as np                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "benchmarks"
K = 6_000_000          # points per worker (constant: weak scaling)
BLOCK = 65_536         # cache-resident block
RUNS = 5
H = 1e-6               # grid spacing


def integrate(worker_id):
    """Midpoint rule over this worker's slice; returns (value, compute_seconds)."""
    t0 = time.perf_counter()
    start = worker_id * K
    total = 0.0
    for b in range(0, K, BLOCK):
        n = min(BLOCK, K - b)
        x = (np.arange(start + b, start + b + n, dtype=np.float64) + 0.5) * H
        total += float(np.sum(np.sin(x) * np.exp(-x / 3.0) + np.sqrt(x + 1.0)))
    return total * H, time.perf_counter() - t0


# ------------------------------------------------------------------ telemetry
INTEL6 = {154: "Alder Lake (P/H mobile)", 151: "Alder Lake (S desktop)",
          183: "Raptor Lake", 186: "Raptor Lake", 191: "Raptor Lake", 170: "Meteor Lake",
          140: "Tiger Lake", 141: "Tiger Lake", 125: "Ice Lake", 126: "Ice Lake",
          165: "Comet Lake", 142: "Kaby/Coffee Lake", 158: "Kaby/Coffee Lake",
          85: "Skylake-SP / Cascade Lake", 106: "Ice Lake-SP", 143: "Sapphire Rapids",
          207: "Emerald Rapids"}


def _windows():
    info = {}
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        info["cpu_model"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        info["vendor"] = winreg.QueryValueEx(k, "VendorIdentifier")[0].strip()
        info["base_mhz"] = int(winreg.QueryValueEx(k, "~MHz")[0])
        m = re.search(r"Family (\d+) Model (\d+)", winreg.QueryValueEx(k, "Identifier")[0])
        if m:
            info["family"], info["model"] = map(int, m.groups())
    except Exception as exc:                                    # noqa: BLE001
        info["registry_error"] = str(exc)

    class CD(ctypes.Structure):
        _fields_ = [("Level", ctypes.c_ubyte), ("Assoc", ctypes.c_ubyte),
                    ("Line", ctypes.c_ushort), ("Size", ctypes.c_uint32), ("Type", ctypes.c_int)]

    class U(ctypes.Union):
        _fields_ = [("Flags", ctypes.c_ubyte), ("Node", ctypes.c_uint32),
                    ("Cache", CD), ("R", ctypes.c_ulonglong * 2)]

    class SLPI(ctypes.Structure):
        _fields_ = [("Mask", ctypes.c_size_t), ("Rel", ctypes.c_int), ("u", U)]

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        n = ctypes.c_uint32(0)
        k32.GetLogicalProcessorInformation(None, ctypes.byref(n))
        buf = (SLPI * (n.value // ctypes.sizeof(SLPI)))()
        k32.GetLogicalProcessorInformation(buf, ctypes.byref(n))
        kinds = {0: "Unified", 1: "Instruction", 2: "Data", 3: "Trace"}
        caches, smt, packages, numa = [], 0, 0, 0
        for e in buf:
            if e.Rel == 0 and e.u.Flags & 1:
                smt += 1
            elif e.Rel == 1:
                numa += 1
            elif e.Rel == 3:
                packages += 1
            elif e.Rel == 2:
                c = e.u.Cache
                caches.append({"level": int(c.Level), "type": kinds.get(int(c.Type), "?"),
                               "size_bytes": int(c.Size), "line_bytes": int(c.Line),
                               "logical_cpus_sharing": bin(e.Mask).count("1")})
        info.update(caches=caches, cores_with_smt=smt, packages=packages, numa_nodes=numa)
    except Exception as exc:                                    # noqa: BLE001
        info["topology_error"] = str(exc)

    try:
        ps = ("Get-CimInstance Win32_PhysicalMemory | "
              "Select-Object Capacity,Speed,ConfiguredClockSpeed,DeviceLocator,BankLabel | "
              "ConvertTo-Json -Compress")
        raw = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        mods = json.loads(raw) if raw else []
        info["memory_modules"] = mods if isinstance(mods, list) else [mods]
    except Exception as exc:                                    # noqa: BLE001
        info["memory_error"] = str(exc)
    return info


def _linux():
    info = {}
    try:
        first = open("/proc/cpuinfo", encoding="utf-8").read().split("\n\n")[0]
        f = {a.strip(): b.strip() for a, b in
             (l.split(":", 1) for l in first.splitlines() if ":" in l)}
        info.update(cpu_model=f.get("model name"), vendor=f.get("vendor_id"),
                    family=int(f.get("cpu family", 0)), model=int(f.get("model", 0)))
    except Exception as exc:                                    # noqa: BLE001
        info["cpuinfo_error"] = str(exc)
    caches, seen = [], set()
    for idx in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cache/index*")):
        try:
            rd = lambda n: (idx / n).read_text().strip()        # noqa: E731
            key = (rd("level"), rd("type"), rd("size"), rd("shared_cpu_list"))
            if key in seen:
                continue
            seen.add(key)
            caches.append({"level": int(rd("level")), "type": rd("type"),
                           "size_bytes": int(re.match(r"\d+", rd("size")).group()) * 1024,
                           "line_bytes": int(rd("coherency_line_size")),
                           "logical_cpus_sharing": rd("shared_cpu_list")})
        except Exception:                                       # noqa: BLE001
            continue
    info["caches"] = caches
    info["numa_nodes"] = len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))) or 1
    return info


def telemetry():
    raw = _windows() if platform.system() == "Windows" else _linux()
    logical = os.cpu_count()
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        mem_total = psutil.virtual_memory().total
    except Exception:                                           # noqa: BLE001
        physical, mem_total = logical, None
    fam, mod = raw.get("family"), raw.get("model")
    micro = (INTEL6.get(mod, f"Intel family 6 model {mod}")
             if "intel" in str(raw.get("vendor", "")).lower() and fam == 6
             else f"family {fam} model {mod}")
    caches = raw.get("caches", [])

    def biggest(level, kinds):
        s = [c["size_bytes"] for c in caches if c["level"] == level and c["type"] in kinds]
        return max(s) if s else None

    smt = raw.get("cores_with_smt")
    return {
        "cpu_model": raw.get("cpu_model") or platform.processor(),
        "microarchitecture": micro,
        "base_frequency_mhz": raw.get("base_mhz"),
        "physical_cores": physical,
        "logical_threads": logical,
        "cores_with_smt": smt,
        "hybrid_topology": bool(smt and physical and 0 < smt < physical),
        "packages": raw.get("packages"),
        "numa_nodes": raw.get("numa_nodes"),
        "l1d_bytes": biggest(1, ("Data",)),
        "l2_bytes": biggest(2, ("Unified", "Data")),
        "l3_bytes": biggest(3, ("Unified", "Data")),
        "cache_line_bytes": min((c["line_bytes"] for c in caches if c.get("line_bytes")), default=64),
        "cache_detail": sorted({(c["level"], c["type"], c["size_bytes"]) for c in caches}),
        "memory_total_bytes": mem_total,
        "memory_modules": raw.get("memory_modules"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "benchmark": {"K_points_per_worker": K, "block": BLOCK, "runs": RUNS, "h": H},
        "errors": {k: v for k, v in raw.items() if k.endswith("_error")},
    }


def main():
    tel = telemetry()
    pmax = os.cpu_count() or 1
    Ps = list(range(1, pmax + 1))

    print("=" * 70)
    print("WEAK-SCALING BENCHMARK (Gustafson)")
    print("=" * 70)
    print(f"CPU   : {tel['cpu_model']} ({tel['microarchitecture']})")
    print(f"Cores : {tel['physical_cores']} physical / {tel['logical_threads']} logical")
    print(f"Work  : {K:,} points per worker, {RUNS} runs per P, P = 1..{pmax}\n")

    integrate(0)                                                 # warm the parent
    rows = []
    for P in Ps:
        with mp.Pool(processes=P) as pool:
            pool.map(integrate, range(P))                        # spawn + warm
            for run in range(1, RUNS + 1):
                t0 = time.perf_counter()
                out = pool.map(integrate, range(P))
                wall = time.perf_counter() - t0
                rows.append({
                    "P": P, "run": run, "points_per_worker": K, "total_points": P * K,
                    "wall_s": wall,
                    "worker_compute_mean_s": sum(t for _, t in out) / P,
                    "worker_compute_max_s": max(t for _, t in out),
                    "integral": sum(v for v, _ in out),
                })
        ws = [r["wall_s"] for r in rows if r["P"] == P]
        print(f"  P = {P:>2}: mean wall {np.mean(ws):.4f} s  (std {np.std(ws, ddof=1):.4f})",
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "telemetry.json").write_text(json.dumps(tel, indent=2), encoding="utf-8")
    print(f"\nWritten {OUT / 'results.csv'} and {OUT / 'telemetry.json'}")


if __name__ == "__main__":
    main()
