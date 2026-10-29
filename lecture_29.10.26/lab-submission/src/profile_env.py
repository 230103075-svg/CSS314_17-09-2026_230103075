"""
Milestone 0, Task 0.1 — Host topology and cache telemetry.

Usage:
    py src/profile_env.py --student-id 230103075

Writes:
    results/host_profile.json    CPU model, microarchitecture, frequency,
                                 physical vs logical cores (SMT), L1d/L2/L3
                                 sizes, cache-line size, OS and Python.
    results/salted_params.json   N, buffer size and RNG seed derived from the
                                 Student ID by src/salt.py.

Cache sizes are read from the operating system itself, never hard-coded:
  Windows  GetLogicalProcessorInformation (kernel32, via ctypes)
  Linux    /sys/devices/system/cpu/cpu*/cache/index*
  macOS    sysctl hw.*cachesize
"""
import argparse
import ctypes
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from salt import get_student_params

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


# ----------------------------------------------------------------- helpers
def parse_size(text):
    """'48K' / '1280K' / '12M' / '32768' -> bytes."""
    text = str(text).strip().upper()
    m = re.match(r"^(\d+)\s*([KMG]?)", text)
    if not m:
        return None
    value = int(m.group(1))
    mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[m.group(2)]
    return value * mult


INTEL_FAMILY6 = {
    154: "Alder Lake (P/H mobile)", 151: "Alder Lake (S desktop)",
    183: "Raptor Lake", 186: "Raptor Lake", 191: "Raptor Lake",
    170: "Meteor Lake", 189: "Lunar Lake", 197: "Arrow Lake", 198: "Arrow Lake",
    140: "Tiger Lake", 141: "Tiger Lake", 125: "Ice Lake", 126: "Ice Lake",
    165: "Comet Lake", 166: "Comet Lake", 142: "Kaby/Coffee/Whiskey Lake",
    158: "Kaby/Coffee Lake", 94: "Skylake", 78: "Skylake",
    85: "Skylake-SP / Cascade Lake", 106: "Ice Lake-SP", 108: "Ice Lake-SP",
    143: "Sapphire Rapids", 207: "Emerald Rapids", 79: "Broadwell-E",
    61: "Broadwell", 60: "Haswell", 63: "Haswell-E", 69: "Haswell", 70: "Haswell",
}
AMD_FAMILY = {23: "Zen / Zen+ / Zen 2", 25: "Zen 3 / Zen 4", 26: "Zen 5"}


def microarchitecture(vendor, family, model):
    try:
        family, model = int(family), int(model)
    except (TypeError, ValueError):
        return "unknown"
    v = (vendor or "").lower()
    if "intel" in v and family == 6:
        return INTEL_FAMILY6.get(model, f"Intel family 6 model {model}")
    if "amd" in v:
        return AMD_FAMILY.get(family, f"AMD family {family}")
    return f"family {family} model {model}"


# ------------------------------------------------------------------ Windows
def windows_profile():
    info = {}
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        info["cpu_model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        info["base_frequency_mhz"] = int(winreg.QueryValueEx(key, "~MHz")[0])
        info["vendor"] = winreg.QueryValueEx(key, "VendorIdentifier")[0].strip()
        ident = winreg.QueryValueEx(key, "Identifier")[0]
        m = re.search(r"Family (\d+) Model (\d+) Stepping (\d+)", ident)
        if m:
            info["family"], info["model"], info["stepping"] = map(int, m.groups())
        info["processor_identifier"] = ident
    except Exception as exc:                                     # noqa: BLE001
        info["registry_error"] = str(exc)

    # ---- topology and caches straight from the kernel
    class CACHE_DESCRIPTOR(ctypes.Structure):
        _fields_ = [("Level", ctypes.c_ubyte), ("Associativity", ctypes.c_ubyte),
                    ("LineSize", ctypes.c_ushort), ("Size", ctypes.c_uint32),
                    ("Type", ctypes.c_int)]

    class _UNION(ctypes.Union):
        _fields_ = [("Flags", ctypes.c_ubyte), ("NodeNumber", ctypes.c_uint32),
                    ("Cache", CACHE_DESCRIPTOR),
                    ("Reserved", ctypes.c_ulonglong * 2)]

    class SLPI(ctypes.Structure):
        _fields_ = [("ProcessorMask", ctypes.c_size_t),
                    ("Relationship", ctypes.c_int), ("u", _UNION)]

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        length = ctypes.c_uint32(0)
        k32.GetLogicalProcessorInformation(None, ctypes.byref(length))
        count = length.value // ctypes.sizeof(SLPI)
        buf = (SLPI * count)()
        if not k32.GetLogicalProcessorInformation(buf, ctypes.byref(length)):
            raise OSError(ctypes.get_last_error())

        caches, cores, smt_cores = [], 0, 0
        type_name = {0: "Unified", 1: "Instruction", 2: "Data", 3: "Trace"}
        for e in buf[: length.value // ctypes.sizeof(SLPI)]:
            if e.Relationship == 0:                  # RelationProcessorCore
                cores += 1
                if e.u.Flags & 1:                    # LTP_PC_SMT
                    smt_cores += 1
            elif e.Relationship == 2:                # RelationCache
                c = e.u.Cache
                caches.append({
                    "level": int(c.Level),
                    "type": type_name.get(int(c.Type), str(c.Type)),
                    "size_bytes": int(c.Size),
                    "line_bytes": int(c.LineSize),
                    "associativity": int(c.Associativity),
                    "logical_cpus_sharing": bin(e.ProcessorMask).count("1"),
                })
        info["caches_raw"] = caches
        info["physical_cores_os"] = cores
        info["cores_with_smt"] = smt_cores
        info["cache_source"] = "Windows GetLogicalProcessorInformation"
    except Exception as exc:                                     # noqa: BLE001
        info["cache_error"] = str(exc)
    return info


# -------------------------------------------------------------------- Linux
def linux_profile():
    info = {}
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            text = fh.read()
        first = text.split("\n\n")[0]
        fields = {}
        for line in first.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fields[k.strip()] = v.strip()
        info["cpu_model"] = fields.get("model name", "unknown")
        info["vendor"] = fields.get("vendor_id", "")
        info["family"] = fields.get("cpu family")
        info["model"] = fields.get("model")
        info["stepping"] = fields.get("stepping")
        core_ids = {(b.split("physical id")[1].split("\n")[0] if "physical id" in b else "0",
                     re.search(r"core id\s*:\s*(\d+)", b).group(1))
                    for b in text.split("\n\n") if re.search(r"core id\s*:\s*(\d+)", b)}
        if core_ids:
            info["physical_cores_os"] = len(core_ids)
    except Exception as exc:                                     # noqa: BLE001
        info["cpuinfo_error"] = str(exc)

    for p in ("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency",
              "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"):
        try:
            info["base_frequency_mhz"] = int(open(p).read().strip()) // 1000
            info["frequency_source"] = p
            break
        except Exception:                                        # noqa: BLE001
            continue
    if "base_frequency_mhz" not in info:
        m = re.search(r"@\s*([\d.]+)\s*GHz", info.get("cpu_model", ""))
        if m:
            info["base_frequency_mhz"] = int(float(m.group(1)) * 1000)
            info["frequency_source"] = "brand string"

    caches, seen = [], set()
    base = Path("/sys/devices/system/cpu")
    for cpu in sorted(base.glob("cpu[0-9]*")):
        for idx in sorted((cpu / "cache").glob("index*")):
            try:
                rd = lambda n: (idx / n).read_text().strip()     # noqa: E731
                shared = rd("shared_cpu_list")
                key = (rd("level"), rd("type"), rd("size"), shared)
                if key in seen:
                    continue
                seen.add(key)
                n_share = 0
                for part in shared.split(","):
                    if "-" in part:
                        a, b = part.split("-")
                        n_share += int(b) - int(a) + 1
                    elif part:
                        n_share += 1
                caches.append({
                    "level": int(rd("level")),
                    "type": rd("type"),
                    "size_bytes": parse_size(rd("size")),
                    "line_bytes": int(rd("coherency_line_size")),
                    "associativity": int(rd("ways_of_associativity") or 0),
                    "logical_cpus_sharing": n_share,
                })
            except Exception:                                    # noqa: BLE001
                continue
    info["caches_raw"] = caches
    info["cache_source"] = "/sys/devices/system/cpu/*/cache"
    return info


# -------------------------------------------------------------------- macOS
def macos_profile():
    def sc(name):
        try:
            return subprocess.check_output(["sysctl", "-n", name]).decode().strip()
        except Exception:                                        # noqa: BLE001
            return None
    caches = []
    for lvl, key in ((1, "hw.l1dcachesize"), (2, "hw.l2cachesize"), (3, "hw.l3cachesize")):
        v = sc(key)
        if v and v.isdigit() and int(v) > 0:
            caches.append({"level": lvl, "type": "Data" if lvl == 1 else "Unified",
                           "size_bytes": int(v), "line_bytes": int(sc("hw.cachelinesize") or 64),
                           "associativity": 0, "logical_cpus_sharing": None})
    freq = sc("hw.cpufrequency")
    return {
        "cpu_model": sc("machdep.cpu.brand_string") or platform.processor(),
        "vendor": sc("machdep.cpu.vendor") or "",
        "family": sc("machdep.cpu.family"), "model": sc("machdep.cpu.model"),
        "physical_cores_os": int(sc("hw.physicalcpu") or 0) or None,
        "base_frequency_mhz": int(freq) // 1_000_000 if freq and freq.isdigit() else None,
        "caches_raw": caches, "cache_source": "sysctl",
    }


# --------------------------------------------------------------------- main
def summarise_caches(caches):
    """Pick headline L1d / L2 / L3 values; keep every distinct cache too."""
    def biggest(level, kinds):
        sizes = [c["size_bytes"] for c in caches
                 if c["level"] == level and c["type"] in kinds and c["size_bytes"]]
        return max(sizes) if sizes else None

    distinct = {}
    for c in caches:
        k = (c["level"], c["type"], c["size_bytes"], c["logical_cpus_sharing"])
        distinct.setdefault(k, {**c, "instances": 0})
        distinct[k]["instances"] += 1
    lines = {c["line_bytes"] for c in caches if c.get("line_bytes")}
    return {
        "l1d_cache_bytes": biggest(1, ("Data",)),
        "l1i_cache_bytes": biggest(1, ("Instruction",)),
        "l2_cache_bytes": biggest(2, ("Unified", "Data")),
        "l3_cache_bytes": biggest(3, ("Unified", "Data")),
        "cache_line_bytes": min(lines) if lines else 64,
        "cache_hierarchy_detail": sorted(distinct.values(),
                                         key=lambda d: (d["level"], d["type"], -d["size_bytes"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", default="230103075")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    raw = (windows_profile() if system == "Windows"
           else macos_profile() if system == "Darwin" else linux_profile())

    logical = os.cpu_count()
    physical = raw.get("physical_cores_os")
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or physical
        freq = psutil.cpu_freq()
        psutil_freq = {"current_mhz": round(freq.current, 1), "max_mhz": round(freq.max, 1)} if freq else None
    except Exception:                                            # noqa: BLE001
        psutil_freq = None

    caches = summarise_caches(raw.get("caches_raw", []))
    profile = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "student_id": args.student_id,
        "cpu": {
            "model_name": raw.get("cpu_model", platform.processor()),
            "vendor": raw.get("vendor"),
            "family": raw.get("family"), "model": raw.get("model"),
            "stepping": raw.get("stepping"),
            "microarchitecture": microarchitecture(raw.get("vendor"), raw.get("family"), raw.get("model")),
            "base_frequency_mhz": raw.get("base_frequency_mhz"),
            "psutil_frequency": psutil_freq,
        },
        "topology": {
            "physical_cores": physical,
            "logical_threads": logical,
            "smt_enabled": bool(physical and logical and logical > physical),
            "threads_per_core_avg": round(logical / physical, 3) if physical else None,
            "cores_with_smt": raw.get("cores_with_smt"),
            "hybrid_note": ("Some cores run 2 threads and some 1: hybrid P-core/E-core design"
                            if raw.get("cores_with_smt") and physical
                            and 0 < raw["cores_with_smt"] < physical else None),
        },
        "caches": {k: v for k, v in caches.items()},
        "cache_source": raw.get("cache_source"),
        "os": {
            "system": system, "release": platform.release(),
            "version": platform.version(), "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "executable_bits": ctypes.sizeof(ctypes.c_void_p) * 8,
        },
        "errors": {k: v for k, v in raw.items() if k.endswith("_error")},
    }
    try:
        import numpy
        profile["python"]["numpy"] = numpy.__version__
    except ImportError:
        pass

    params = get_student_params(args.student_id)

    (RESULTS / "host_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    (RESULTS / "salted_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    kb = lambda b: f"{b // 1024} KB" if b else "unknown"                # noqa: E731
    print("=" * 70)
    print("MILESTONE 0 — HOST TOPOLOGY")
    print("=" * 70)
    print(f"CPU            : {profile['cpu']['model_name']}")
    print(f"Microarch      : {profile['cpu']['microarchitecture']}")
    print(f"Base frequency : {profile['cpu']['base_frequency_mhz']} MHz")
    print(f"Cores          : {physical} physical / {logical} logical "
          f"(SMT {'ON' if profile['topology']['smt_enabled'] else 'OFF'})")
    if profile["topology"]["hybrid_note"]:
        print(f"                 {profile['topology']['hybrid_note']}")
    print(f"L1d / L2 / L3  : {kb(caches['l1d_cache_bytes'])} / "
          f"{kb(caches['l2_cache_bytes'])} / {kb(caches['l3_cache_bytes'])}")
    print(f"Cache line     : {caches['cache_line_bytes']} bytes")
    print(f"OS / Python    : {platform.platform()} / {platform.python_version()}")
    print("-" * 70)
    print(f"Salted params  : {params}")
    print(f"Written        : {RESULTS / 'host_profile.json'}")
    print(f"                 {RESULTS / 'salted_params.json'}")
    if profile["errors"]:
        print(f"WARNINGS       : {profile['errors']}", file=sys.stderr)


if __name__ == "__main__":
    main()
