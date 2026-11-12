"""Shared helpers (stdlib only): hardware detection, tables, result saving."""
import glob, json, os, platform, re, subprocess, sys
import multiprocessing as mp
from harness import fnv1a_hash

FAST = bool(os.environ.get("LAB_FAST"))          # smoke-test mode (tiny sizes)
WARMUP = 1 if FAST else int(os.environ.get("LAB_WARMUP", 2))
RUNS = 3 if FAST else int(os.environ.get("LAB_RUNS", 7))
OUT = "out"


def get_id():
    return sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LAB_ID", "0")


def seed_of(sid):
    d = re.sub(r"\D", "", sid)
    return int(d) if d else fnv1a_hash(sid) % 10**9


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return ""


def _sc(name):
    try:
        return int(_run(["sysctl", "-n", name]).strip())
    except Exception:
        return None


def _size(s):
    m = re.match(r"(\d+)\s*([KMG]?)", s.strip().upper())
    if not m:
        return None
    return int(m.group(1)) * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[m.group(2)]


def _ps(expr):
    return _run(["powershell", "-NoProfile", "-Command", expr]).strip()


def detect_pphys():
    """Physical cores. Override with env PPHYS if detection is wrong."""
    if os.environ.get("PPHYS"):
        return int(os.environ["PPHYS"])
    n = 0
    try:
        sysname = platform.system()
        if sysname == "Linux":
            out = _run(["lscpu", "-p=CORE,SOCKET"])
            n = len({l for l in out.splitlines() if l and not l.startswith("#")})
        elif sysname == "Darwin":
            n = _sc("hw.physicalcpu") or 0
        elif sysname == "Windows":
            n = int(_ps("(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"))
    except Exception:
        n = 0
    return n if n > 0 else max(1, mp.cpu_count() // 2)


def cpu_model():
    s = platform.system()
    try:
        if s == "Linux":
            for l in open("/proc/cpuinfo"):
                if l.lower().startswith("model name"):
                    return l.split(":", 1)[1].strip()
        elif s == "Darwin":
            return _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        elif s == "Windows":
            return _ps("(Get-CimInstance Win32_Processor | Select -First 1).Name")
    except Exception:
        pass
    return platform.processor() or "unknown"


def cache_info():
    """Best-effort bytes for L1d/L2/L3 and line size (None if unknown; env L1D_KB, L2_KB, L3_KB, LINE override)."""
    info = {"l1d": None, "l2": None, "l3": None, "line": 64}
    s = platform.system()
    try:
        if s == "Linux":
            for d in sorted(glob.glob("/sys/devices/system/cpu/cpu0/cache/index*")):
                rd = lambda f: open(os.path.join(d, f)).read().strip()
                lvl, typ, size = rd("level"), rd("type"), _size(rd("size"))
                if lvl == "1" and typ == "Data":
                    info["l1d"] = size
                elif lvl == "2":
                    info["l2"] = size
                elif lvl == "3":
                    info["l3"] = size
                try:
                    info["line"] = int(rd("coherency_line_size"))
                except Exception:
                    pass
        elif s == "Darwin":
            info["l1d"], info["l2"], info["l3"] = _sc("hw.l1dcachesize"), _sc("hw.l2cachesize"), _sc("hw.l3cachesize")
            info["line"] = _sc("hw.cachelinesize") or 64
        elif s == "Windows":
            info["l2"] = int(_ps("(Get-CimInstance Win32_Processor | Select -First 1).L2CacheSize")) * 1024
            info["l3"] = int(_ps("(Get-CimInstance Win32_Processor | Select -First 1).L3CacheSize")) * 1024
    except Exception:
        pass
    for k, e in (("l1d", "L1D_KB"), ("l2", "L2_KB"), ("l3", "L3_KB")):
        if os.environ.get(e):
            info[k] = int(os.environ[e]) * 1024
    if os.environ.get("LINE"):
        info["line"] = int(os.environ["LINE"])
    return info


def fmt_bytes(b):
    if b is None:
        return "n/a (fill manually)"
    return f"{b // 1024} KiB" if b < 1024**2 else f"{b / 1024**2:.1f} MiB"


def p_list(plog):
    ps, p = [], 1
    while p <= plog:
        ps.append(p)
        p *= 2
    if plog not in ps:
        ps.append(plog)
    return ps


def karp_flatt(S, p):
    return (1 / S - 1 / p) / (1 - 1 / p)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def save_phase(n, md, data):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/phase{n}.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(f"{OUT}/phase{n}.json", "w") as f:
        json.dump(data, f, indent=1)
    print(md)
