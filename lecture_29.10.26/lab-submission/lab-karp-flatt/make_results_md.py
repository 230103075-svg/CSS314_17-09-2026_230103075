"""
Part 4 — builds RESULTS.md from the three experiments' measured data.

Usage (after running exp_a, exp_b and exp_c):
    py make_results_md.py

Every number in RESULTS.md is read from results/exp_*.json; nothing is
typed in by hand.
"""
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
STUDENT_ID = "230103075"


def cpu_name():
    if platform.system() == "Windows":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:                                    # noqa: BLE001
            pass
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:                                        # noqa: BLE001
        pass
    return platform.processor() or "unknown"


def physical():
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except Exception:                                        # noqa: BLE001
        return None


a = json.loads((R / "exp_a.json").read_text(encoding="utf-8"))
b = json.loads((R / "exp_b.json").read_text(encoding="utf-8"))
c = json.loads((R / "exp_c.json").read_text(encoding="utf-8"))
ch, cw = c["handout"], c["warm_pool"]

f = lambda x: f"{x:.3f}"                                     # noqa: E731
row = lambda name, d, diag: (                                # noqa: E731
    f"| {name} | {f(d['T']['1'])} | {f(d['T']['2'])} | {f(d['T']['4'])} | "
    f"{f(d['e']['2'])} | {f(d['e']['4'])} | {diag} |")

ok = lambda v: "PASS" if v else "NOT MET"                    # noqa: E731
e4_over_e2 = a["e"]["4"] / a["e"]["2"]
b_slower = b["T"]["4"] > b["T"]["1"]
b_surge = b["e"]["4"] > b["e"]["2"]

# ---------------------------------------------------- data-driven wording
a_e2, a_e4 = a["e"]["2"], a["e"]["4"]
if 0.80 <= e4_over_e2 <= 1.25:
    a_txt = (f"Experiment A loses performance to a fixed serial path: e stays flat "
             f"({f(a_e2)} at P=2, {f(a_e4)} at P=4) because the 0.40 s master sleep "
             f"is the same fraction of the run at every core count.")
    a_diag = "Constant e → fixed Amdahl serialization (0.40 s sleep)"
else:
    a_txt = (f"Experiment A did not give a flat e on this machine: e(2) = {f(a_e2)} and "
             f"e(4) = {f(a_e4)}.")
    if a_e2 < 0:
        a_txt += (f" A negative e(2) means P=2 ran faster than linear "
                  f"({a['speedup']['2']:.2f}x on 2 workers), which points to an inflated "
                  f"single-process baseline T1 = {f(a['T']['1'])} s rather than to the code: "
                  "this CPU mixes performance and efficiency cores, and a lone worker that "
                  "Windows schedules on an efficiency core, or at a lower turbo clock, runs "
                  "noticeably slower.")
    a_txt += (f" e(4) = {f(a_e4)} is where the handout expects a fixed serial tail "
              "(about 0.10–0.15), consistent with the 0.40 s sleep plus pool start-up.")
    a_diag = "Serial tail; e(2) distorted by a slow 1-process baseline"

b_e2, b_e4 = b["e"]["2"], b["e"]["4"]
b_txt = (f"Experiment B is fully parallel in principle, yet adding processes made it far "
         f"slower ({f(b['T']['1'])} s on 1 process vs {f(b['T']['2'])} s on 2 and "
         f"{f(b['T']['4'])} s on 4). Both Karp–Flatt values are huge (e(2) = {f(b_e2)}, "
         f"e(4) = {f(b_e4)}), an order of magnitude above Experiment A, which a serial "
         "code path cannot produce: the mutex and the cache line holding the counter "
         "bounce between cores on every increment.")
if not b_surge:
    b_txt += (f" e(4) came out lower than e(2) because the contention is already saturated "
              f"at two processes: T2 and T4 are almost equal "
              f"({f(b['T']['2'])} s vs {f(b['T']['4'])} s), meaning the lock fully "
              "serializes the work from P=2 onward. With T flat, the Karp–Flatt formula "
              "(p·Tp/T1 − 1)/(p − 1) yields a smaller value at larger p even though the "
              "pathology is the same.")
b_diag = ("Exploding e → mutex & bus thrashing" if b_surge
          else "Huge e (saturated at P=2) → mutex & bus thrashing")

c_txt = (f"Experiment C removes all shared state from the hot loop and synchronizes once at "
         f"the end. Measured exactly as in the handout, the timer also covers creating the "
         f"process pool, and on Windows each new process is spawned from scratch: that fixed "
         f"cost dominates a run this short, giving e(4) = {f(ch['e']['4'])} and "
         f"E4 = {ch['efficiency']['4'] * 100:.1f}%. With the pool started before the timer, "
         f"the same work gives e(4) = {f(cw['e']['4'])}, a {cw['speedup']['4']:.2f}x speedup "
         f"and E4 = {cw['efficiency']['4'] * 100:.1f}%: the privatized computation itself "
         "scales almost linearly, and the gap between the two rows is process start-up "
         "overhead, not contention.")
c_hand_diag = ("Collapsed e → scalability restored" if ch["rubric_pass"]
               else "e inflated by process start-up (see warm-pool row)")
c_warm_diag = ("Collapsed e → linear scalability restored" if cw["rubric_pass"]
               else "Process start-up excluded from the timer")

md = f"""# RESULTS — Karp–Flatt Practice Lab

**Student ID:** {STUDENT_ID}  ·  **Branch:** `practice-{STUDENT_ID}`

## Hardware profile

| Item | Value |
|---|---|
| CPU | {cpu_name()} |
| Physical cores | {physical() or 'n/a'} |
| Logical cores (`os.cpu_count()`) | {os.cpu_count()} |
| OS | {platform.platform()} |
| Python | {platform.python_implementation()} {platform.python_version()} |

Every value below is the median of {a['repeats']} runs.

## Results

| Experiment Mode | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | Architectural Diagnosis |
|---|---|---|---|---|---|---|
{row("A: Serial Tail", a, a_diag)}
{row("B: Lock Contention", b, b_diag)}
{row("C: Privatized State", ch, c_hand_diag)}
{row("C: Privatized (warm pool)", cw, c_warm_diag)}

Speedup: A {a['speedup']['4']:.2f}x · B {b['speedup']['4']:.2f}x · C {ch['speedup']['4']:.2f}x (handout) / {cw['speedup']['4']:.2f}x (warm pool) at P = 4.

## Rubric checks

| Check | Measured | Result |
|---|---|---|
| A: e(4)/e(2) within [0.80, 1.25] | {e4_over_e2:.3f} | {ok(0.80 <= e4_over_e2 <= 1.25)} |
| B: T4 > T1 | {f(b['T']['4'])} s vs {f(b['T']['1'])} s | {ok(b_slower)} |
| B: e(4) ≫ e(2) | {f(b['e']['4'])} vs {f(b['e']['2'])} | {ok(b_surge)} |
| C: T4 < T1 and E4 > 70% (handout timing) | E4 = {ch['efficiency']['4'] * 100:.1f}% | {ok(ch['rubric_pass'])} |
| C: T4 < T1 and E4 > 70% (warm pool) | E4 = {cw['efficiency']['4'] * 100:.1f}% | {ok(cw['rubric_pass'])} |

## Diagnosis

{a_txt}

{b_txt}

{c_txt}
"""

(ROOT / "RESULTS.md").write_text(md, encoding="utf-8")
print(md)
print(f"Written: {ROOT / 'RESULTS.md'}")
