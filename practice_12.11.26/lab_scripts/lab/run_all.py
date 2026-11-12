"""Usage: python run_all.py <StudentID>   -> runs all phases, writes lab_report_<ID>.md"""
import glob, re, subprocess, sys

sid = sys.argv[1] if len(sys.argv) > 1 else "0"
TASKS = ["task0_cal.py", "task1_falsesharing.py", "task2_scaling.py", "task3_roofline_gemm.py", "task4_scheduling.py"]
for t in TASKS:
    print(f"\n===== {t} =====", flush=True)
    subprocess.run([sys.executable, t, sid], check=True)

parts = [open(f"out/phase{i}.md", encoding="utf-8").read() for i in range(5)]
hashes = [m for p in parts for m in re.findall(r"VERIFY_HASH\[PHASE\d\]:[^\n]*", p)]
rep = [f"# Lab Report - {sid}\n", "## VERIFY_HASH lines\n", "```", *hashes, "```\n", *parts,
       "\n## Complete scripts\n"]
for f in ["harness.py", "labutil.py"] + TASKS + ["run_all.py"]:
    rep.append(f"### {f}\n\n```python\n{open(f, encoding='utf-8').read()}```\n")
out = f"lab_report_{sid}.md"
open(out, "w", encoding="utf-8").write("\n".join(rep))
print(f"\nReport written: {out}")
