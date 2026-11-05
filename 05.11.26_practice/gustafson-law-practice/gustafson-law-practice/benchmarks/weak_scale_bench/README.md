# Weak-scaling benchmark (Task 22)

Run from the repository root:

```
py benchmarks/weak_scale_bench/bench.py
```

Close other programs and plug the laptop in (battery mode lowers the CPU frequency).
Outputs: `benchmarks/results.csv` (raw timings per P and run) and `benchmarks/telemetry.json` (CPU, cores, caches, memory modules, OS).
Then run `py benchmarks/fit_gustafson.py`, `py benchmarks/plot_scaling.py`, `py benchmarks/make_audit.py` (or just `py run_all.py`).
