import math
import multiprocessing as mp
import time
from harness import run_benchmark, emit_verify_tag
import labutil as L

N = 800 if L.FAST else 8000
W = 1 if L.FAST else int(__import__("os").environ.get("LAB_P4_WARMUP", 1))
R = 3 if L.FAST else int(__import__("os").environ.get("LAB_P4_RUNS", 3))   # heavy workload: fewer runs


def work(i):
    k = 1 + (i * i) // 1600                       # == 1 + floor((i/40)^2)
    x = 0.001 * i + 0.1
    for _ in range(k):
        x = math.sqrt(abs(math.sin(x) * math.cos(x))) + 0.5
    return x


def static_worker(mode, t, p, tarr):
    t0 = time.perf_counter()
    it = range(t * N // p, (t + 1) * N // p) if mode == "block" else range(t, N, p)
    for i in it:
        work(i)
    tarr[t] = time.perf_counter() - t0


def dynamic_worker(counter, C, t, tarr):
    t0 = time.perf_counter()
    while True:
        with counter.get_lock():
            s = counter.value
            counter.value = s + C
        if s >= N:
            break
        for i in range(s, min(s + C, N)):
            work(i)
    tarr[t] = time.perf_counter() - t0


def lam_of(tarr):
    v = list(tarr)
    return max(v) / (sum(v) / len(v))


def run_strategy(kind, p, C=None):
    """Returns (median Twall ms, median lambda)."""
    lams = []

    def trial():
        tarr = mp.Array("d", p)
        if kind == "dynamic":
            counter = mp.Value("i", 0)
            procs = [mp.Process(target=dynamic_worker, args=(counter, C, t, tarr)) for t in range(p)]
        else:
            procs = [mp.Process(target=static_worker, args=(kind, t, p, tarr)) for t in range(p)]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        lams.append(lam_of(tarr))

    st = run_benchmark(trial, W, R)
    meas = sorted(lams[W:])
    return st.median_ms, meas[len(meas) // 2]


if __name__ == "__main__":
    seed = L.seed_of(L.get_id())
    pphys = L.detect_pphys()
    res = {}
    res["Block static"] = run_strategy("block", pphys)
    res["Cyclic static"] = run_strategy("cyclic", pphys)
    for C in (1, 8, 64, 512):
        res[f"Dynamic C={C}"] = run_strategy("dynamic", pphys, C)
    dyn = {k: v for k, v in res.items() if k.startswith("Dynamic")}
    best_key = min(dyn, key=lambda k: dyn[k][0])
    Cstar = int(best_key.split("=")[1])
    tag = emit_verify_tag("PHASE4", seed, res["Block static"][0], dyn[best_key][0])
    t = L.md_table(["Strategy", "T_wall ms", "lambda"], [[k, f"{v[0]:.1f}", f"{v[1]:.3f}"] for k, v in res.items()])
    md = (f"## Phase 4: Skewed Workload Scheduling (N={N}, p=Pphys={pphys})\n\n{t}\n\n### Analysis\n\n"
          f"- Block static gives the last worker the heaviest items (K grows ~ i^2), so lambda = {res['Block static'][1]:.2f} "
          f"and T_wall is set by the slowest worker. Cyclic static interleaves cheap and heavy items: lambda = {res['Cyclic static'][1]:.2f}.\n"
          f"- C=1 maximizes balance but every item takes the lock on the shared counter (synchronization + cache-line bouncing of the counter); "
          f"C=512 amortizes the lock but the final chunk can be a large block of heavy items (tail imbalance, lambda = {res['Dynamic C=512'][1]:.2f}).\n"
          f"- Optimal C* = {Cstar} (T_wall = {dyn[best_key][0]:.1f} ms, lambda = {dyn[best_key][1]:.2f}): "
          f"small enough to keep the tail short, large enough that lock cost per item is negligible next to the work per chunk.\n\n```\n{tag}\n```\n")
    L.save_phase(4, md, {"hash": tag, "Cstar": Cstar})
