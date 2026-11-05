"""
Generates every worked numeric solution in calculations/ from first
principles, so each number in those files is reproducible.

Usage (from the repository root or this folder):
    py calculations/compute_calculations.py

Writes:
    task06_parametric_table.md   task07_tolerable_serial.md
    task08_cross_divergence.md   task09_multicomponent.md
    task10_memory_growth.md      task16_gemm_scaling.md
    task17_stencil_scaling.md
Numbers are printed at full working precision; only the final display is
formatted.
"""
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent


def gustafson(P, a):
    return P - a * (P - 1)


def amdahl(P, s):
    return 1 / (s + (1 - s) / P)


def write(name, text):
    (HERE / name).write_text(text.strip() + "\n", encoding="utf-8")
    print(f"written {name}")


# ---------------------------------------------------------------- Task 06
a = 0.04
rows = []
for P in (16, 64, 256, 1024, 4096, 16384):
    S = gustafson(P, a)
    E = S / P
    rows.append(f"| {P:,} | {P} − 0.04·{P - 1} = {S:,.2f} | {E:.6f} | {(1 - E) * 100:.4f} % | "
                f"{(E - (1 - a)) * 100:.6f} pp |")
write("task06_parametric_table.md", f"""
# Task 06 — Parametric Speedup Across Processor Densities

Serial fraction on the parallel machine: **α = 0.04**.

```
S(P) = P − α(P − 1)
E(P) = S(P)/P = 1 − α(P − 1)/P
```

| P | S(P) | E(P) | efficiency lost | E(P) − (1 − α) |
|---|---|---|---|---|
{chr(10).join(rows)}

## Evaluation

* S(P) grows **linearly** with slope 1 − α = 0.96: each 4× increase in P gives almost exactly 4× the scaled speedup.
* E(P) falls only from 0.9625 (P = 16) to 0.960002 (P = 16,384). Efficiency **does not degrade** as P grows exponentially: it converges to the limit 1 − α = 0.96 (Task 04), and the remaining gap α/P halves each time P doubles.
* The efficiency lost is almost entirely the constant α = 4 %: the serial phase idles every processor but one for 4 % of the run, whatever P is.
* Contrast with Amdahl at s = 0.04: S could never exceed 25, so E at P = 16,384 would be below 0.0016.
""")

# ---------------------------------------------------------------- Task 07
P, S_req = 8192, 7500
amax = Fraction(P - S_req, P - 1)
write("task07_tolerable_serial.md", f"""
# Task 07 — Maximum Tolerable Serial Overhead

**Given:** P = 8,192 cores, required S(P) ≥ 7,500.

## Derivation

```
S(P) = P − α(P − 1) ≥ S_req
        α(P − 1) ≤ P − S_req
               α ≤ (P − S_req) / (P − 1)
```

S(P) decreases as α increases (dS/dα = −(P − 1) < 0), so the largest α that meets the requirement is the boundary value:

```
α_max = (8192 − 7500) / (8192 − 1) = 692 / 8191
```

## Exact numerical tolerance

* exact fraction: **α_max = 692/8191** (already in lowest terms)
* decimal: **α_max = {float(amax):.10f}** ≈ **{float(amax) * 100:.6f} %**

## Check

```
S = 8192 − (692/8191)·8191 = 8192 − 692 = 7500   ✓
```

Any α ≤ 0.0844829691 (8.448 % of the parallel run spent serially) meets the target; at α_max the efficiency is exactly 7500/8192 = {7500 / 8192:.6f}.
""")

# ---------------------------------------------------------------- Task 08
s, P = 0.08, 128
SA = amdahl(P, s)
SG = gustafson(P, s)
write("task08_cross_divergence.md", f"""
# Task 08 — Cross-Over Point: Amdahl vs. Gustafson (s = α = 0.08, P = 128)

## Amdahl (fixed size)

```
S_A = 1 / (s + (1 − s)/P)
    = 1 / (0.08 + 0.92/128)
    = 1 / (0.08 + {0.92 / 128})
    = 1 / {0.08 + 0.92 / 128}
    = {SA:.6f}
```

## Gustafson (fixed time)

```
S_G = P − α(P − 1)
    = 128 − 0.08 × 127
    = 128 − 10.16
    = {SG:.2f}
```

## Discrepancy

| Measure | Value |
|---|---|
| Ratio S_G / S_A | {SG / SA:.6f} |
| Absolute difference S_G − S_A | {SG - SA:.6f} |
| Percentage discrepancy relative to Amdahl, (S_G − S_A)/S_A | {(SG - SA) / SA * 100:.4f} % |
| Percentage discrepancy relative to Gustafson, (S_G − S_A)/S_G | {(SG - SA) / SG * 100:.4f} % |
| Efficiency, Amdahl | {SA / P:.6f} |
| Efficiency, Gustafson | {SG / P:.6f} |

## Why they diverge as P grows

* Amdahl's speedup approaches the ceiling 1/s = 12.5; at P = 128 it is already at {SA / 12.5 * 100:.2f} % of that ceiling.
* Gustafson's speedup keeps growing with slope 1 − α = 0.92.
* Their ratio grows roughly in proportion to P: for large P, S_G/S_A ≈ s(1 − s)P.

| P | S_A | S_G | S_G / S_A |
|---|---|---|---|
""" + "\n".join(f"| {p:,} | {amdahl(p, s):.4f} | {gustafson(p, s):.2f} | {gustafson(p, s) / amdahl(p, s):.4f} |"
                for p in (2, 8, 32, 128, 512, 2048)) + """

The divergence is not a contradiction. The same 8 % is measured in different frames (Task 03): Amdahl's s is the serial share of a fixed problem on one processor; Gustafson's α is the serial share of a problem scaled to fill 128 processors. For the scaled problem the uniprocessor serial share is only α/S_G = """ + f"{s / SG:.6f}" + """, so Amdahl's ceiling for that problem is far above 117.84.
""")

# ---------------------------------------------------------------- Task 09
phases = {"(A) disk I/O and parameter broadcast": 0.02,
          "(B) boundary message-passing synchronisation": 0.06,
          "(C) core domain flux computation": 0.92}
a9 = 0.02 + 0.06
r9 = "\n".join(f"| {P:,} | {P} − 0.08·{P - 1} = {gustafson(P, a9):,.2f} | {gustafson(P, a9) / P:.6f} |"
               for P in (512, 2048))
write("task09_multicomponent.md", f"""
# Task 09 — Multi-Component Workload Partitioning

## Phase breakdown (shares of total parallel execution time)

| Phase | Share | Classification |
|---|---|---|
| (A) Initial disk I/O and parameter broadcast | 2 % | serial: one reader, then a broadcast; does not shrink with P |
| (B) Boundary message-passing synchronisation | 6 % | non-scaling: time spent waiting/exchanging, not doing domain work |
| (C) Core domain flux computation | 92 % | parallel: work divided across all processors |

Only (C) is useful parallel work that scales with the problem. Both (A) and (B) are time during which the machine is not doing scalable computation, so in Gustafson's model both count towards the serial fraction:

```
α = α_A + α_B = 0.02 + 0.06 = {a9:.2f}
p = 1 − α     = {1 - a9:.2f}
```

## Composite scaled speedup

| P | S(P) = P − α(P − 1) | E(P) |
|---|---|---|
{r9}

## Analysis

* At P = 2,048 the simulation delivers 1,884.24× the work of one processor in the same wall-clock time; 92 % efficiency is retained.
* The synchronisation phase is **three times** as costly as the I/O phase: reducing (B) from 6 % to 3 % would raise S(2048) to {gustafson(2048, 0.05):,.2f}, while removing (A) entirely would only reach {gustafson(2048, 0.06):,.2f}.
* Caveat: (B) is unlikely to stay at exactly 6 % as P grows; halo exchanges stay roughly constant per processor, but any global synchronisation grows as O(log P) (Task 12). The figures above are the classical model with α held constant.
""")

# ---------------------------------------------------------------- Task 10
GiB = 2 ** 30
cap = 32 * GiB
lines = []
for b, label in ((8, "1 field, float64 (8 B/point)"), (40, "5 fields, float64 (40 B/point)")):
    for nodes in (64, 512):
        total = nodes * cap
        pts = total // b
        n = round(pts ** (1 / 3))
        while n ** 3 > pts:
            n -= 1
        while (n + 1) ** 3 <= pts:
            n += 1
        lines.append(f"| {label} | {nodes} | {total:,} B | {pts:,} | **{n:,}** | {n ** 3 * b / total * 100:.4f} % |")
write("task10_memory_growth.md", f"""
# Task 10 — Memory Growth Rate vs. Scaled Speedup

## Formulation

Under weak scaling W(P) = P·W(1), and the aggregate memory footprint is M(P) = c·P: each node holds a constant amount of data. For a 3-D grid of N × N × N points with b bytes per point:

```
M(N) = b·N³
```

With per-node memory capped at m = 32 GB, P nodes provide P·m bytes, so

```
b·N³ ≤ P·m    ⇒    N_max(P) = ⌊ (P·m / b)^(1/3) ⌋
```

N_max grows only as **P^(1/3)**: 8× more nodes doubles the grid edge.

**Assumptions.** 32 GB is taken as 32 GiB = 32 × 2³⁰ = {cap:,} bytes, and the whole node memory is available to the grid (an upper bound; a real code leaves room for the OS, buffers and halos).

## Capacity limits

| Data per grid point | Nodes | Aggregate memory | Max grid points | N_max | memory used |
|---|---|---|---|---|---|
{chr(10).join(lines)}

## Observations

* For one float64 field: 64 nodes hold a grid of edge 6,501 and 512 nodes an edge of 13,003, i.e. 2× the edge for 8× the nodes (the exact cube roots are 2^(38/3) ≈ 6,501.99 and 2^(41/3) ≈ 13,003.99; flooring to whole grid points gives the table values).
* Per node the share is always 32 GiB / b points: ≈ {cap // 8:,} points (an edge of {round((cap // 8) ** (1 / 3)):,}) for 8 B/point.
* Consequence for scaled speedup: if the computation per grid point is constant, work ∝ N³ ∝ P, so Gustafson's linear W(P) = P·W(1) is achievable exactly up to the memory ceiling. If the algorithm's work grows faster than N³ (e.g. more time steps on a finer mesh, which CFL stability typically requires, ∝ N), the work becomes ∝ N⁴ ∝ P^(4/3) and the runtime can no longer stay constant: memory, not processor count, then limits the scaled problem (Sun–Ni, Task 11).
""")

# ---------------------------------------------------------------- Task 16
N1 = 4096
r16 = []
for P in (1, 8, 64, 512, 4096, 32768):
    N = N1 * round(P ** (1 / 3))
    flops = 2 * N ** 3
    mem = 3 * N * N * 8
    r16.append(f"| {P:,} | {N:,} | {flops / 1e12:,.3f} | {flops / P / 1e9:,.3f} | "
               f"{mem / GiB:,.3f} | {mem / P / 2 ** 20:,.3f} |")
write("task16_gemm_scaling.md", f"""
# Task 16 — Dense Matrix Multiplication (GEMM) Scaling

## Derivation

For C = A × B with N × N matrices:

```
computation  W(N) = 2N³ flops        (N³ multiply-adds)
memory       M(N) = 3N² words        (A, B, C)
```

Gustafson requires the parallel runtime to stay constant on P processors. With perfectly divided work,

```
T(P) = W(N_P) / (P·r) = 2N_P³ / (P·r) = const = 2N_1³ / r
⇒ N_P³ = P·N_1³
⇒ N_P = N_1 · P^(1/3)
```

where r is the per-processor flop rate. **Matrix dimension grows as the cube root of P.**

Memory then grows as

```
M(N_P) = 3N_P² = 3N_1²·P^(2/3)        (total)
M(N_P)/P = 3N_1²·P^(−1/3)             (per processor)
```

so per-processor memory **shrinks** as P grows. GEMM is compute-bound under fixed-time scaling: memory is never the binding constraint, and the problem could grow further if one were willing to let time grow (Sun–Ni's G(P) = P^(3/2) memory-bounded scaling, Task 11).

## Scaling targets (N_1 = 4,096, float64)

| P | N_P | total TFLOP | GFLOP per processor | total memory (GiB) | memory per processor (MiB) |
|---|---|---|---|---|---|
{chr(10).join(r16)}

Every row does the same 137.439 GFLOP per processor, so runtime stays constant, while memory per processor halves for every 8× increase in P.
""")

# ---------------------------------------------------------------- Task 17
n = 128
r17 = []
for P in (1, 8, 64, 512, 4096, 32768):
    k = round(P ** (1 / 3))
    Ng = n * k
    vol = n ** 3
    surf = 6 * n * n
    r17.append(f"| {P:,} | {Ng:,} | {vol:,} | {surf:,} | {surf / vol:.6f} | {6 / Ng:.6f} |")
r17s = []
Nfix = 1024
for P in (1, 8, 64, 512, 4096, 32768):
    sub = Nfix / round(P ** (1 / 3))
    r17s.append(f"| {P:,} | {sub:g} | {6 / sub:.6f} |")
write("task17_stencil_scaling.md", f"""
# Task 17 — 3-D Stencil Hydrodynamics: Surface-to-Volume Scaling

## Setup

A 7-point stencil on a cubic domain, decomposed into P cubic subdomains of edge n:

```
computation per subdomain  ∝ 7n³   (7 points read per interior update)
communication per subdomain ∝ 6n²  (one ghost layer on each of 6 faces)
```

## Proof under Gustafson (weak) scaling

Gustafson fixes the work per processor, so n is constant and the global domain grows as

```
N³ = P·n³    ⇒    N = n·P^(1/3)
```

Per-processor communication-to-computation ratio:

```
R = (6n²)/(7n³) ∝ 6/n        — independent of P
```

**The ratio stays constant.** Each subdomain exchanges the same 6n² ghost cells per step however many neighbours there are, so communication time per step is constant and S(P) stays close to Gustafson's linear law. The whole domain's surface-to-volume ratio 6/N even **improves** as P^(−1/3): the external boundary becomes a vanishing fraction of a growing interior.

Table (n = 128 points per subdomain edge):

| P | global edge N | cells per subdomain | ghost cells per subdomain | per-processor ratio 6n²/n³ | global ratio 6/N |
|---|---|---|---|---|---|
{chr(10).join(r17)}

## Contrast: strong scaling (fixed N = 1,024)

With N fixed, the subdomain edge shrinks as n = N/P^(1/3) and the ratio **degrades** as P^(1/3):

| P | subdomain edge n | ratio 6/n |
|---|---|---|
{chr(10).join(r17s)}

## Caveats

1. Constant halo volume is not constant halo **time**: message latency is per message, and at very small n latency dominates bandwidth.
2. Global operations (a time-step reduction, a CFL check) add an O(log P) term each step; that is the term that eventually bends weak scaling away from linear (Task 12).
3. Network contention can raise exchange time on very large machines if neighbouring subdomains are not mapped to neighbouring nodes.

**Conclusion:** under Gustafson scaling the stencil's communication-to-computation ratio is constant per processor and improves globally, which is why explicit stencil codes are the textbook example of near-ideal weak scaling.
""")
