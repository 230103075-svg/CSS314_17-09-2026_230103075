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
| 1 | 128 | 2,097,152 | 98,304 | 0.046875 | 0.046875 |
| 8 | 256 | 2,097,152 | 98,304 | 0.046875 | 0.023438 |
| 64 | 512 | 2,097,152 | 98,304 | 0.046875 | 0.011719 |
| 512 | 1,024 | 2,097,152 | 98,304 | 0.046875 | 0.005859 |
| 4,096 | 2,048 | 2,097,152 | 98,304 | 0.046875 | 0.002930 |
| 32,768 | 4,096 | 2,097,152 | 98,304 | 0.046875 | 0.001465 |

## Contrast: strong scaling (fixed N = 1,024)

With N fixed, the subdomain edge shrinks as n = N/P^(1/3) and the ratio **degrades** as P^(1/3):

| P | subdomain edge n | ratio 6/n |
|---|---|---|
| 1 | 1024 | 0.005859 |
| 8 | 512 | 0.011719 |
| 64 | 256 | 0.023438 |
| 512 | 128 | 0.046875 |
| 4,096 | 64 | 0.093750 |
| 32,768 | 32 | 0.187500 |

## Caveats

1. Constant halo volume is not constant halo **time**: message latency is per message, and at very small n latency dominates bandwidth.
2. Global operations (a time-step reduction, a CFL check) add an O(log P) term each step; that is the term that eventually bends weak scaling away from linear (Task 12).
3. Network contention can raise exchange time on very large machines if neighbouring subdomains are not mapped to neighbouring nodes.

**Conclusion:** under Gustafson scaling the stencil's communication-to-computation ratio is constant per processor and improves globally, which is why explicit stencil codes are the textbook example of near-ideal weak scaling.
