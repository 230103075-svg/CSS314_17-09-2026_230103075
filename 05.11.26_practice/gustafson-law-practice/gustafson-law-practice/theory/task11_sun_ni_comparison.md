# Task 11 — Sun–Ni Memory-Bounded Speedup vs. Gustafson's Law

## The memory-bounded model

Sun and Ni (1990, 1993) observed that the largest problem a machine can run is limited by **memory**, and that memory grows linearly with P. Let the parallel workload grow with the memory capacity by a factor G(P) when P processors (with P times the memory) are used. With serial fraction s and parallel fraction 1 − s measured on one processor:

```
            s + (1 − s)·G(P)
S_SN(P) = ─────────────────────
           s + (1 − s)·G(P)/P
```

* numerator — work of the scaled problem, done serially;
* denominator — time on P processors: the serial part, plus the scaled parallel work divided over P.

## Amdahl and Gustafson as special cases

| G(P) | Meaning | S_SN(P) |
|---|---|---|
| G = 1 | work fixed | 1 / (s + (1 − s)/P) — **Amdahl** |
| G = P | work grows with P | s + (1 − s)·P — **Gustafson** (fixed time) |
| G > P | work grows faster than P | exceeds Gustafson |
| 1 < G < P | work grows slower than P | between Amdahl and Gustafson |

So Sun–Ni is the general law and the other two are its endpoints.

## Where G comes from: the memory–work relationship

If memory scales as M(N) and work as W(N), filling P times the memory sets N, and G(P) = W(N_P)/W(N_1).

* **Vector or explicit stencil**: M ∝ N, W ∝ N → G(P) = P → Gustafson exactly.
* **Dense matrix multiplication**: M ∝ N², W ∝ N³ → N grows as √P, so G(P) = P^{3/2} > P.
* **Dense LU / Gaussian elimination**: same, G(P) = P^{3/2}.
* **Sorting, FFT**: M ∝ N, W ∝ N log N → G(P) = P·log(P N₁)/log N₁, slightly above P.
* **Algorithms with super-linear memory**, e.g. an all-pairs table M ∝ N², W ∝ N² → G(P) = P.

## When Gustafson overestimates or underestimates

**Gustafson underestimates** achievable *scaled* throughput when G(P) > P: memory-bound scaling grows the work faster than P. For GEMM with s = 1 % and P = 64, G = 512:

```
S_SN = (0.01 + 0.99·512) / (0.01 + 0.99·512/64) = 506.89 / 7.93 ≈ 63.9
S_G  = 0.01 + 0.99·64 = 63.37
```

The gain is modest here, but it grows with s. The catch is time: with G > P the scaled run takes longer than the original (here 7.93 vs 1.00 time units), so fixed-time scaling is no longer what is happening.

**Gustafson overestimates** when:

1. G(P) < P — the problem cannot grow as fast as the machine (fixed input data, memory filled before the work scales);
2. the serial part grows with problem size, so α measured at small scale is not invariant;
3. communication grows with P (Task 12) — neither Gustafson nor basic Sun–Ni include it;
4. memory per node is the binding constraint and has already been reached, so adding nodes cannot grow the problem as assumed (Task 10).

## Critical assessment

Sun–Ni's contribution is to tie the scaling of the problem to a physical resource, memory, rather than to the analyst's choice. It explains why Gustafson works well for stencils (G = P) and why dense linear algebra scales even better in throughput but worse in time-to-solution. Its limitation is the same as Gustafson's: it treats parallel work as perfectly divisible and ignores the communication and synchronisation that grow with P.
