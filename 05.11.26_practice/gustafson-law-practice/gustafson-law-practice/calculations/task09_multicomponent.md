# Task 09 — Multi-Component Workload Partitioning

## Phase breakdown (shares of total parallel execution time)

| Phase | Share | Classification |
|---|---|---|
| (A) Initial disk I/O and parameter broadcast | 2 % | serial: one reader, then a broadcast; does not shrink with P |
| (B) Boundary message-passing synchronisation | 6 % | non-scaling: time spent waiting/exchanging, not doing domain work |
| (C) Core domain flux computation | 92 % | parallel: work divided across all processors |

Only (C) is useful parallel work that scales with the problem. Both (A) and (B) are time during which the machine is not doing scalable computation, so in Gustafson's model both count towards the serial fraction:

```
α = α_A + α_B = 0.02 + 0.06 = 0.08
p = 1 − α     = 0.92
```

## Composite scaled speedup

| P | S(P) = P − α(P − 1) | E(P) |
|---|---|---|
| 512 | 512 − 0.08·511 = 471.12 | 0.920156 |
| 2,048 | 2048 − 0.08·2047 = 1,884.24 | 0.920039 |

## Analysis

* At P = 2,048 the simulation delivers 1,884.24× the work of one processor in the same wall-clock time; 92 % efficiency is retained.
* The synchronisation phase is **three times** as costly as the I/O phase: reducing (B) from 6 % to 3 % would raise S(2048) to 1,945.65, while removing (A) entirely would only reach 1,925.18.
* Caveat: (B) is unlikely to stay at exactly 6 % as P grows; halo exchanges stay roughly constant per processor, but any global synchronisation grows as O(log P) (Task 12). The figures above are the classical model with α held constant.
