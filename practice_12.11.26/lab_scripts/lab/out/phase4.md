## Phase 4: Skewed Workload Scheduling (N=8000, p=Pphys=8)

| Strategy | T_wall ms | lambda |
|---|---|---|
| Block static | 6345.6 | 2.342 |
| Cyclic static | 4248.6 | 1.020 |
| Dynamic C=1 | 4048.7 | 1.006 |
| Dynamic C=8 | 4368.6 | 1.010 |
| Dynamic C=64 | 4691.6 | 1.050 |
| Dynamic C=512 | 4993.6 | 1.414 |

### Analysis

- Block static gives the last worker the heaviest items (K grows ~ i^2), so lambda = 2.34 and T_wall is set by the slowest worker. Cyclic static interleaves cheap and heavy items: lambda = 1.02.
- C=1 maximizes balance but every item takes the lock on the shared counter (synchronization + cache-line bouncing of the counter); C=512 amortizes the lock but the final chunk can be a large block of heavy items (tail imbalance, lambda = 1.41).
- Optimal C* = 1 (T_wall = 4048.7 ms, lambda = 1.01): small enough to keep the tail short, large enough that lock cost per item is negligible next to the work per chunk.

```
VERIFY_HASH[PHASE4]: 472526913700F9B5 (P_log=12)
```
