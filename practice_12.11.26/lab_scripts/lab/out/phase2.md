## Phase 2: Strong vs Weak Scaling

### Strong (N=1200000)

| p | T_p ms | S_p | E_p |
|---|---|---|---|
| 1 | 3002.5 | 1.00 | 1.00 |
| 2 | 1498.9 | 2.00 | 1.00 |
| 3 | 519.5 | 5.78 | 1.93 |
| 4 | 451.8 | 6.65 | 1.66 |
| 5 | 397.3 | 7.56 | 1.51 |
| 6 | 364.2 | 8.24 | 1.37 |
| 7 | 384.0 | 7.82 | 1.12 |
| 8 | 345.6 | 8.69 | 1.09 |
| 9 | 353.0 | 8.51 | 0.95 |
| 10 | 827.8 | 3.63 | 0.36 |
| 11 | 349.0 | 8.60 | 0.78 |
| 12 | 344.5 | 8.71 | 0.73 |

Stage A(1) = 30.3 ms, s = 0.0101, Amdahl cap S_inf = 1/s = 99.2; observed S_Plog = 8.71 (9% of the cap). At this p the cap is not the limit; hardware (SMT sharing, turbo drop, pool overhead) is.

### Weak (N0=200000 per worker)

| p | N(p) | T_weak ms | E_weak | alpha | S_Gustafson |
|---|---|---|---|---|---|
| 1 | 200000 | 177.8 | 1.00 | 0.0262 | 1.00 |
| 2 | 400000 | 197.2 | 0.90 | 0.0521 | 1.95 |
| 3 | 600000 | 225.7 | 0.79 | 0.0661 | 2.87 |
| 4 | 800000 | 311.7 | 0.57 | 0.1630 | 3.51 |
| 5 | 1000000 | 401.5 | 0.44 | 0.1562 | 4.38 |
| 6 | 1200000 | 426.5 | 0.42 | 0.1829 | 5.09 |
| 7 | 1400000 | 1047.7 | 0.17 | 0.0324 | 6.81 |
| 8 | 1600000 | 422.0 | 0.42 | 0.0988 | 7.31 |
| 9 | 1800000 | 480.6 | 0.37 | 0.1027 | 8.18 |
| 10 | 2000000 | 531.5 | 0.33 | 0.0951 | 9.14 |
| 11 | 2200000 | 548.2 | 0.32 | 0.1135 | 9.86 |
| 12 | 2400000 | 677.4 | 0.26 | 0.0974 | 10.93 |

### Architectural inflection

Marginal speedup S(9) - S(8) = -0.18 versus an average of 1.10 per added process below Pphys (-17% of earlier gain): beyond 8 physical cores extra processes land on SMT siblings (or E-cores) that share execution units and L1/L2, so returns diminish.

```
VERIFY_HASH[PHASE2]: 21EBB6B918199699 (P_log=12)
```
