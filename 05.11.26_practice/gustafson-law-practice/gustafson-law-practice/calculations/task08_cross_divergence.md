# Task 08 — Cross-Over Point: Amdahl vs. Gustafson (s = α = 0.08, P = 128)

## Amdahl (fixed size)

```
S_A = 1 / (s + (1 − s)/P)
    = 1 / (0.08 + 0.92/128)
    = 1 / (0.08 + 0.0071875)
    = 1 / 0.0871875
    = 11.469534
```

## Gustafson (fixed time)

```
S_G = P − α(P − 1)
    = 128 − 0.08 × 127
    = 128 − 10.16
    = 117.84
```

## Discrepancy

| Measure | Value |
|---|---|
| Ratio S_G / S_A | 10.274175 |
| Absolute difference S_G − S_A | 106.370466 |
| Percentage discrepancy relative to Amdahl, (S_G − S_A)/S_A | 927.4175 % |
| Percentage discrepancy relative to Gustafson, (S_G − S_A)/S_G | 90.2669 % |
| Efficiency, Amdahl | 0.089606 |
| Efficiency, Gustafson | 0.920625 |

## Why they diverge as P grows

* Amdahl's speedup approaches the ceiling 1/s = 12.5; at P = 128 it is already at 91.76 % of that ceiling.
* Gustafson's speedup keeps growing with slope 1 − α = 0.92.
* Their ratio grows roughly in proportion to P: for large P, S_G/S_A ≈ s(1 − s)P.

| P | S_A | S_G | S_G / S_A |
|---|---|---|---|
| 2 | 1.8519 | 1.92 | 1.0368 |
| 8 | 5.1282 | 7.44 | 1.4508 |
| 32 | 9.1954 | 29.52 | 3.2103 |
| 128 | 11.4695 | 117.84 | 10.2742 |
| 512 | 12.2254 | 471.12 | 38.5361 |
| 2,048 | 12.4302 | 1884.24 | 151.5856 |

The divergence is not a contradiction. The same 8 % is measured in different frames (Task 03): Amdahl's s is the serial share of a fixed problem on one processor; Gustafson's α is the serial share of a problem scaled to fill 128 processors. For the scaled problem the uniprocessor serial share is only α/S_G = 0.000679, so Amdahl's ceiling for that problem is far above 117.84.
