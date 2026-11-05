# Task 06 — Parametric Speedup Across Processor Densities

Serial fraction on the parallel machine: **α = 0.04**.

```
S(P) = P − α(P − 1)
E(P) = S(P)/P = 1 − α(P − 1)/P
```

| P | S(P) | E(P) | efficiency lost | E(P) − (1 − α) |
|---|---|---|---|---|
| 16 | 16 − 0.04·15 = 15.40 | 0.962500 | 3.7500 % | 0.250000 pp |
| 64 | 64 − 0.04·63 = 61.48 | 0.960625 | 3.9375 % | 0.062500 pp |
| 256 | 256 − 0.04·255 = 245.80 | 0.960156 | 3.9844 % | 0.015625 pp |
| 1,024 | 1024 − 0.04·1023 = 983.08 | 0.960039 | 3.9961 % | 0.003906 pp |
| 4,096 | 4096 − 0.04·4095 = 3,932.20 | 0.960010 | 3.9990 % | 0.000977 pp |
| 16,384 | 16384 − 0.04·16383 = 15,728.68 | 0.960002 | 3.9998 % | 0.000244 pp |

## Evaluation

* S(P) grows **linearly** with slope 1 − α = 0.96: each 4× increase in P gives almost exactly 4× the scaled speedup.
* E(P) falls only from 0.9625 (P = 16) to 0.960002 (P = 16,384). Efficiency **does not degrade** as P grows exponentially: it converges to the limit 1 − α = 0.96 (Task 04), and the remaining gap α/P halves each time P doubles.
* The efficiency lost is almost entirely the constant α = 4 %: the serial phase idles every processor but one for 4 % of the run, whatever P is.
* Contrast with Amdahl at s = 0.04: S could never exceed 25, so E at P = 16,384 would be below 0.0016.
