# Task 03 — The Serial Fraction Invariance Paradox

## The apparent contradiction

Take a program whose serial fraction is quoted as 1 %. Amdahl says no machine can exceed 100× speedup. Gustafson says 1,000 processors give about 990×. Both cannot describe the same run — and they do not, because "1 %" is measured in two different frames of reference.

## Two frames of reference

Let a run have serial time t_s and parallel-phase time t_p **on P processors**.

* **Gustafson's α** — measured on the parallel machine:

  ```
  α = t_s / (t_s + t_p)
  ```

* **Amdahl's s** — measured on a uniprocessor running the *same* (scaled) problem. One processor needs t_s + P·t_p:

  ```
  s = t_s / (t_s + P·t_p)
  ```

## Converting between them

Divide numerator and denominator of s by (t_s + t_p) and use α:

```
s = α / (α + P(1 − α))                                      (1)
α = P·s / (1 + (P − 1)·s)                                   (2)
```

Both describe one run. Substituting (1) into Amdahl's formula reproduces Gustafson's result exactly:

```
S_Amdahl = 1 / (s + (1 − s)/P)
         = (α + P(1 − α)) / ( α + (1 − α) )      [after substituting (1) and simplifying]
         = α + P(1 − α)
         = S_Gustafson
```

There is one speedup and two ways of describing where the time goes. Moreover

```
s = α / S(P)                                                (3)
```

so Amdahl's s is always smaller than Gustafson's α by exactly the speedup.

## Numerical illustration (α = 1 % on the parallel machine)

| P | S(P) = P − α(P − 1) | equivalent uniprocessor s = α/S |
|---|---|---|
| 10 | 9.91 | 0.1009 % |
| 100 | 99.01 | 0.0101 % |
| 1,000 | 990.01 | 0.00101 % |
| 10,000 | 9,900.01 | 0.000101 % |

A constant α on the parallel machine corresponds to an Amdahl fraction that **shrinks as 1/P**. Amdahl's ceiling 1/s therefore rises with the problem: at P = 1,000 it is about 99,000, far above the 990 actually achieved, so Amdahl's Law is never violated.

## Why this resolves the paradox

Amdahl's Law assumes s is **invariant** as the machine grows. That is true only when the problem is fixed. When the problem is scaled with the machine, the serial work (setup, I/O, a global reduction) typically grows slowly or stays constant while the parallel work grows with P. Measured on a uniprocessor, the serial share then falls towards zero; measured on the parallel machine, it stays at the constant α.

Measuring overhead on the parallel machine is therefore the honest frame for scaled workloads: it is what you can observe with a profiler on the real system, it stays stable as the machine grows, and it predicts the speedup directly. Serial overhead does throttle scaling — but only by the fixed amount α per processor (slope 1 − α), never with a hard ceiling.

## Caveat

The resolution holds only while α really is invariant. If the serial or communication part grows with P (an O(log P) reduction, an all-to-all exchange), α itself rises with P and Gustafson's linear prediction becomes an overestimate (Task 12).
