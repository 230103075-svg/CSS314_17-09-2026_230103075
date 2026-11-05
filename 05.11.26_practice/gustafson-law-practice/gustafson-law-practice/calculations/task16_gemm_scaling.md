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
| 1 | 4,096 | 0.137 | 137.439 | 0.375 | 384.000 |
| 8 | 8,192 | 1.100 | 137.439 | 1.500 | 192.000 |
| 64 | 16,384 | 8.796 | 137.439 | 6.000 | 96.000 |
| 512 | 32,768 | 70.369 | 137.439 | 24.000 | 48.000 |
| 4,096 | 65,536 | 562.950 | 137.439 | 96.000 | 24.000 |
| 32,768 | 131,072 | 4,503.600 | 137.439 | 384.000 | 12.000 |

Every row does the same 137.439 GFLOP per processor, so runtime stays constant, while memory per processor halves for every 8× increase in P.
