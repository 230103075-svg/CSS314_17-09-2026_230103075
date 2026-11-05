# Task 12 — Communication Overhead Penalty Integration

## Why classical Gustafson is incomplete

Classical scaled speedup S = P − α(P − 1) assumes the parallel run stays at s + p = 1 for every P. In distributed systems, part of each step is spent communicating: halo exchanges, global reductions, barriers. Those costs typically **grow with P** even when the work per node is constant, so the parallel runtime drifts above 1.

## Model

Keep Gustafson's normalisation for the computation and add an overhead term C(P) with C(1) = 0:

```
T_P = s + p + C(P) = 1 + C(P)
T_1 = s + P·p                       (equivalent serial work: no communication on one node)

S_C(P) = (s + P·p) / (1 + C(P)) = (P − α(P − 1)) / (1 + C(P))
E_C(P) = S_C(P)/P
```

The classical law is the special case C ≡ 0.

## Two common overhead shapes

**Tree-structured collectives (reductions, broadcasts, barriers):**

```
C(P) = γ · log₂ P
S_C(P) = (P − α(P − 1)) / (1 + γ log₂ P)
```

**Communication with √P growth** (e.g. 2-D process grid where a row or column of √P processes takes part in each step, or contention growing with network diameter):

```
C(P) = β · √P
S_C(P) = (P − α(P − 1)) / (1 + β √P)
```

## Asymptotics

| Overhead | S_C(P) for large P | Efficiency limit |
|---|---|---|
| none | ≈ (1 − α)P | 1 − α |
| γ log₂P | ≈ (1 − α)P / (γ log₂P) | → 0, but slowly (∝ 1/log P) |
| β √P | ≈ (1 − α)√P / β | → 0 (∝ 1/√P) |

With logarithmic overhead the speedup is still unbounded, merely sub-linear. With √P overhead the speedup still grows, but only as √P: quadrupling the machine doubles the throughput.

## An equivalent "effective serial fraction"

Rewriting S_C in Gustafson form defines the serial fraction an experimenter would infer:

```
α_eff(P) = (P − S_C(P)) / (P − 1)
```

Because C(P) grows, α_eff rises with P even though the code's serial part is constant. This is exactly the rising s_emp / Karp–Flatt signature seen in the lab benchmarks: overhead shows up as an apparently growing serial fraction.

## Worked example (α = 0.02, γ = 0.01, β = 0.005)

| P | classical | γ log₂P model | β√P model |
|---|---|---|---|
| 64 | 62.74 | 59.19 | 60.33 |
| 1,024 | 1,003.54 | 912.31 | 865.12 |
| 16,384 | 16,056.34 | 14,084.51 | 9,790.45 |

(Values from S_C = (P − 0.02(P − 1)) / (1 + C(P)).)

## Design implications

1. Keep C(P) logarithmic: tree or recursive-doubling collectives instead of linear gathers.
2. Overlap communication with computation, which removes it from the critical path and shrinks the effective C(P).
3. Grow the work per node: since S_C divides by 1 + C(P), larger computation per step makes a fixed absolute communication cost a smaller relative C(P).
