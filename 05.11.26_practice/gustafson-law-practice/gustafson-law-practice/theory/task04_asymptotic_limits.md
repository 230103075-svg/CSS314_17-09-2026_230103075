# Task 04 — Asymptotic Analysis of Parallel Efficiency

## Gustafson efficiency

```
E_G(P) = S(P)/P = (P − α(P − 1)) / P = 1 − α + α/P
```

### Claim: lim_{P→∞} E_G(P) = 1 − α

**Proof (ε–N).** |E_G(P) − (1 − α)| = α/P. For any ε > 0 choose N > α/ε. Then for all P > N, α/P < α/N < ε. Hence E_G(P) → 1 − α. ∎

### Shape of the approach

* E_G(1) = 1 and E_G is strictly decreasing for α > 0: dE_G/dP = −α/P² < 0.
* The gap to the limit, α/P, halves every time P doubles.
* The limit is **positive** for every α < 1: a constant fraction 1 − α of every processor stays useful no matter how large the machine.

Examples: α = 0.04 → limit 0.96; α = 0.10 → limit 0.90; α = 0.50 → limit 0.50.

## Amdahl efficiency (same fraction s, fixed problem)

```
S_A(P) = 1 / (s + (1 − s)/P)
E_A(P) = S_A(P)/P = 1 / (sP + 1 − s)
```

### Claim: lim_{P→∞} E_A(P) = 0 for every s > 0

**Proof.** The denominator sP + 1 − s → ∞ because s > 0, so E_A(P) → 0. More precisely E_A(P) ~ 1/(sP): efficiency decays like the reciprocal of P. ∎

Equivalently S_A(P) → 1/s, so P processors share a bounded speedup and the useful fraction per processor goes to zero.

## Contrast

| | Gustafson (fixed time) | Amdahl (fixed size) |
|---|---|---|
| E(P) | 1 − α + α/P | 1 / (sP + 1 − s) |
| lim E(P) | **1 − α** (positive constant) | **0** |
| Rate of approach | gap ∝ 1/P | E ∝ 1/P itself |
| S(P) as P → ∞ | grows linearly, slope 1 − α | bounded by 1/s |

## Deduction

The two limits differ because of what grows with P. Under Amdahl the serial time is fixed while the parallel time per processor shrinks to zero, so the serial time ends up dominating every processor's day. Under Gustafson the parallel time per processor is fixed by construction, so the serial share α never grows and efficiency settles at 1 − α. Weak scaling can therefore keep large machines productive; strong scaling cannot, beyond P ≈ 1/s.
