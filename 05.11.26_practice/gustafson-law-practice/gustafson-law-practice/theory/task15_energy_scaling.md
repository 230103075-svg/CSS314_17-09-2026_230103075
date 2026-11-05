# Task 15 — Energy and Power Proportionality under Scaled Speedup

## Power model

Each node draws

* P_idle — static power whenever it is powered on (leakage, memory refresh, uncore),
* P_active — additional dynamic power while it computes.

## Gustafson run on P nodes

Normalise the parallel runtime to T = s + p = 1 time unit (serial fraction α = s).

* During the serial phase (duration α) one node is active and P − 1 nodes idle.
* During the parallel phase (duration 1 − α) all P nodes are active.

```
E_par(P) = P·P_idle·T + P_active·[α·1 + (1 − α)·P]·T
         = P·P_idle + P_active·(α + (1 − α)P)          (T = 1)
```

The second term is P_active × total useful work (in node-time units), the first is static power for every node over the whole run.

## Equivalent work on one node

The same scaled problem on one node takes T₁ = S(P) = α + (1 − α)P time units (Task 02):

```
E_seq(P) = (P_idle + P_active)·(α + (1 − α)P)
```

## Comparison

```
E_seq − E_par = P_idle·(α + (1 − α)P) − P·P_idle
             = P_idle·(α − αP)
             = −α(P − 1)·P_idle
```

* The **dynamic energy is identical** in both cases: the same work is done, and P_active × work is independent of how it is spread.
* The **static energy differs** by α(P − 1)·P_idle: the parallel run pays for P − 1 nodes idling through the serial phase.

So for identical work, the parallel run uses slightly **more** energy, by a fraction that depends only on α and the static-to-dynamic power ratio, while finishing S(P) times sooner.

## "Identical wall-clock runtime" comparison

The task fixes runtime. In time T = 1:

* one core does 1 unit of work and uses E₁ = P_idle + P_active;
* P cores do S(P) = α + (1 − α)P units and use E_par(P) above.

Energy per unit of work:

```
e₁     = P_idle + P_active
e_P    = [P·P_idle + P_active·S(P)] / S(P) = P_active + P_idle·P/S(P)
```

Since P/S(P) = 1/E(P) ≥ 1, the parallel machine spends a little more energy per unit of work: the static term is inflated by 1/E(P) = 1/(1 − α(P − 1)/P), at most 1/(1 − α).

### Example (α = 0.04, P = 1,024, P_idle = 40 W, P_active = 60 W, T = 1 h)

| | Work done | Energy | Energy per unit work |
|---|---|---|---|
| 1 core | 1 | 100 Wh | 100.00 Wh |
| 1,024 cores | S = 983.08 | 40,960 + 58,984.8 = 99,944.8 Wh | 101.66 Wh |

The 1,024-node machine does 983.08× the work in the same hour for 1.66 % more energy per unit of work.

## Conclusions

1. Under Gustafson scaling, total energy grows linearly with P because the work grows linearly with P — the expected cost of solving a bigger problem.
2. Energy efficiency is governed by the static power wasted during serial phases: the overhead factor is 1/E(P), bounded by 1/(1 − α).
3. Energy proportionality (P_idle → 0) removes the penalty entirely; power-gating idle nodes during serial phases or shrinking α both improve energy per unit work.
