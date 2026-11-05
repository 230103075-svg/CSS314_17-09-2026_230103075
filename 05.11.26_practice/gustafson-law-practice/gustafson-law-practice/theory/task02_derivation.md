# Task 02 — Formal Derivation of Scaled Speedup

**Goal.** Derive S(P) = s + p·P and prove S(P) = P − α(P − 1).

## Step 1 — Normalise the parallel run

Measure the program on the parallel machine with P processors. Split its wall-clock time into

* s — time spent in code that only one processor executes (serial part),
* p — time spent in code that all P processors execute concurrently (parallel part).

Normalise the parallel runtime to one unit:

```
T_P = s + p = 1                                             (1)
```

*Justification:* only ratios of times matter for speedup, so any unit may be chosen.

## Step 2 — Determine the equivalent serial workload

Ask how long **one** processor would need to do the same total work.

* The serial part is done by one processor already, so it costs s on a single processor as well.
* During the parallel part, P processors each work for time p. One processor must do all P shares one after another, costing P·p.

```
T_1 = s + P·p                                               (2)
```

*Justification:* work is conserved; the parallel phase contains P·p processor-seconds of work, and one processor delivers one processor-second per second.

## Step 3 — Define scaled speedup

```
S(P) = T_1 / T_P = (s + P·p) / (s + p)                      (3)
```

Substitute (1), s + p = 1:

```
S(P) = s + p·P                                              (4)
```

which is the scaled-speedup formula.

## Step 4 — Express it with the serial fraction α

On the parallel machine the serial fraction is the share of the normalised run spent serially:

```
α = s / (s + p) = s          (because s + p = 1)            (5)
p = 1 − α                                                   (6)
```

Substitute (5) and (6) into (4):

```
S(P) = α + (1 − α)·P                                        (7)
     = α + P − α·P                                          (8)
     = P − α·P + α                                          (9)
     = P − α(P − 1)                                         (10)  ∎
```

*Line-by-line:* (7) substitution; (8) expand the product; (9) reorder; (10) factor α out of −αP + α.

## Step 5 — Checks

| Case | Formula gives | Meaning |
|---|---|---|
| α = 0 | S = P | perfectly parallel: ideal linear speedup |
| α = 1 | S = 1 | entirely serial: extra processors do nothing |
| P = 1 | S = 1 | one processor compared with itself |
| dS/dP | 1 − α | each added processor adds (1 − α) of a processor's worth of speed |

The last line is the essential result: scaled speedup is a **straight line** in P with slope 1 − α and no asymptote.

## Step 6 — Efficiency

```
E(P) = S(P)/P = 1 − α(P − 1)/P   →   1 − α   as P → ∞
```

(The limit is proven in Task 04.)
