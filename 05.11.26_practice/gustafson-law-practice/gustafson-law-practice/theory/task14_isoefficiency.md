# Task 14 — Isoefficiency Relation with Gustafson Scaling

## Isoefficiency (Kumar & Grama)

Let W be the problem size measured in serial work (time on one processor) and T_o(W, P) the **total overhead**: all time spent by P processors that a serial run would not spend (idle, communication, extra computation):

```
P · T_P = W + T_o(W, P)
E = W / (P · T_P) = 1 / (1 + T_o(W, P)/W)
```

Holding E constant requires W/T_o constant:

```
W = K · T_o(W, P),     K = E / (1 − E)
```

The **isoefficiency function** is the growth of W with P that satisfies this. Small growth (e.g. Θ(P)) means a highly scalable system.

## Express Gustafson's model in these terms

Gustafson normalises the parallel run to s + p = 1 with serial fraction α = s. Work on one processor is W = s + P·p. The serial part occupies one processor while the other P − 1 are idle, so the overhead is the idle time of those processors:

```
T_o = (P − 1)·s = (P − 1)·α
W   = α + P(1 − α)
E   = W / (P · 1) = (α + P(1 − α)) / P = 1 − α(P − 1)/P
```

which reproduces Gustafson's efficiency (Task 04) — the two frameworks agree.

## Isoefficiency of a Gustafson-compliant program

Gustafson compliance means α is constant in absolute time while the parallel work per processor stays constant. Then

* T_o(W, P) = (P − 1)α = Θ(P),
* to keep E fixed, W must grow as K · Θ(P).

**Result:** the isoefficiency function of a Gustafson-compliant weak-scaling program is

```
W(P) = Θ(P)
```

**Proof.** E = W/(W + T_o). With T_o = (P − 1)α, E is held at E₀ exactly when W = K(P − 1)α with K = E₀/(1 − E₀), which is linear in P. Conversely, Gustafson's prescription W(P) = α + P(1 − α) is linear in P and yields E → 1 − α > 0, a constant. Linear growth of W is therefore both necessary and sufficient. ∎

This is the best isoefficiency possible for any program with a non-zero overhead per processor, since T_o cannot be smaller than Θ(P) once each added processor contributes even a constant amount of idle or coordination time.

## Critique

1. **The Θ(P) result is only as good as the assumption that T_o is Θ(P).** With a tree reduction per step, T_o = Θ(P log P) and the isoefficiency becomes Θ(P log P): the problem must grow faster than linearly to hold efficiency, and Gustafson's linear scaling would let efficiency slowly fall (consistent with Task 12).
2. **Memory may forbid it.** If W grows faster than memory (W ∝ N³, memory ∝ N², as in GEMM), the isoefficiency growth may exceed what P nodes can store. Isoefficiency can then demand a problem the machine cannot hold — Sun–Ni's concern (Task 11).
3. **Gustafson fixes time; isoefficiency fixes efficiency.** They coincide only when T_o is linear in P. In general, isoefficiency is the stricter, more informative criterion because it tells you *how fast* the problem must grow, while Gustafson simply asserts it grows linearly.
