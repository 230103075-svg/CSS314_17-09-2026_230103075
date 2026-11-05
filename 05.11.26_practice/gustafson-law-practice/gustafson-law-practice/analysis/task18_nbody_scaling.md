# Task 18 — N-Body Gravitational Problem Scaling

## Requirement

Scale the particle count N with processor count P so that the time per iteration stays constant (Gustafson), then see how the serial-to-parallel ratio moves.

## Regime 1 — Direct all-pairs, O(N²)

Each step computes N(N − 1)/2 pair interactions.

```
T(P) ∝ N² / P = const   ⇒   N(P) = N₁·√P
```

* Work grows as P (by construction), but the **particle count grows only as √P**: 4× the processors simulate only 2× the particles.
* Communication: every processor needs every particle position each step (all-gather), O(N) data per processor = O(N₁√P). Computation per processor is constant, so communication *per processor* grows as √P.
* Serial parts — integrating a global time step, gathering diagnostics (total energy, momentum) — are O(N) or O(log P) and are dwarfed by the O(N²/P) computation.

**Serial/parallel ratio:** the truly serial part stays tiny relative to computation, but the all-gather term grows as √P relative to constant compute, so the effective α rises slowly. All-pairs is compute-dense enough that it scales well in practice on moderate P (it is the textbook weak-scaling GPU kernel), yet it wastes the machine on a problem that grows only as √P.

## Regime 2 — Barnes–Hut tree code, O(N log N)

```
T(P) ∝ N log N / P = const   ⇒   N log N ∝ P   ⇒   N(P) ≈ N₁·P / log(P) (to leading order)
```

* The particle count grows almost linearly with P: a far larger simulated system for the same machine.
* But new non-scalable pieces appear:
  1. **Tree construction and the global top levels**: the root and upper levels of the octree are shared by all processors and must be built or replicated, O(P) or O(log N) serial/redundant work.
  2. **Domain decomposition** (space-filling curve / orthogonal recursive bisection): a global sort or partition each step, with O(log P) communication rounds.
  3. **Load imbalance**: clustered particles make some subtrees far deeper; imbalance grows as the simulated structure collapses gravitationally.
  4. **Irregular remote access**: fetching "locally essential trees" from other processors, latency-bound and growing with log P.

**Serial/parallel ratio:** per particle, the computation is far cheaper than all-pairs (log N instead of N), so the *fixed* and *synchronisation* costs above are a larger fraction of each step. The effective α is noticeably larger and grows with P.

## Comparison

| | All-pairs O(N²) | Barnes–Hut O(N log N) |
|---|---|---|
| N for constant time | ∝ √P | ∝ P / log P |
| Compute per particle | O(N) — heavy | O(log N) — light |
| Communication pattern | regular all-gather | irregular tree fetches + sort |
| Effective α under scaling | small, rises ~√P via all-gather | larger, rises via tree top, sort, imbalance |
| Fits Gustafson's linear law | well, but on a slowly growing problem | less well, but on a much larger problem |

## Conclusion

The algorithm with worse complexity looks *better* by Gustafson's metric because its heavy, regular computation drowns its overheads. The efficient algorithm simulates far more particles in the same time but exposes its serial and synchronisation costs, so its scaled speedup curve bends earlier. This is the general lesson: scaled speedup measures how well overhead is amortised, not how much science is done; comparing algorithms requires looking at time-to-solution for the target N as well.
