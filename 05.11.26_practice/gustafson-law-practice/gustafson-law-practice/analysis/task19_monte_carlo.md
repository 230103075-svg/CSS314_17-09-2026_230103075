# Task 19 — Monte Carlo Path Simulation: the Ideal Weak-Scaling Model

Monte Carlo is the canonical embarrassingly parallel workload: each processor draws its own samples, computes them independently and contributes a partial sum. Under weak scaling (constant samples per processor, total samples ∝ P) it comes closer to S = P than almost anything else. Yet α is never exactly zero. The non-zero sources:

## 1. PRNG seed management

Each processor needs an **independent** random stream.

* **Naive**: the master draws P seeds serially and sends them out — O(P) serial work plus a scatter.
* **Correct and scalable**: counter-based or splittable generators (Philox, ThreeFry, NumPy `SeedSequence.spawn`, jump-ahead in xoshiro / MT) where processor k derives its stream from (global seed, k) locally in O(1). No communication at all.
* **Hazard**: seeding every worker identically (or from the clock at the same instant) gives correlated or identical streams. The run still "scales" but the estimate has the statistical power of one processor — a correctness failure that a speedup number never reveals.

The lab experiment on PRNG state locking showed the other failure mode: a *shared* generator behind a lock serialises sample generation entirely.

## 2. Reduction of partial results

The final estimate needs the sum (and usually the sum of squares, for the error bar) over all processors.

* **Linear reduction** at the master: P − 1 sequential receives, time ∝ P.
* **Tree reduction** (MPI_Reduce, recursive doubling): ⌈log₂ P⌉ rounds.

### Effect on S(P)

Normalise computation per processor to 1 and let one reduction round cost r (latency plus a few doubles of bandwidth):

```
linear:  T_P = 1 + r(P − 1)       S_lin(P)  = P / (1 + r(P − 1))
tree:    T_P = 1 + r·⌈log₂ P⌉     S_tree(P) = P / (1 + r·⌈log₂ P⌉)
```

(The serial reference does the same P units of sampling with no communication.)

With r = 10⁻⁴ (a 10 µs message against a 0.1 s sampling phase):

| P | S_lin | S_tree | ideal |
|---|---|---|---|
| 64 | 63.60 | 63.96 | 64 |
| 1,024 | 928.97 | 1,022.98 | 1,024 |
| 16,384 | 6,210.06 | 16,361.09 | 16,384 |
| 131,072 | 9,291.21 | 130,849.56 | 131,072 |

A linear gather becomes the bottleneck as soon as rP approaches 1; the tree keeps S within 0.2 % of ideal even at 131,072 processors. The effective serial fraction of the tree version is α_eff ≈ r·log₂P — tiny and only logarithmically growing.

## 3. Output serialisation

* Writing per-sample paths (for option-pricing audits or particle tracks) through one process makes output time ∝ total samples ∝ P: a serial term that grows with the problem. Parallel I/O (MPI-IO, one file per rank, HDF5) restores scalability.
* Writing only the reduced estimate is O(1) and harmless.

## 4. Other minor terms

* Start-up and parameter broadcast: O(log P), once per run.
* Variance-reduction schemes that need global statistics (control variates fitted across all paths) add a reduction per iteration.
* Load imbalance from variable path lengths (early-exercise options, absorbed particles) — handled by over-decomposition or dynamic work queues.

## Summary

With counter-based PRNGs, tree reductions and parallel or reduced output, Monte Carlo's α is effectively r·log₂P, so it tracks Gustafson's line almost perfectly. Each naive choice — serial seeding, a linear gather, single-writer output — reintroduces a term that grows with P and is exactly the kind of overhead that makes measured α rise with machine size.
