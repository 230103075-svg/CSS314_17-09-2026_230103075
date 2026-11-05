# Task 13 — Boundary Cases: When Gustafson Fails

Gustafson's premise: grow the problem with P so that work per processor, and therefore runtime, stays constant. It fails whenever growing the problem changes the **algorithm's cost per processor** or adds **synchronisation that grows with P**. Two concrete cases:

## Case 1 — Iterative linear solvers on growing meshes (Jacobi / Conjugate Gradient)

**Setting.** Solve the 2-D Poisson equation on an N × N grid distributed over P processors, with N² ∝ P (constant points per processor).

**Why the premise breaks.**

1. *Iteration count grows with N.* The condition number of the discrete Laplacian is κ = O(N²). Jacobi needs O(N²) iterations to converge and unpreconditioned CG needs O(√κ) = O(N) iterations. Each iteration costs constant time per processor under weak scaling, but the **number of iterations grows** with the problem: with N ∝ √P, CG's runtime grows as O(√P) and Jacobi's as O(P).
2. *Global reductions every iteration.* CG computes two dot products per iteration, each a global all-reduce costing O(log P). That latency is paid once per iteration, and there are O(√P) iterations.

**Result.** T(P) = O(√P · (1 + log P)) instead of constant. The scaled speedup grows only like √P / log P, not linearly. Multigrid, with O(1) iterations, restores near-Gustafson behaviour for exactly this reason: it removes the growth in iteration count.

## Case 2 — Parallel 3-D FFT (distributed transpose)

**Setting.** A 3-D FFT of N³ points over P processors with N³ ∝ P.

**Why the premise breaks.**

1. *Work is super-linear.* FFT costs O(N³ log N). Per processor the computation is O((N³/P) log N), and log N grows with P, so even computation per processor is not constant.
2. *All-to-all communication.* Each 1-D FFT pass needs the data transposed, which is an all-to-all exchange: every processor sends a piece to every other (or to √P of them with a pencil decomposition). Total traffic is O(N³) per transpose and crosses the network's **bisection**, whose bandwidth grows more slowly than P on most networks. Each processor's communication time therefore grows with P.

**Result.** The transpose dominates at scale and runtime rises steadily; FFT-based spectral codes are a well-known example where weak scaling degrades well before strong scaling would be the limit.

## Other failure patterns (brief)

* **Contended synchronisation** — a shared lock or counter (seen in the Karp–Flatt lab, Experiment B): every processor competes for one resource, so the serial part grows with P.
* **Load imbalance on irregular data** — graph frontiers, adaptive meshes (Task 20): growing the problem amplifies the imbalance instead of averaging it out.
* **Non-scalable I/O** — a single file system or master node writing results: output size grows with the problem, so serial I/O time grows with P.

## Common root cause

Gustafson assumes the serial fraction α measured on the parallel machine is **invariant**. Every failure above is a mechanism by which α (or an overhead term C(P), Task 12) **grows** with P: more iterations, larger transposes, longer queues for a shared resource. The test for whether an algorithm is a good weak-scaling candidate is whether its per-processor cost — computation plus communication — is genuinely independent of P.
