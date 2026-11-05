# Task 20 — Parallel Graph Traversal & Irregular Workloads

Dense linear algebra and stencils are Gustafson's best case: work per processor is predictable, data access is regular, and communication is local and bounded. Graph analytics on power-law (scale-free) graphs — BFS, PageRank, connected components — break each of those properties.

## 1. Work per processor is not controllable

Weak scaling assumes we can give each processor a constant amount of work. In a graph the work of a vertex is proportional to its degree, and in a power-law graph the degree distribution is P(k) ∝ k^(−γ) with γ ≈ 2–3: a few hub vertices have millions of edges while most have a handful.

* Partitioning vertices evenly gives wildly uneven edge counts; partitioning edges evenly (2-D / edge-cut partitioning) splits hubs across processors and creates extra communication.
* As the graph grows with P, the largest hub grows too (the maximum degree of a scale-free graph grows with graph size), so the **load imbalance gets worse** as the problem is scaled — the opposite of the averaging Gustafson relies on.

## 2. Frontier imbalance and data-dependent synchronisation (BFS)

BFS proceeds level by level with a global barrier between levels.

* Frontier size varies by orders of magnitude between levels: a few vertices, then a sudden explosion through hubs, then a long tail.
* Each level's runtime is set by the processor with the most frontier edges; the rest wait at the barrier.
* The number of levels depends on the graph's diameter, which grows (slowly, ~log N) with graph size, so the number of global synchronisations grows with the scaled problem.

Under Gustafson the serial/synchronisation fraction is therefore not invariant: it depends on the graph's shape, which changes as the graph grows.

## 3. Irregular memory access

* Visiting neighbours means random accesses into vertex arrays: almost every access is a cache miss and many are TLB misses. Performance is bound by **memory latency**, not bandwidth or flops.
* Arithmetic intensity is tiny (a few operations per 8–16 bytes fetched), so graph kernels sit at the far left of the roofline.
* Hardware prefetchers cannot predict the pattern, so adding cores mostly adds outstanding misses to an already saturated memory system.

## 4. Communication volume grows with the cut

Power-law graphs have no small separators: any partition into P parts cuts a large fraction of edges, often most of them. Each BFS level or PageRank iteration sends messages along every cut edge:

* communication per processor ≈ (cut edges)/P, and the cut fraction approaches 1 − 1/P for random-like graphs, so most edge traversals become remote;
* traffic is all-to-all and bursty, stressing network bisection bandwidth exactly as the parallel FFT does (Task 13).

Compare a 3-D stencil, where communication per processor is a constant 6n² surface (Task 17).

## 5. PageRank specifically

PageRank is a sparse matrix–vector product repeated to convergence. It inherits the irregular access and all-to-all communication, plus:

* a global reduction per iteration (convergence check, dangling-node mass): O(log P) each time;
* iteration counts that depend on the damping factor and graph structure, not on P.

## Why dense linear algebra is different

| Property | Dense LA / stencil | Power-law graph |
|---|---|---|
| Work per processor | exactly controllable | set by degree distribution, skewed |
| Access pattern | regular, prefetchable | random, latency-bound |
| Arithmetic intensity | high (GEMM) or moderate | very low |
| Communication | local or structured, bounded | all-to-all, proportional to the cut |
| Growth with problem | overheads amortised | imbalance and cut grow |

## Mitigations

Degree-aware 2-D partitioning and hub replication (vertex-cut), direction-optimising BFS (switching top-down/bottom-up to shrink work on huge frontiers), asynchronous / bulk-synchronous-free execution, and message aggregation. They improve weak scaling considerably, but graph analytics remains the standard example where measured scaled speedup sits well below Gustafson's line: the Graph500 benchmark exists precisely because LINPACK-style scaling does not predict it.
