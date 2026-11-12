## Phase 3: Roofline & Cache-Blocked GEMM (N=256, p=Pphys=8)

### 3A Ceilings

| Ceiling | Value |
|---|---|
| BW_sustained (copy, Pphys workers) | 5.83 GB/s (median 22.0 ms) |
| P_sustained (16-var scalar FMA, Pphys workers) | 0.4250 GFLOP/s (median 1129.5 ms) |
| Ridge point I* | 0.0729 FLOP/B |

### 3B GEMM variants

| Variant | median ms | GFLOP/s | I (FLOP/B, assumed) | Roofline bound | % of bound |
|---|---|---|---|---|---|
| 1 naive i-j-k | 450.7 | 0.0744 | 0.125 | 0.42 | 17.52% |
| 2 reordered i-k-j | 497.5 | 0.0674 | 0.25 | 0.42 | 15.87% |
| 3 blocked BS*=128 | 449.0 | 0.0747 | 10.67 | 0.42 | 17.59% |

### Tile sweep (Variant 3)

| BS | median ms | GFLOP/s | W_tile (B) | I_tiled = BS/12 |
|---|---|---|---|---|
| 8 | 602.9 | 0.0557 | 1536 | 0.67 |
| 16 | 547.8 | 0.0612 | 6144 | 1.33 |
| 32 | 520.2 | 0.0645 | 24576 | 2.67 |
| 64 | 499.1 | 0.0672 | 98304 | 5.33 |
| 96 | 472.2 | 0.0711 | 221184 | 8.00 |
| 128 | 449.0 | 0.0747 | 393216 | 10.67 |

max |C3 - C2| over all BS = 0.00e+00 (PASS, threshold 1e-9)

### Analysis

- BS* = 128: W_tile = 393216 B (384.0 KiB), fits L2 (7.0 MiB). The 3 tiles (A, B, C) should live in L1d/L2 so the inner j-loop reuses B and C rows from cache.
- Variant 1 walks B with stride N*8 = 2048 B (one new cache line per k step); Variant 2 makes the inner loop unit-stride and hoists a_ik; Variant 3 adds reuse of tiles, raising operational intensity to ~BS/12.
- All GEMM points sit far below the roofline: CPython interpreter overhead, not memory or FPU, is the bottleneck (P_sustained itself is measured through the same interpreter, so it is an interpreter-limited ceiling). Cache blocking gains therefore look smaller than in C. Assumed I for Variants 1-2 is a rough estimate.

```
VERIFY_HASH[PHASE3]: 44E40AE05A78D9EE (P_log=12)
```
