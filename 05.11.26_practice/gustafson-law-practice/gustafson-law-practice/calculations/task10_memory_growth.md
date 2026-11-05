# Task 10 — Memory Growth Rate vs. Scaled Speedup

## Formulation

Under weak scaling W(P) = P·W(1), and the aggregate memory footprint is M(P) = c·P: each node holds a constant amount of data. For a 3-D grid of N × N × N points with b bytes per point:

```
M(N) = b·N³
```

With per-node memory capped at m = 32 GB, P nodes provide P·m bytes, so

```
b·N³ ≤ P·m    ⇒    N_max(P) = ⌊ (P·m / b)^(1/3) ⌋
```

N_max grows only as **P^(1/3)**: 8× more nodes doubles the grid edge.

**Assumptions.** 32 GB is taken as 32 GiB = 32 × 2³⁰ = 34,359,738,368 bytes, and the whole node memory is available to the grid (an upper bound; a real code leaves room for the OS, buffers and halos).

## Capacity limits

| Data per grid point | Nodes | Aggregate memory | Max grid points | N_max | memory used |
|---|---|---|---|---|---|
| 1 field, float64 (8 B/point) | 64 | 2,199,023,255,552 B | 274,877,906,944 | **6,501** | 99.9541 % |
| 1 field, float64 (8 B/point) | 512 | 17,592,186,044,416 B | 2,199,023,255,552 | **13,003** | 99.9772 % |
| 5 fields, float64 (40 B/point) | 64 | 2,199,023,255,552 B | 54,975,581,388 | **3,802** | 99.9693 % |
| 5 fields, float64 (40 B/point) | 512 | 17,592,186,044,416 B | 439,804,651,110 | **7,604** | 99.9693 % |

## Observations

* For one float64 field: 64 nodes hold a grid of edge 6,501 and 512 nodes an edge of 13,003, i.e. 2× the edge for 8× the nodes (the exact cube roots are 2^(38/3) ≈ 6,501.99 and 2^(41/3) ≈ 13,003.99; flooring to whole grid points gives the table values).
* Per node the share is always 32 GiB / b points: ≈ 4,294,967,296 points (an edge of 1,625) for 8 B/point.
* Consequence for scaled speedup: if the computation per grid point is constant, work ∝ N³ ∝ P, so Gustafson's linear W(P) = P·W(1) is achievable exactly up to the memory ceiling. If the algorithm's work grows faster than N³ (e.g. more time steps on a finer mesh, which CFL stability typically requires, ∝ N), the work becomes ∝ N⁴ ∝ P^(4/3) and the runtime can no longer stay constant: memory, not processor count, then limits the scaled problem (Sun–Ni, Task 11).
