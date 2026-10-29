# CS482 Lab Practicum — Memory Hierarchy, Contention and Scaling

**Student ID:** 230103075  ·  **Salted parameters:** N = 1408, seed = 10877
**Host:** 12th Gen Intel Core i5-12450H (Alder Lake, 4 P-cores + 4 E-cores, 8 physical / 12 logical), L1d 48 KB, L2 2 MB, L3 12 MB, 64-byte lines · Windows 11, Python 3.14

## Bottleneck findings

On this machine almost every slowdown traced back to how data moves through the memory hierarchy rather than to arithmetic. The stride benchmark over a 134 MB buffer shows the cache-line cliff directly: reads cost 1.72 ns per element at stride 1 but about 13 ns from stride 16 onward (~8x), because each 64-byte line then serves one double instead of eight (the full cliff appears at 16 rather than 8 because Intel's adjacent-line prefetcher fetches lines in 128-byte pairs). Interchanging the loops of the 704×704 matrix product from i-j-k to i-k-j made the inner loop stride-1 and cut runtime from 767 ms to 179 ms (4.29x) with bit-identical results. In shared memory, eight processes incrementing neighbouring counters on one cache line ran 4.87x slower than a single process and 2.91x slower than the same code with 64-byte padding; the Karp-Flatt metric rose to 0.55–0.63 for the unpadded layout while staying at or below 0.12 when padded, and thread-local privatization was fastest of all. The 1408×1408 stencil reached 1.75x on 2 cores and 2.77x on 4 but only 2.83x on 8, giving a fitted Amdahl serial fraction of 0.202 (ceiling 4.95x); the per-P fraction climbs from 0.14 to 0.26 as cores are added, which marks bandwidth saturation rather than a true serial section, and weak scaling reached only 3.06x at P = 8 for the same reason. The roofline analysis confirms it: the stencil's arithmetic intensity of at most 0.25 FLOP/byte sits far below the ridge point of about 7.2 FLOP/byte, its 32 MB working set exceeds the 12 MB L3, and at 8 workers it already moves roughly 60 GB/s, so the kernel is memory-bandwidth bound and additional cores cannot help once the memory channels are full.

## Reproduce

```
py src/profile_env.py --student-id 230103075   # M0
py src/stride_bench.py                         # M1.1
py src/dense_kernel.py                         # M1.2
py src/false_sharing.py                        # M2
py src/scaling_study.py                        # M3
py self_check.py                               # numerical self-check of M0–M3
```

Note on Task 3.2: the weak-scaling formula as printed in the handout, P·Work(1)·T₁ / (Work(P)·T_P), reduces to T₁/T_P when Work(P) = P·Work(1), so its ideal value is 1. `scaling_metrics.tsv` therefore reports the standard Gustafson scaled speedup (Work(P)/Work(1))·(T₁/T_P), whose ideal is P, and lists the handout formula's values alongside it.
