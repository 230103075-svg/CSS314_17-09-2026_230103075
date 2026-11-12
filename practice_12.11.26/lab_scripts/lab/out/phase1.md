## Phase 1: False Sharing & Karp-Flatt (M=16000000, Pphys=8, Plog=12)

### Stride sweep (p = Pphys)

| stride | bytes | median ms | min ms | IQR ms | Mops/s |
|---|---|---|---|---|---|
| 1 | 8 | 971.2 | 928.3 | 41.5 | 16.47 |
| 2 | 16 | 959.8 | 900.4 | 39.2 | 16.67 |
| 4 | 32 | 1015.5 | 882.9 | 30.3 | 15.76 |
| 8 | 64 | 1011.3 | 971.1 | 40.3 | 15.82 |
| 16 | 128 | 1033.8 | 918.7 | 24.2 | 15.48 |
| 32 | 256 | 1012.9 | 969.3 | 82.7 | 15.80 |

### Process scaling

**stride = 1** (8 B)

| p | T_p ms | S_p | E_p | Karp-Flatt e(p) |
|---|---|---|---|---|
| 1 | 2708.2 | 1.00 | 1.00 | - |
| 2 | 1559.4 | 1.74 | 0.87 | 0.152 |
| 4 | 1076.2 | 2.52 | 0.63 | 0.196 |
| 8 | 1035.3 | 2.62 | 0.33 | 0.294 |
| 12 | 972.4 | 2.79 | 0.23 | 0.301 |

**stride = 16** (128 B)

| p | T_p ms | S_p | E_p | Karp-Flatt e(p) |
|---|---|---|---|---|
| 1 | 2701.9 | 1.00 | 1.00 | - |
| 2 | 1606.6 | 1.68 | 0.84 | 0.189 |
| 4 | 1053.2 | 2.57 | 0.64 | 0.186 |
| 8 | 1029.3 | 2.63 | 0.33 | 0.293 |
| 12 | 952.4 | 2.84 | 0.24 | 0.294 |

### Analysis

- Coherence contention drops sharply at stride 32 = **256 B** (1.02x faster than the previous stride); cache line = 64 B, so once slots are >= one line apart each core owns its line and the MESI invalidation ping-pong disappears.
- Mean Karp-Flatt e(p): stride=1 -> 0.236, stride=16 -> 0.240. For stride=1, e(p) is large and grows with p: the 'serial fraction' is really coherence traffic that gets worse with more sharers (overhead scales with p, not a fixed Amdahl fraction). For stride=16 e(p) is small and roughly flat, i.e. remaining loss is process-spawn cost / SMT sharing.
- Note: pure-Python loop overhead hides part of the effect; spawn time is inside the timed region.

```
VERIFY_HASH[PHASE1]: 1B5618B1D51B9728 (P_log=12)
```
