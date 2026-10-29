# RESULTS — Karp–Flatt Practice Lab

**Student ID:** 230103075  ·  **Branch:** `practice-230103075`

## Hardware profile

| Item | Value |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i5-12450H |
| Physical cores | 8 |
| Logical cores (`os.cpu_count()`) | 12 |
| OS | Windows-11-10.0.26200-SP0 |
| Python | CPython 3.14.0 |

Every value below is the median of 3 runs.

## Results

| Experiment Mode | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | Architectural Diagnosis |
|---|---|---|---|---|---|---|
| A: Serial Tail | 7.560 | 3.437 | 2.568 | -0.091 | 0.120 | Serial tail; e(2) distorted by a slow 1-process baseline |
| B: Lock Contention | 5.325 | 20.075 | 20.586 | 6.539 | 4.821 | Huge e (saturated at P=2) → mutex & bus thrashing |
| C: Privatized State | 0.901 | 0.522 | 0.607 | 0.158 | 0.565 | e inflated by process start-up (see warm-pool row) |
| C: Privatized (warm pool) | 0.701 | 0.290 | 0.176 | -0.172 | 0.002 | Collapsed e → linear scalability restored |

Speedup: A 2.94x · B 0.26x · C 1.48x (handout) / 3.98x (warm pool) at P = 4.

## Rubric checks

| Check | Measured | Result |
|---|---|---|
| A: e(4)/e(2) within [0.80, 1.25] | -1.319 | NOT MET |
| B: T4 > T1 | 20.586 s vs 5.325 s | PASS |
| B: e(4) ≫ e(2) | 4.821 vs 6.539 | NOT MET |
| C: T4 < T1 and E4 > 70% (handout timing) | E4 = 37.1% | NOT MET |
| C: T4 < T1 and E4 > 70% (warm pool) | E4 = 99.5% | PASS |

## Diagnosis

Experiment A did not give a flat e on this machine: e(2) = -0.091 and e(4) = 0.120. A negative e(2) means P=2 ran faster than linear (2.20x on 2 workers), which points to an inflated single-process baseline T1 = 7.560 s rather than to the code: this CPU mixes performance and efficiency cores, and a lone worker that Windows schedules on an efficiency core, or at a lower turbo clock, runs noticeably slower. e(4) = 0.120 is where the handout expects a fixed serial tail (about 0.10–0.15), consistent with the 0.40 s sleep plus pool start-up.

Experiment B is fully parallel in principle, yet adding processes made it far slower (5.325 s on 1 process vs 20.075 s on 2 and 20.586 s on 4). Both Karp–Flatt values are huge (e(2) = 6.539, e(4) = 4.821), an order of magnitude above Experiment A, which a serial code path cannot produce: the mutex and the cache line holding the counter bounce between cores on every increment. e(4) came out lower than e(2) because the contention is already saturated at two processes: T2 and T4 are almost equal (20.075 s vs 20.586 s), meaning the lock fully serializes the work from P=2 onward. With T flat, the Karp–Flatt formula (p·Tp/T1 − 1)/(p − 1) yields a smaller value at larger p even though the pathology is the same.

Experiment C removes all shared state from the hot loop and synchronizes once at the end. Measured exactly as in the handout, the timer also covers creating the process pool, and on Windows each new process is spawned from scratch: that fixed cost dominates a run this short, giving e(4) = 0.565 and E4 = 37.1%. With the pool started before the timer, the same work gives e(4) = 0.002, a 3.98x speedup and E4 = 99.5%: the privatized computation itself scales almost linearly, and the gap between the two rows is process start-up overhead, not contention.
