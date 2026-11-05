# Task 05 — Deconstructing "Amdahl's Pessimism": the 1988 nCUBE Experiment

## Setting

In 1988 John Gustafson, Gary Montry and Robert Benner at Sandia National Laboratories ran three scientific applications on a 1,024-processor nCUBE 10 hypercube (published in *SIAM J. Sci. Stat. Comput.*, 1988, and summarised in Gustafson's note "Reevaluating Amdahl's Law", *Communications of the ACM*, May 1988). The applications were:

1. beam stress analysis using a conjugate-gradient solver,
2. baffled surface-wave simulation using explicit finite differences,
3. unstable fluid flow using flux-corrected transport.

Scaled speedups of about **1,016–1,021×** on 1,024 processors were reported, i.e. close to linear — an outcome widely regarded at the time as impossible.

## Why it looked impossible

Profiling showed a serial fraction of a few tenths of a percent; take s = 0.4 %. Under Amdahl's Law on a fixed problem:

```
S_A(1024) = 1 / (0.004 + 0.996/1024) = 1 / (0.004 + 0.000973) ≈ 201
S_A(∞)    = 1 / 0.004 = 250
```

On this reading even an infinitely large machine stays below 250×, and 1,024 processors should give about 200×, not 1,000×.

## Reconstruction: what the experiment actually did

The team did not hold the problem fixed. They scaled each problem with the machine so that **each processor kept the same amount of work** and runtime stayed roughly constant: finer meshes and more grid points, more degrees of freedom. The serial parts — loading parameters, setup, I/O, the host-side control logic — did not grow in proportion.

On the parallel machine the serial share of the run was measured as α ≈ 0.4 %. Gustafson's formula then predicts

```
S_G(1024) = 1024 − 0.004 × 1023 = 1024 − 4.092 = 1019.9
```

which matches the reported 1,016–1,021× almost exactly.

## How problem expansion dismantled the ceiling

The 0.4 % is a fraction of the **parallel** run. Converted to Amdahl's uniprocessor frame (Task 03):

```
s_uni = α / S = 0.004 / 1019.9 ≈ 0.0000039   (≈ 0.0004 %)
Amdahl ceiling 1/s_uni ≈ 255,000
```

On a single processor the scaled problem would take about 1,020 units of time, of which the serial work is still only 0.004 units. The serial part did not grow with the problem, so its uniprocessor share collapsed by a factor of the speedup. Amdahl's ceiling was never broken — it was raised by three orders of magnitude, because the "0.4 % serial" quoted for the fixed problem is not the serial fraction of the scaled problem.

The team also reported fixed-size speedups for the same codes, which were much lower (in the hundreds), consistent with Amdahl's Law for the original problem size. Both results are correct; they measure different things.

## Lessons

1. Speedup claims must state whether the problem was fixed or scaled.
2. The serial fraction is not a property of a program alone but of a program at a problem size.
3. For capacity computing, the useful question is how much bigger a problem fits in the same time. The nCUBE experiment showed the answer can be close to linear when serial work is size-independent and communication is local (as in explicit finite differences, where each processor talks only to hypercube neighbours).
