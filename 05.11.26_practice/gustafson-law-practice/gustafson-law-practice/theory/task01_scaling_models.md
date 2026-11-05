# Task 01 — Strong Scaling vs. Weak Scaling Foundations

## 1. The two questions

Amdahl (1967) and Gustafson (1988) answer different questions about the same machine.

| | Amdahl (1967) | Gustafson (1988) |
|---|---|---|
| Question asked | *How much faster can P processors solve **this** problem?* | *How much **more** problem can P processors solve in the same time?* |
| What is held fixed | Total problem size W | Wall-clock time T |
| What grows with P | Nothing — the work is divided | The problem size, W(P) ≈ P·W(1) |
| Where s is measured | On one processor | On the P-processor machine |
| Speedup | S = 1 / (s + (1 − s)/P) | S = s + (1 − s)·P = P − α(P − 1) |
| Limit P → ∞ | S → 1/s (a hard ceiling) | S → ∞ (grows linearly) |
| Scaling regime | Strong scaling | Weak scaling |

## 2. Problem dimension as P → ∞

**Amdahl.** The problem dimension N is constant. As P → ∞ each processor's share N/P → 0. The parallel part's time (1 − s)/P → 0, but the serial part s stays, so runtime approaches s and speedup approaches 1/s. With s = 5 %, no machine of any size exceeds 20×.

**Gustafson.** The problem dimension grows with the machine so that the parallel work per processor stays constant. If work is linear in N, then N(P) = P·N(1) and N → ∞ with P. The runtime on the parallel machine stays at s + p = 1 by construction; it is the *equivalent serial time*, s + p·P, that grows without bound.

## 3. Why "fixed-size" vs "fixed-time"

* **Amdahl is fixed-size** because the benchmark is defined by the problem: a fixed W is executed on 1 processor and on P processors and the times are compared. Adding processors can only shrink time; it cannot shrink the serial part.
* **Gustafson is fixed-time** because the benchmark is defined by the deadline: users run the largest problem that finishes in an acceptable time (overnight, one time step per second). Adding processors grows the problem, not the speed. The serial part — I/O, setup, a reduction — usually grows slowly or not at all, so its *share* of the run shrinks as the problem expands.

## 4. Philosophical and physical divergence

1. **Model of user behaviour.** Amdahl assumes the user has a fixed problem and wants it sooner. Gustafson observed that scientists keep wall-clock time roughly constant and refine resolution: a finer mesh, more particles, more time steps. Both behaviours are real; they apply to different workloads (a fixed database query vs a climate model).
2. **Frame of reference for s.** Amdahl's s is a property of the *uniprocessor* run. Gustafson's α is a property of the *parallel* run. These are not the same number for the same program (Task 03), and confusing them produces the impression that the laws contradict each other.
3. **Physical scaling of the serial part.** Amdahl implicitly assumes the serial part scales *with* the problem (constant fraction). Gustafson assumes it is roughly *constant in absolute time* while the parallel part grows. For most simulations the serial work (reading parameters, global setup) grows far more slowly than the volume of computation, which supports Gustafson's view.
4. **Neither is complete.** Amdahl ignores that problems grow; Gustafson ignores communication, memory and synchronization costs that grow with P (Tasks 12, 13). Sun–Ni's memory-bounded model (Task 11) contains both as special cases.

## 5. Summary

Amdahl's Law is not pessimistic and Gustafson's is not optimistic: they hold different quantities fixed. Amdahl's bound is the right model for latency-critical, fixed-size work; Gustafson's is the right model for capacity computing, where a bigger machine is bought to run a bigger problem in the same time.
