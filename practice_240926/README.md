# Shared-Memory Concurrency & OpenMP Paradigms — Java Edition (Labs 1–3)

**Kystaubay Ayanat — Student ID 230103075 — 24.09.2026**
High-Performance & Parallel Computing | Instructor: Sufyan bin Uzayr

Full write-up: `OpenMP_Lab_Report_230103075.pdf` (Sections I–V, all tables, plots and answers to the analytical questions of Labs 1–3).

## Execution environment

| Item | Value |
|---|---|
| Host | Google Colab VM (KVM, Ubuntu Linux 6.6.122 x86-64) |
| CPU | Intel Xeon @ 2.20 GHz, family 6 / model 79 (Broadwell-EP) |
| Cores | 1 physical core, 2 threads per core → 2 logical CPUs |
| Cache | L1d 32 KiB, L1i 32 KiB, L2 256 KiB, L3 55 MiB, 64-byte line |
| Runtime | OpenJDK 21.0.12 (64-Bit Server VM) |
| Laptop | Huawei MateBook D 14, Windows 11 Home 25H2 (JDK toolchain could not be installed locally, so all benchmarks were run on the cloud VM; `hw_info.txt` matches the reported timings) |

Raw hardware dump: `hw_info.txt` (`lscpu`).

## Build & run

All programs are single-file Java 17+ sources and run directly with the source launcher, no compilation step required:

```bash
java lab1/Lab1ForkJoin.java      | tee lab1_log.txt   # ~1 min
java lab2/Lab2PiReduction.java   | tee lab2_log.txt   # ~3 min
java lab3/Lab3Mandelbrot.java    | tee lab3_log.txt   # ~10 min
```

Classic compile-then-run works as well:

```bash
javac Lab1ForkJoin.java && java Lab1ForkJoin
```

Each program writes its own `labN_results.csv` into the working directory.

## Contents

| Path | Description |
|---|---|
| `lab1/Lab1ForkJoin.java` | Fork-join team creation, 10 non-determinism runs, oversubscription sweep P = 1…64, CPU saturation workload |
| `lab2/Lab2PiReduction.java` | Pi integration: naive race, `synchronized` critical section, parallel reduction + speedup/efficiency |
| `lab3/Lab3Mandelbrot.java` | Mandelbrot 1920×1080: static vs. dynamic scheduling, 4×4 sweep (P × chunk), load-imbalance metric |
| `lab1_results.csv` … `lab3_results.csv` | Raw benchmark data behind every table and plot |
| `lab1_log.txt` … `lab3_log.txt` | Full console output of each run |
| `lab1_runs.txt` | Task 1.1: raw output of the 10 consecutive team runs |
| `lab1_plot.png` … `lab3_plot.png` | Scaling curves, speedup/efficiency, scheduling heatmap and static-vs-dynamic comparison |
| `hw_info.txt` | `lscpu` output of the execution host |

## Headline results

- **Lab 1** — team spawn+join cost grows 0.166 ms → 20.9 ms from P = 1 to P = 64; the compute workload improves only at P = 2 (66.3 ms vs 90.0 ms) and then degrades linearly with oversubscription.
- **Lab 2** — the unsynchronised accumulator returns π = 0.434 at P = 8 (error 2.71); `synchronized` costs +130 % (P = 2) and +232 % (P = 4) over a single thread; the reduction reaches S(P) = 1.94 with exact π.
- **Lab 3** — chunk C = 256 leaves only 4 chunks for the whole team and drives load imbalance to 9.656 at P = 16; small chunks (1–16) are fastest; static degrades at P = 8 (891.7 ms, imbalance 3.262).

All measurements are bounded by the host's single physical core, which is discussed in Sections IV and V of the report.
