/*
 * Lab 2: Numerical Integration (Pi) & Parallel Reductions
 * Build/Run:  java Lab2PiReduction.java       (JDK 17+)
 * Outputs: console tables + lab2_results.csv
 *   Task 2.1 race condition error table (P = 1,2,4,8)
 *   Task 2.2 critical-section overhead (N = 1,000,000)
 *   Task 2.3 strong scaling of the reduction (P = 1..16, 5 trials)
 *   Task 2.4 speedup S(P) and efficiency E(P)
 */
import java.io.*;
import java.util.concurrent.*;
import java.util.stream.LongStream;

public class Lab2PiReduction {
    static final long N = 100_000_000L;
    static final double STEP = 1.0 / N;

    /* Variant A: unsynchronized shared accumulator (data race) */
    static double runNaiveRace(int threads) throws InterruptedException {
        double[] shared = new double[1];
        Thread[] pool = new Thread[threads];
        long chunk = N / threads;
        for (int t = 0; t < threads; t++) {
            final long start = t * chunk, end = (t == threads - 1) ? N : start + chunk;
            pool[t] = new Thread(() -> {
                for (long i = start; i < end; i++) {
                    double x = (i + 0.5) * STEP;
                    shared[0] += 4.0 / (1.0 + x * x);      // unprotected write
                }
            });
            pool[t].start();
        }
        for (Thread t : pool) t.join();
        return shared[0] * STEP;
    }

    /* Variant B: synchronized critical section, one lock per step */
    static double[] runCriticalSection(int threads, long steps) throws InterruptedException {
        final Object lock = new Object();
        final double step = 1.0 / steps;
        double[] shared = new double[1];
        Thread[] pool = new Thread[threads];
        long chunk = steps / threads;
        long t0 = System.nanoTime();
        for (int t = 0; t < threads; t++) {
            final long start = t * chunk, end = (t == threads - 1) ? steps : start + chunk;
            pool[t] = new Thread(() -> {
                for (long i = start; i < end; i++) {
                    double x = (i + 0.5) * step;
                    double term = 4.0 / (1.0 + x * x);
                    synchronized (lock) { shared[0] += term; }   // #pragma omp critical
                }
            });
            pool[t].start();
        }
        for (Thread t : pool) t.join();
        return new double[]{shared[0] * step, (System.nanoTime() - t0) / 1e6};
    }

    /* Serial baseline over an arbitrary step count */
    static double[] runSerial(long steps) {
        double step = 1.0 / steps, sum = 0.0;
        long t0 = System.nanoTime();
        for (long i = 0; i < steps; i++) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
        return new double[]{sum * step, (System.nanoTime() - t0) / 1e6};
    }

    /* Variant C: parallel reduction (private accumulators merged in a tree) */
    static double[] runParallelReduction(int parallelism) throws Exception {
        ForkJoinPool pool = new ForkJoinPool(parallelism);
        long t0 = System.nanoTime();
        double sum = pool.submit(() ->
                LongStream.range(0, N).parallel()
                        .mapToDouble(i -> { double x = (i + 0.5) * STEP; return 4.0 / (1.0 + x * x); })
                        .sum()).get();
        double ms = (System.nanoTime() - t0) / 1e6;
        pool.shutdown();
        return new double[]{sum * STEP, ms};
    }

    public static void main(String[] args) throws Exception {
        PrintWriter csv = new PrintWriter(new FileWriter("lab2_results.csv"));
        csv.println("task,variant,threads_P,pi,abs_error,time_ms,extra");
        System.out.println("=== ENVIRONMENT ===");
        System.out.println("Java " + System.getProperty("java.version")
                + " | availableProcessors = " + Runtime.getRuntime().availableProcessors()
                + " | N = " + N);

        System.out.println("\n[warm-up JIT]");
        runSerial(5_000_000); runParallelReduction(2);

        /* ---- Task 2.1 ---- */
        System.out.println("\n=== TASK 2.1: Race condition quantification (Variant A) ===");
        System.out.printf("%-6s %-20s %-14s%n", "P", "Pi (computed)", "Abs error");
        for (int p : new int[]{1, 2, 4, 8}) {
            double pi = runNaiveRace(p);
            double err = Math.abs(pi - Math.PI);
            System.out.printf("%-6d %-20.12f %-14.3e%n", p, pi, err);
            csv.printf("2.1,naive_race,%d,%.12f,%.3e,,%n", p, pi, err);
        }

        /* ---- Task 2.2 ---- */
        System.out.println("\n=== TASK 2.2: Critical-section overhead (N = 1,000,000) ===");
        double[] ser = runSerial(1_000_000);
        System.out.printf("Serial baseline      : pi = %.12f | %.2f ms%n", ser[0], ser[1]);
        for (int p : new int[]{2, 4}) {
            double[] cs = runCriticalSection(p, 1_000_000);
            double overhead = (cs[1] - ser[1]) / ser[1] * 100.0;
            System.out.printf("synchronized, P = %-2d : pi = %.12f | %.2f ms | overhead %+.1f%%%n",
                    p, cs[0], cs[1], overhead);
            csv.printf("2.2,critical,%d,%.12f,%.3e,%.2f,overhead_pct=%.1f%n",
                    p, cs[0], Math.abs(cs[0] - Math.PI), cs[1], overhead);
        }
        csv.printf("2.2,serial,1,%.12f,%.3e,%.2f,baseline%n", ser[0], Math.abs(ser[0] - Math.PI), ser[1]);

        /* ---- Task 2.3 + 2.4 ---- */
        System.out.println("\n=== TASK 2.3 / 2.4: Reduction strong scaling (5 trials, N = " + N + ") ===");
        System.out.printf("%-6s %-12s %-12s %-10s %-10s %-16s%n",
                "P", "Avg T(P) ms", "Best ms", "S(P)", "E(P)", "Pi");
        double t1 = 0;
        for (int p : new int[]{1, 2, 4, 8, 16}) {
            double sum = 0, best = Double.MAX_VALUE, pi = 0;
            for (int trial = 0; trial < 5; trial++) {
                double[] r = runParallelReduction(p);
                sum += r[1]; best = Math.min(best, r[1]); pi = r[0];
            }
            double avg = sum / 5.0;
            if (t1 == 0) t1 = avg;
            double s = t1 / avg, e = s / p * 100.0;
            System.out.printf("%-6d %-12.1f %-12.1f %-10.2f %-9.1f%% %-16.12f%n", p, avg, best, s, e, pi);
            csv.printf("2.3,reduction,%d,%.12f,%.3e,%.1f,S=%.2f E=%.1f%% best=%.1f%n",
                    p, pi, Math.abs(pi - Math.PI), avg, s, e, best);
        }
        csv.close();
        System.out.println("\nWritten: lab2_results.csv");
    }
}
