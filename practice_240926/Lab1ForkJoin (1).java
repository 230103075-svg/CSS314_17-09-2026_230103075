/*
 * Lab 1: Fork-Join Model, Team Creation, Thread Scoping
 * Build/Run:  java Lab1ForkJoin.java        (JDK 17+)
 * Outputs: console tables, lab1_runs.txt (Task 1.1), lab1_results.csv (Tasks 1.2 / 1.3)
 */
import java.io.*;
import java.util.concurrent.*;
import java.util.stream.IntStream;

public class Lab1ForkJoin {

    /* ---------- Task 1.1: fork-join team, non-deterministic ordering ---------- */
    static String runTeam(int targetThreads) {
        StringBuilder sb = new StringBuilder();
        ForkJoinPool pool = new ForkJoinPool(targetThreads);
        try {
            pool.submit(() -> IntStream.range(0, targetThreads).parallel().forEach(idx -> {
                long osTid = Thread.currentThread().threadId();
                String name = Thread.currentThread().getName();
                String role = (idx == 0) ? "Master" : "Worker";
                synchronized (Lab1ForkJoin.class) {
                    sb.append(String.format("[%s] Logical Rank: %d | Thread: %s (OS ID: %d)%n",
                            role, idx, name, osTid));
                }
            })).join();                       // implicit barrier
        } finally { pool.shutdown(); }
        return sb.toString();
    }

    /* ---------- Task 1.2: oversubscription sweep (team create + join cost) ---------- */
    static double spawnJoinMs(int p) throws InterruptedException {
        long t0 = System.nanoTime();
        Thread[] team = new Thread[p];
        for (int i = 0; i < p; i++) team[i] = new Thread(() -> { });   // empty body
        for (Thread t : team) t.start();
        for (Thread t : team) t.join();                                 // barrier
        return (System.nanoTime() - t0) / 1e6;
    }

    /* ---------- Task 1.3: CPU saturation, 10,000,000 sqrt per thread ---------- */
    static double workloadMs(int p) throws InterruptedException {
        long t0 = System.nanoTime();
        Thread[] team = new Thread[p];
        for (int i = 0; i < p; i++) {
            team[i] = new Thread(() -> {
                double acc = 0;
                for (int k = 1; k <= 10_000_000; k++) acc += Math.sqrt(k);
                if (acc < 0) System.out.print("");   // keep JIT honest
            });
        }
        for (Thread t : team) t.start();
        for (Thread t : team) t.join();
        return (System.nanoTime() - t0) / 1e6;
    }

    public static void main(String[] args) throws Exception {
        PrintWriter csv = new PrintWriter(new FileWriter("lab1_results.csv"));
        csv.println("section,threads_P,metric,value");
        System.out.println("=== ENVIRONMENT ===");
        System.out.println("Java " + System.getProperty("java.version") + " | availableProcessors = "
                + Runtime.getRuntime().availableProcessors());

        /* Task 1.1 */
        System.out.println("\n=== TASK 1.1: 10 consecutive runs of a 4-thread team ===");
        PrintWriter runs = new PrintWriter(new FileWriter("lab1_runs.txt"));
        for (int r = 1; r <= 10; r++) {
            String out = runTeam(4);
            runs.println("--- Run " + r + " ---");
            runs.print(out);
            System.out.print("Run " + r + " rank order: ");
            for (String line : out.split("\n"))
                System.out.print(line.replaceAll(".*Logical Rank: (\\d+).*", "$1") + " ");
            System.out.println();
        }
        runs.close();

        /* Task 1.2 */
        System.out.println("\n=== TASK 1.2: Oversubscription sweep (spawn + join, no work) ===");
        System.out.printf("%-6s %-14s%n", "P", "Time (ms)");
        int[] ps = {1, 2, 4, 8, 16, 32, 64};
        for (int p : ps) {
            spawnJoinMs(p);                                   // warm-up
            double best = Double.MAX_VALUE;
            for (int r = 0; r < 5; r++) best = Math.min(best, spawnJoinMs(p));
            System.out.printf("%-6d %-14.3f%n", p, best);
            csv.printf("oversubscription,%d,spawn_join_ms,%.3f%n", p, best);
        }

        /* Task 1.3 */
        System.out.println("\n=== TASK 1.3: CPU saturation (10,000,000 sqrt per thread) ===");
        System.out.printf("%-6s %-14s %-14s%n", "P", "Time (ms)", "Time / P=1");
        workloadMs(1);                                        // warm-up JIT
        double base = 0;
        for (int p : new int[]{1, 2, 4, 8, 16, 32}) {
            double t = workloadMs(p);
            if (base == 0) base = t;
            System.out.printf("%-6d %-14.1f %-14.2f%n", p, t, t / base);
            csv.printf("saturation,%d,workload_ms,%.1f%n", p, t);
        }
        csv.close();
        System.out.println("\nWritten: lab1_results.csv, lab1_runs.txt");
        System.out.println("NOTE for Task 1.3: open your system monitor while this section runs "
                + "and record CPU utilisation per core.");
    }
}
