/*
 * Lab 3: Work-Sharing & Loop Scheduling Policies (Mandelbrot)
 * Build/Run:  java Lab3Mandelbrot.java      (JDK 17+)
 * Outputs: console tables + lab3_results.csv
 *   Task 3.1  static and dynamic schedulers
 *   Task 3.2  4x4 sweep: P in {2,4,8,16} x chunk in {1,16,64,256}, 3 runs per cell
 *   Task 3.4  load imbalance = (max - min) / average rows-work per thread
 */
import java.io.*;
import java.util.concurrent.atomic.AtomicInteger;

public class Lab3Mandelbrot {
    static final int WIDTH = 1920, HEIGHT = 1080, MAX_ITER = 1000;
    static final int[][] IMAGE = new int[HEIGHT][WIDTH];

    static int computePixel(int px, int py) {
        double x0 = (px - WIDTH / 2.0) * 4.0 / WIDTH;
        double y0 = (py - HEIGHT / 2.0) * 4.0 / HEIGHT;
        double x = 0.0, y = 0.0; int iter = 0;
        while (x * x + y * y <= 4.0 && iter < MAX_ITER) {
            double tmp = x * x - y * y + x0;
            y = 2.0 * x * y + y0; x = tmp; iter++;
        }
        return iter;
    }

    /* static: contiguous block of rows per thread (N / P) */
    static double[] runStatic(int threads, long[] work) throws InterruptedException {
        Thread[] pool = new Thread[threads];
        int rows = (HEIGHT + threads - 1) / threads;
        long t0 = System.nanoTime();
        for (int t = 0; t < threads; t++) {
            final int tid = t, start = t * rows, end = Math.min(start + rows, HEIGHT);
            pool[t] = new Thread(() -> {
                long local = 0;
                for (int y = start; y < end; y++)
                    for (int x = 0; x < WIDTH; x++) local += IMAGE[y][x] = computePixel(x, y);
                work[tid] = local;                 // iterations actually executed
            });
            pool[t].start();
        }
        for (Thread t : pool) t.join();
        return new double[]{(System.nanoTime() - t0) / 1e6};
    }

    /* dynamic: shared atomic work queue, configurable chunk */
    static double[] runDynamic(int threads, int chunk, long[] work) throws InterruptedException {
        AtomicInteger queue = new AtomicInteger(0);
        Thread[] pool = new Thread[threads];
        long t0 = System.nanoTime();
        for (int t = 0; t < threads; t++) {
            final int tid = t;
            pool[t] = new Thread(() -> {
                long local = 0; int startRow;
                while ((startRow = queue.getAndAdd(chunk)) < HEIGHT) {
                    int endRow = Math.min(startRow + chunk, HEIGHT);
                    for (int y = startRow; y < endRow; y++)
                        for (int x = 0; x < WIDTH; x++) local += IMAGE[y][x] = computePixel(x, y);
                }
                work[tid] = local;
            });
            pool[t].start();
        }
        for (Thread t : pool) t.join();
        return new double[]{(System.nanoTime() - t0) / 1e6};
    }

    static double imbalance(long[] work, int threads) {
        long mx = Long.MIN_VALUE, mn = Long.MAX_VALUE; double avg = 0;
        for (int i = 0; i < threads; i++) { mx = Math.max(mx, work[i]); mn = Math.min(mn, work[i]); avg += work[i]; }
        avg /= threads;
        return (mx - mn) / avg;
    }

    public static void main(String[] args) throws Exception {
        PrintWriter csv = new PrintWriter(new FileWriter("lab3_results.csv"));
        csv.println("schedule,threads_P,chunk,run1_ms,run2_ms,run3_ms,mean_ms,imbalance");
        System.out.println("=== ENVIRONMENT ===");
        System.out.println("Java " + System.getProperty("java.version") + " | availableProcessors = "
                + Runtime.getRuntime().availableProcessors()
                + " | image " + WIDTH + "x" + HEIGHT + " | max_iter " + MAX_ITER);
        long[] w = new long[64];
        System.out.println("\n[warm-up JIT]"); runStatic(2, w);

        System.out.println("\n=== TASK 3.1 / 3.4: STATIC schedule (default N/P blocks) ===");
        System.out.printf("%-6s %-10s %-10s %-10s %-10s %-10s%n","P","run1","run2","run3","mean ms","imbalance");
        for (int p : new int[]{2,4,8,16}) {
            double[] r = new double[3]; double imb = 0;
            for (int i = 0; i < 3; i++) { r[i] = runStatic(p, w)[0]; imb = imbalance(w, p); }
            double mean = (r[0]+r[1]+r[2])/3.0;
            System.out.printf("%-6d %-10.1f %-10.1f %-10.1f %-10.1f %-10.3f%n",p,r[0],r[1],r[2],mean,imb);
            csv.printf("static,%d,N/P,%.1f,%.1f,%.1f,%.1f,%.3f%n",p,r[0],r[1],r[2],mean,imb);
        }

        System.out.println("\n=== TASK 3.2: DYNAMIC sweep 4x4 (P x chunk), 3 runs per cell ===");
        System.out.printf("%-6s %-8s %-10s %-10s %-10s %-10s %-10s%n","P","chunk","run1","run2","run3","mean ms","imbalance");
        for (int p : new int[]{2,4,8,16}) {
            for (int chunk : new int[]{1,16,64,256}) {
                double[] r = new double[3]; double imb = 0;
                for (int i = 0; i < 3; i++) { r[i] = runDynamic(p, chunk, w)[0]; imb = imbalance(w, p); }
                double mean = (r[0]+r[1]+r[2])/3.0;
                System.out.printf("%-6d %-8d %-10.1f %-10.1f %-10.1f %-10.1f %-10.3f%n",p,chunk,r[0],r[1],r[2],mean,imb);
                csv.printf("dynamic,%d,%d,%.1f,%.1f,%.1f,%.1f,%.3f%n",p,chunk,r[0],r[1],r[2],mean,imb);
            }
        }
        csv.close();
        System.out.println("\nWritten: lab3_results.csv");
    }
}
