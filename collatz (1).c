/*
 * Zeba Academy - Parallel Computing Practicum: The Amdahl Reality Gap
 * Student: Kystaubay Ayanat (ID 230103075)
 * Workload: N = 10,000,000 + (3075 * 1,000) = 13,075,000
 *
 * Build:   gcc -O2 -fopenmp collatz.c -o collatz
 * Run:     ./collatz            (Windows: collatz.exe)
 * Output:  console tables + results.csv
 *
 * Phases covered:
 *   Phase 2 - sequential baseline (warm-up discarded, T_seq = (run2+run3)/2)
 *   Phase 3 - OpenMP scaling for k = 1,2,4,8,16 + Amdahl fitting
 *   Phase 4A - false sharing: naive hit_count[tid]++ vs padded/reduction
 *   Phase 4B - loop scheduling: static, static 1000, dynamic 100, dynamic 10000, guided
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#define N            13075000ULL      /* 10,000,000 + 3075 * 1,000 */
#define MOD          1000000007ULL
#define MAX_THREADS  256
#define RUNS         3                /* run 1 = cold warm-up, discarded */

static inline uint32_t collatz_steps(uint64_t n) {
    uint32_t steps = 0;
    while (n > 1) {
        if ((n & 1) == 0) n >>= 1;
        else n = 3 * n + 1;
        steps++;
    }
    return steps;
}

/* ---------------- Phase 2: sequential baseline ---------------- */
static double seq_run(uint32_t *max_steps, uint64_t *checksum) {
    uint32_t mx = 0; uint64_t sum = 0;
    double t0 = omp_get_wtime();
    for (uint64_t i = 1; i <= N; i++) {
        uint32_t s = collatz_steps(i);
        if (s > mx) mx = s;
        sum = (sum + s) % MOD;
    }
    double t = omp_get_wtime() - t0;
    *max_steps = mx; *checksum = sum;
    return t;
}

/* ---------------- Phase 3: parallel scaling ---------------- */
static double par_run(int k, uint32_t *max_steps, uint64_t *checksum) {
    uint32_t mx = 0; uint64_t sum = 0;
    double t0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 10000) num_threads(k) reduction(max:mx) reduction(+:sum)
    for (int64_t i = 1; i <= (int64_t)N; i++) {
        uint32_t s = collatz_steps((uint64_t)i);
        if (s > mx) mx = s;
        sum += s;
    }
    double t = omp_get_wtime() - t0;
    *max_steps = mx; *checksum = sum % MOD;
    return t;
}

/* ---------------- Phase 4A: false sharing ---------------- */
static int hit_count[MAX_THREADS];                 /* neighbouring ints -> same 64-byte line */
struct Padded { int count; char pad[60]; };        /* one counter per cache line */
static struct Padded padded[MAX_THREADS];

static double fs_naive(int k, long long *hits) {
    memset(hit_count, 0, sizeof(hit_count));
    double t0 = omp_get_wtime();
#pragma omp parallel num_threads(k)
    {
        int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (int64_t i = 1; i <= (int64_t)N; i++)
            if (collatz_steps((uint64_t)i) > 100) hit_count[tid]++;   /* FALSE SHARING */
    }
    double t = omp_get_wtime() - t0;
    long long total = 0;
    for (int i = 0; i < k; i++) total += hit_count[i];
    *hits = total;
    return t;
}

static double fs_padded(int k, long long *hits) {
    memset(padded, 0, sizeof(padded));
    double t0 = omp_get_wtime();
#pragma omp parallel num_threads(k)
    {
        int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (int64_t i = 1; i <= (int64_t)N; i++)
            if (collatz_steps((uint64_t)i) > 100) padded[tid].count++; /* own cache line */
    }
    double t = omp_get_wtime() - t0;
    long long total = 0;
    for (int i = 0; i < k; i++) total += padded[i].count;
    *hits = total;
    return t;
}

/* ---------------- Phase 4B: scheduling ---------------- */
static double sched_run(int k, int mode, long long *hits) {
    long long total = 0;
    double t0 = omp_get_wtime();
    switch (mode) {
    case 0:
#pragma omp parallel for schedule(static) num_threads(k) reduction(+:total)
        for (int64_t i = 1; i <= (int64_t)N; i++) if (collatz_steps((uint64_t)i) > 100) total++;
        break;
    case 1:
#pragma omp parallel for schedule(static, 1000) num_threads(k) reduction(+:total)
        for (int64_t i = 1; i <= (int64_t)N; i++) if (collatz_steps((uint64_t)i) > 100) total++;
        break;
    case 2:
#pragma omp parallel for schedule(dynamic, 100) num_threads(k) reduction(+:total)
        for (int64_t i = 1; i <= (int64_t)N; i++) if (collatz_steps((uint64_t)i) > 100) total++;
        break;
    case 3:
#pragma omp parallel for schedule(dynamic, 10000) num_threads(k) reduction(+:total)
        for (int64_t i = 1; i <= (int64_t)N; i++) if (collatz_steps((uint64_t)i) > 100) total++;
        break;
    default:
#pragma omp parallel for schedule(guided) num_threads(k) reduction(+:total)
        for (int64_t i = 1; i <= (int64_t)N; i++) if (collatz_steps((uint64_t)i) > 100) total++;
    }
    double t = omp_get_wtime() - t0;
    *hits = total;
    return t;
}

int main(void) {
    FILE *csv = fopen("results.csv", "w");
    uint32_t mx; uint64_t chk;
    int maxthr = omp_get_max_threads();

    printf("=== ENVIRONMENT ===\n");
    printf("N = %llu | omp_get_max_threads() = %d | OpenMP %d\n\n",
           (unsigned long long)N, maxthr, _OPENMP);
    fprintf(csv, "section,key,run1_cold,run2,run3,avg,extra\n");

    /* ---- Phase 2 ---- */
    printf("=== PHASE 2: SEQUENTIAL BASELINE (3 runs, run 1 discarded) ===\n");
    double sr[RUNS];
    for (int r = 0; r < RUNS; r++) {
        sr[r] = seq_run(&mx, &chk);
        printf("Run %d: %.4f s\n", r + 1, sr[r]);
    }
    double t_seq = (sr[1] + sr[2]) / 2.0;
    printf("T_seq = (run2 + run3)/2 = %.4f s | max steps = %u | checksum = %llu\n\n",
           t_seq, mx, (unsigned long long)chk);
    fprintf(csv, "baseline,T_seq,%.4f,%.4f,%.4f,%.4f,max_steps=%u checksum=%llu\n",
            sr[0], sr[1], sr[2], t_seq, mx, (unsigned long long)chk);

    /* ---- Phase 3 ---- */
    printf("=== PHASE 3: OPENMP SCALING ===\n");
    printf("%-6s %-10s %-10s %-10s %-10s %-9s %-9s %-8s\n",
           "k", "Run1(cold)", "Run2", "Run3", "Avg T_k", "S_emp", "S_theo", "Delta");
    int ks[] = {1, 2, 4, 8, 16};
    double avg[5]; double p = 0;
    for (int idx = 0; idx < 5; idx++) {
        int k = ks[idx];
        double r1 = par_run(k, &mx, &chk);
        double r2 = par_run(k, &mx, &chk);
        double r3 = par_run(k, &mx, &chk);
        avg[idx] = (r2 + r3) / 2.0;
        double s_emp = t_seq / avg[idx];
        if (k == 2) p = 2.0 * (1.0 - 1.0 / s_emp);          /* p from dual-core speedup */
        double s_theo = (idx == 0) ? 1.0 : 1.0 / ((1.0 - p) + p / k);
        printf("k=%-4d %-10.4f %-10.4f %-10.4f %-10.4f %-9.3f %-9.3f %-8.3f\n",
               k, r1, r2, r3, avg[idx], s_emp, s_theo, s_theo - s_emp);
        fprintf(csv, "scaling,k=%d,%.4f,%.4f,%.4f,%.4f,S_emp=%.3f S_theo=%.3f delta=%.3f checksum=%llu\n",
                k, r1, r2, r3, avg[idx], s_emp, s_theo, s_theo - s_emp, (unsigned long long)chk);
    }
    printf("Derived parallel fraction p = 2 * [1 - 1/S_emp(2)] = %.4f\n", p);
    printf("Amdahl ceiling S_max = 1/(1-p) = %.2fx\n\n", 1.0 / (1.0 - p));
    fprintf(csv, "amdahl,p,,,,%.4f,S_max=%.3f\n", p, 1.0 / (1.0 - p));

    /* ---- Phase 4A ---- */
    printf("=== PHASE 4A: FALSE SHARING (threads = %d) ===\n", maxthr);
    long long hits;
    double n1 = fs_naive(maxthr, &hits);          /* warm-up */
    double n2 = fs_naive(maxthr, &hits);
    double n3 = fs_naive(maxthr, &hits);
    double naive = (n2 + n3) / 2.0;
    double p1 = fs_padded(maxthr, &hits);
    double p2 = fs_padded(maxthr, &hits);
    double p3 = fs_padded(maxthr, &hits);
    double pad = (p2 + p3) / 2.0;
    printf("Variant 1 naive  hit_count[tid]++ : %.4f s | %.2f M iter/s\n", naive, N / naive / 1e6);
    printf("Variant 2 padded struct           : %.4f s | %.2f M iter/s\n", pad, N / pad / 1e6);
    printf("Penalty ratio (naive / padded)    : %.2fx | hits > 100 steps = %lld\n\n", naive / pad, hits);
    fprintf(csv, "false_sharing,naive,%.4f,%.4f,%.4f,%.4f,throughput=%.2fM\n", n1, n2, n3, naive, N / naive / 1e6);
    fprintf(csv, "false_sharing,padded,%.4f,%.4f,%.4f,%.4f,throughput=%.2fM penalty=%.2f hits=%lld\n",
            p1, p2, p3, pad, N / pad / 1e6, naive / pad, hits);

    /* ---- Phase 4B ---- */
    printf("=== PHASE 4B: LOOP SCHEDULING (threads = %d) ===\n", maxthr);
    const char *names[] = {"static (default)", "static, 1000", "dynamic, 100", "dynamic, 10000", "guided"};
    for (int m = 0; m < 5; m++) {
        double a = sched_run(maxthr, m, &hits);
        double b = sched_run(maxthr, m, &hits);
        double c = sched_run(maxthr, m, &hits);
        double av = (b + c) / 2.0;
        printf("%-18s : %.4f s\n", names[m], av);
        fprintf(csv, "schedule,%s,%.4f,%.4f,%.4f,%.4f,hits=%lld\n", names[m], a, b, c, av, hits);
    }
    fclose(csv);
    printf("\nresults.csv written.\n");
    return 0;
}
