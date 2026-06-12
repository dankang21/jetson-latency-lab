// SPDX-License-Identifier: Apache-2.0
// IsolBench-Bandwidth-style memory-bandwidth adversary for contention runs.
//
// Part 3 used stress-ng (--vm/--cache), which produced ~0.3% slowdown on the
// victim: its workers are page-fault/cache-heavy, not DRAM-bandwidth-heavy.
// This tool does what IsolBench's Bandwidth does instead: each thread streams
// sequentially over a buffer far larger than all caches (L1 64K + L2 256K per
// core, 2x2MB L3), so every access goes to DRAM and competes with the victim
// for EMC bandwidth.
//
// Self-reports achieved GB/s once per second on stderr and a summary line on
// stdout at exit — the adversary's own throughput is the evidence of how much
// pressure was actually applied in each experiment cell.
//
//   gcc -O2 -pthread -o membw membw.c
//   ./membw -t 4 -c 0,1,2,3 -m 256 -M write -d 60

#define _GNU_SOURCE
#include <getopt.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t stop_flag = 0;
static void on_signal(int sig) { (void)sig; stop_flag = 1; }

enum mode { MODE_WRITE, MODE_READ };

struct worker {
    pthread_t tid;
    int cpu;                 // -1 = no pinning
    size_t buf_bytes;
    enum mode mode;
    _Atomic unsigned long long bytes_done;
};

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void *worker_main(void *arg) {
    struct worker *w = arg;

    if (w->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(w->cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0)
            fprintf(stderr, "membw: warning: pin to cpu%d failed\n", w->cpu);
    }

    unsigned char *buf;
    if (posix_memalign((void **)&buf, 4096, w->buf_bytes) != 0) {
        fprintf(stderr, "membw: alloc %zu bytes failed\n", w->buf_bytes);
        return NULL;
    }
    memset(buf, 1, w->buf_bytes);   // fault pages in before measuring

    if (w->mode == MODE_WRITE) {
        unsigned char v = 2;        // nonzero: avoid the DC ZVA zero-fill fast path
        while (!stop_flag) {
            memset(buf, v, w->buf_bytes);
            if (++v == 0) v = 1;    // skip 0 on wraparound too
            atomic_fetch_add_explicit(&w->bytes_done, w->buf_bytes,
                                      memory_order_relaxed);
        }
    } else {
        // 64B stride = one load per cache line; volatile sink defeats DCE.
        volatile uint64_t sink = 0;
        while (!stop_flag) {
            for (size_t i = 0; i < w->buf_bytes; i += 64)
                sink += *(volatile uint64_t *)(buf + i);
            atomic_fetch_add_explicit(&w->bytes_done, w->buf_bytes,
                                      memory_order_relaxed);
        }
        (void)sink;
    }
    free(buf);
    return NULL;
}

int main(int argc, char **argv) {
    int nthreads = 4, duration_s = 0, mb = 256;
    enum mode mode = MODE_WRITE;
    int cpus[64], ncpus = 0;

    int opt;
    while ((opt = getopt(argc, argv, "t:m:M:c:d:h")) != -1) {
        switch (opt) {
        case 't': nthreads = atoi(optarg); break;
        case 'm': mb = atoi(optarg); break;
        case 'M': mode = strcmp(optarg, "read") == 0 ? MODE_READ : MODE_WRITE; break;
        case 'c': {
            char *tok = strtok(optarg, ",");
            while (tok && ncpus < 64) { cpus[ncpus++] = atoi(tok); tok = strtok(NULL, ","); }
            break;
        }
        case 'd': duration_s = atoi(optarg); break;
        default:
            fprintf(stderr,
                "usage: %s [-t threads] [-m MB/thread] [-M write|read] "
                "[-c cpu,cpu,...] [-d seconds]\n", argv[0]);
            return 1;
        }
    }
    if (nthreads < 1 || nthreads > 64) { fprintf(stderr, "membw: bad -t\n"); return 1; }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    struct worker *ws = calloc(nthreads, sizeof(*ws));
    for (int i = 0; i < nthreads; i++) {
        ws[i].cpu = ncpus ? cpus[i % ncpus] : -1;
        ws[i].buf_bytes = (size_t)mb << 20;
        ws[i].mode = mode;
        pthread_create(&ws[i].tid, NULL, worker_main, &ws[i]);
    }

    long long t_start = now_ns(), t_last = t_start;
    unsigned long long last_total = 0;
    while (!stop_flag) {
        sleep(1);
        unsigned long long total = 0;
        for (int i = 0; i < nthreads; i++)
            total += atomic_load_explicit(&ws[i].bytes_done, memory_order_relaxed);
        long long t = now_ns();
        double gbps = (double)(total - last_total) / (double)(t - t_last);
        fprintf(stderr, "membw: %.2f GB/s (%d thr, %s)\n",
                gbps, nthreads, mode == MODE_WRITE ? "write" : "read");
        last_total = total; t_last = t;
        if (duration_s && (t - t_start) / 1000000000LL >= duration_s)
            stop_flag = 1;
    }

    for (int i = 0; i < nthreads; i++)
        pthread_join(ws[i].tid, NULL);

    unsigned long long total = 0;
    for (int i = 0; i < nthreads; i++) total += ws[i].bytes_done;
    double secs = (double)(now_ns() - t_start) / 1e9;
    printf("{\"threads\":%d,\"mode\":\"%s\",\"mb_per_thread\":%d,"
           "\"seconds\":%.2f,\"total_gb\":%.2f,\"avg_gbps\":%.2f}\n",
           nthreads, mode == MODE_WRITE ? "write" : "read", mb,
           secs, (double)total / 1e9, (double)total / 1e9 / secs);
    free(ws);
    return 0;
}
