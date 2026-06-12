// SPDX-License-Identifier: Apache-2.0
// CPU/EMC frequency-transition probe (RQ3 pilot).
//
// Repeatedly executes a fixed work chunk on a pinned core and timestamps every
// repetition with CLOCK_MONOTONIC (same clock as python time.monotonic_ns, so
// the orchestrator can align transition timestamps with probe samples).
//
// Two chunk types:
//   --chunk cpu   dependent integer multiply-adds — duration tracks core clock.
//   --chunk mem   sequential 64B-stride reads of a chunk inside a buffer much
//                 larger than all caches — duration tracks EMC bandwidth.
//
// Chunk sizes are deliberately small (cpu ~1.8us@1728MHz/~26us@115MHz, mem
// 128KB ~11us at single-core peak): the analyzer's stall dead zone is bounded
// by Q_slow - Q_fast, which must stay well under the 100us kill criterion
// (review finding: 1MB/5k-iter chunks had a 150-245us masking dead zone).
//
// Samples are buffered in a preallocated, pre-touched in-memory array and
// written out only after the measurement loop: an fprintf in the timed loop
// charged each 4MB stdio flush (~4-5ms, reproduced on this board) to a single
// chunk — 48x the kill line.
//
// The process locks memory and is expected to be started under an RT class
// (chrt -f 50) by the orchestrator: SCHED_OTHER runs showed 100-800us
// preemption noise that swamps the verdict statistic.
//
//   gcc -O2 -o freq_probe freq_probe.c
//   chrt -f 50 ./freq_probe --chunk mem --cpu 5 --ms 15000 --out probe.csv

#define _GNU_SOURCE
#include <getopt.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

// SIGTERM ends the loop early but still writes the CSV: the orchestrator
// budgets the probe for the worst case and terminates it once the toggle
// loop is done.
static volatile sig_atomic_t stop_flag = 0;
static void on_signal(int sig) { (void)sig; stop_flag = 1; }

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// 1500 dependent multiply-adds (latency-bound chain): ~5.4us at 1.7GHz,
// ~81us at 115MHz. Sized so a 300-transition run fits the sample buffer;
// the analyzer's quantum column discloses the resulting dead zone.
static uint64_t chunk_cpu(void) {
    volatile uint64_t x = 1;
    for (int i = 0; i < 1500; i++)
        x = x * 3 + 1;
    return x;
}

// One pass over `chunk` bytes at 64B stride, rotating through a big buffer so
// every rep misses cache. 128KB ≈ 11us at single-core ~12GB/s, ~50us at the
// lowest EMC points.
static uint64_t chunk_mem(const unsigned char *buf, size_t buf_sz,
                          size_t chunk, size_t *off) {
    volatile uint64_t sink = 0;
    const unsigned char *p = buf + *off;
    for (size_t i = 0; i < chunk; i += 64)
        sink += *(volatile uint64_t *)(p + i);
    *off += chunk;
    if (*off + chunk > buf_sz) *off = 0;
    return sink;
}

int main(int argc, char **argv) {
    int cpu = -1, ms = 10000;
    int mem_mode = 0;
    size_t buf_mb = 256, chunk_kb = 128;
    long max_samples = 24 * 1000 * 1000;   // 384MB of records, ample for ~40s
    const char *out_path = NULL;

    static struct option lo[] = {
        {"chunk", required_argument, 0, 'k'},
        {"cpu", required_argument, 0, 'c'},
        {"ms", required_argument, 0, 'd'},
        {"out", required_argument, 0, 'o'},
        {"buf-mb", required_argument, 0, 'b'},
        {"chunk-kb", required_argument, 0, 'z'},
        {"max-samples", required_argument, 0, 'n'},
        {0, 0, 0, 0},
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "", lo, NULL)) != -1) {
        switch (opt) {
        case 'k': mem_mode = strcmp(optarg, "mem") == 0; break;
        case 'c': cpu = atoi(optarg); break;
        case 'd': ms = atoi(optarg); break;
        case 'o': out_path = optarg; break;
        case 'b': buf_mb = (size_t)atoi(optarg); break;
        case 'z': chunk_kb = (size_t)atoi(optarg); break;
        case 'n': max_samples = atol(optarg); break;
        default: return 1;
        }
    }
    if (!out_path) { fprintf(stderr, "freq_probe: --out required\n"); return 1; }

    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0)
            fprintf(stderr, "freq_probe: warning: pin to cpu%d failed\n", cpu);
    }

    unsigned char *buf = NULL;
    size_t buf_sz = buf_mb << 20, chunk = chunk_kb << 10, off = 0;
    if (mem_mode) {
        if (posix_memalign((void **)&buf, 4096, buf_sz) != 0) {
            fprintf(stderr, "freq_probe: work buffer alloc failed\n");
            return 1;
        }
        memset(buf, 1, buf_sz);
    }

    struct sample { long long end_ns; int dur_ns; } *rec;
    rec = malloc((size_t)max_samples * sizeof(*rec));
    if (!rec) { fprintf(stderr, "freq_probe: sample buffer alloc failed\n"); return 1; }
    memset(rec, 0, (size_t)max_samples * sizeof(*rec));   // pre-touch: no faults in loop

    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
        fprintf(stderr, "freq_probe: warning: mlockall failed\n");
    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);

    long long deadline = now_ns() + (long long)ms * 1000000LL;
    long n = 0, dropped = 0;
    long long t0 = now_ns();
    while (t0 < deadline && !stop_flag) {
        if (mem_mode)
            chunk_mem(buf, buf_sz, chunk, &off);
        else
            chunk_cpu();
        long long t1 = now_ns();
        if (n < max_samples) {
            rec[n].end_ns = t1;
            rec[n].dur_ns = (int)(t1 - t0 > 2000000000LL ? 2000000000LL : t1 - t0);
            n++;
        } else {
            dropped++;
        }
        t0 = t1;
    }

    FILE *out = fopen(out_path, "w");
    if (!out) { perror("freq_probe: fopen"); return 1; }
    static char iobuf[1 << 22];
    setvbuf(out, iobuf, _IOFBF, sizeof(iobuf));
    fprintf(out, "end_ns,dur_ns\n");
    for (long i = 0; i < n; i++)
        fprintf(out, "%lld,%d\n", rec[i].end_ns, rec[i].dur_ns);
    fclose(out);
    if (dropped)
        fprintf(stderr, "freq_probe: WARNING: %ld samples dropped (max-samples)\n",
                dropped);
    free(rec);
    free(buf);
    return dropped ? 2 : 0;
}
