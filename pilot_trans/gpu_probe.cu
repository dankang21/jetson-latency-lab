// SPDX-License-Identifier: Apache-2.0
// GPU frequency-transition probe (RQ3 pilot).
//
// Repeatedly launches a kernel that spins for a fixed number of GPU clock
// cycles, synchronizes, and timestamps each launch with CLOCK_MONOTONIC.
// Because the spin is cycle-counted, kernel duration is inversely
// proportional to the GPU clock: a frequency step shows up as a duration
// step, and any pipeline stall during the switch shows up as one long rep.
//
// 15k cycles ≈ 15us at 1020MHz / ≈ 49us at 306MHz, keeping the analyzer's
// masking dead zone (Q_slow - Q_fast ≈ 34us) well under the 100us kill
// criterion. Samples go to a preallocated in-memory array and are written
// after the loop (stdio flush inside the loop fakes multi-ms stalls).
// Every CUDA call is checked: a sticky error would otherwise degrade the
// loop into launch-overhead-only timings that look like valid data.
//
// Run under chrt -f 50 + taskset (orchestrator does both).
//
//   nvcc -O2 -arch=sm_87 -o gpu_probe gpu_probe.cu
//   ./gpu_probe --ms 15000 --cycles 15000 --out /tmp/gpu_probe.csv

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <cuda_runtime.h>

// SIGTERM ends the loop early but still writes the CSV (orchestrator budgets
// worst case, terminates when the toggle loop is done).
static volatile sig_atomic_t stop_flag = 0;
static void on_signal(int sig) { (void)sig; stop_flag = 1; }

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                       \
            fprintf(stderr, "gpu_probe: %s failed: %s\n", #call,           \
                    cudaGetErrorString(err__));                            \
            exit(2);                                                       \
        }                                                                  \
    } while (0)

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

__global__ void spin_kernel(long long cycles, int *sink) {
    long long start = clock64();
    while (clock64() - start < cycles) { }
    if (sink) *sink = (int)(clock64() & 0x7fffffff);
}

int main(int argc, char **argv) {
    int ms = 10000;
    long long cycles = 15000;
    long max_samples = 4 * 1000 * 1000;
    const char *out_path = NULL;

    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--ms")) ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cycles")) cycles = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--out")) out_path = argv[++i];
        else if (!strcmp(argv[i], "--max-samples")) max_samples = atol(argv[++i]);
    }
    if (!out_path) { fprintf(stderr, "gpu_probe: --out required\n"); return 1; }

    int *sink;
    CUDA_CHECK(cudaMalloc(&sink, sizeof(int)));

    // Warm the context + JIT before the measured loop.
    for (int i = 0; i < 50; i++) {
        spin_kernel<<<1, 1>>>(cycles, sink);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    struct sample { long long end_ns; int dur_ns; } *rec;
    rec = (struct sample *)malloc((size_t)max_samples * sizeof(*rec));
    if (!rec) { fprintf(stderr, "gpu_probe: sample buffer alloc failed\n"); return 1; }
    memset(rec, 0, (size_t)max_samples * sizeof(*rec));

    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);

    long long deadline = now_ns() + (long long)ms * 1000000LL;
    long n = 0, dropped = 0;
    long long t0 = now_ns();
    while (t0 < deadline && !stop_flag) {
        spin_kernel<<<1, 1>>>(cycles, sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
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
    if (!out) { perror("gpu_probe: fopen"); return 1; }
    static char iobuf[1 << 22];
    setvbuf(out, iobuf, _IOFBF, sizeof(iobuf));
    fprintf(out, "end_ns,dur_ns\n");
    for (long i = 0; i < n; i++)
        fprintf(out, "%lld,%d\n", rec[i].end_ns, rec[i].dur_ns);
    fclose(out);
    if (dropped)
        fprintf(stderr, "gpu_probe: WARNING: %ld samples dropped\n", dropped);
    free(rec);
    cudaFree(sink);
    return dropped ? 2 : 0;
}
