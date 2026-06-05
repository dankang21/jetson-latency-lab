# jetson-latency-lab

**Real-time inference latency on a Jetson Orin Nano Super, measured the way it actually deploys.**

A reproducible study of periodic ML inference latency on a `PREEMPT_RT` Jetson.
Most benchmarks run inference back to back; a robot runs it on a clock. Those are
not the same measurement — and the gap is large.

---

## TL;DR

**Part 1 (published):** the same MobileNetV2 that runs back to back in 3.88 ms
runs in **6.33 ms** when driven at a real 100 Hz period — a **62%** penalty. The
per-cycle decomposition shows the cause is GPU dynamic clocking (the GPU idles
down in the ~6 ms gap between cycles), **not** the scheduler: release jitter
stays under 45 µs throughout. Locking clocks with `jetson_clocks` recovers
deterministic ~3.9 ms. Full writeup:
[cleinsoft.com/dk](https://www.cleinsoft.com/dk/posts/back-to-back-benchmarks-lie).

The harness separates three quantities per cycle, which is what makes the cause
attributable rather than guessed:

| quantity | meaning | what it blames |
|---|---|---|
| `release_jitter` | wake-up − scheduled release | scheduler / IRQ latency |
| `compute` | inference done − wake-up | GPU + framework |
| `response` | inference done − scheduled release | end-to-end, vs deadline |

**Part 2 (in progress):** drive the same loop under contention (CPU, memory
bandwidth, cache, IO, timer/IRQ, thermal) to find where the deadline breaks and
which layer breaks it. The stress harness is in this repo; results pending.

## Test bed

| | |
|---|---|
| Board | Jetson Orin Nano Super 8 GB |
| Power | `MAXN_SUPER` (default governor and `jetson_clocks` both measured) |
| Kernel | `PREEMPT_RT` |
| Runtime | ONNX Runtime 1.23.0, CUDA EP (PyTorch 2.9.1 / CUDA 12.6 / TensorRT 10.3.0 stack) |
| Model | MobileNetV2-12, ONNX, input `1×3×224×224` |
| Loop | 100 Hz (10 ms period), 100k cycles, `SCHED_FIFO`, pinned to a dedicated core, `mlockall` |

Headline numbers are in [Results](#results). The 3.882 ms back-to-back figure is
the tight-loop baseline (`bench.py`); the periodic figures are this harness.

## Method

The harness ([`harness/infer_bench.py`](harness/infer_bench.py)) runs a
fixed-rate loop using absolute `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)`
so scheduling error does not accumulate across cycles. The loop runs under
`SCHED_FIFO` pinned to a dedicated core with locked memory, so the *measurement*
is not itself a jitter source. Stressors run unpinned on the remaining cores and
contend for shared resources (cache, memory bandwidth, IRQs, the scheduler).

Stress axes are defined in
[`experiments/profiles.yaml`](experiments/profiles.yaml) and driven by
[`experiments/run_matrix.sh`](experiments/run_matrix.sh).

### Reproduce

```bash
# 0. deps
sudo apt-get install stress-ng
pip install -r requirements.txt          # numpy / matplotlib / pyyaml

# 1. isolate a core at boot (kernel cmdline), then confirm:
#    isolcpus=5 nohz_full=5 rcu_nocbs=5
cat /sys/devices/system/cpu/isolated

# 2. point the harness at your model and your inference path
#    (see harness/backends.py — wire YourBaselineBackend to your baseline code
#    so stress numbers are apples-to-apples with the baseline above)

# 3. run the full matrix (untuned), then again tuned
sudo CPU=5 MODEL=models/mobilenetv2-12.onnx ./experiments/run_matrix.sh
sudo CPU=5 MITIGATIONS=1 ./experiments/run_matrix.sh

# 4. tables + plots
python3 analysis/analyze.py     # markdown table + results/summary.csv
python3 analysis/plot.py        # tail / jitter-vs-compute / CDF PNGs
```

> Note: the baseline stack needs the documented `libcudss` `LD_LIBRARY_PATH`
> workaround in the venv activation; the harness inherits whatever env you
> launch it from.

## Results

100 Hz periodic loop, 100,000 cycles, MobileNetV2 on the CUDA EP.

| condition | resp p50 | resp p99.99 | resp max | compute p50 | jitter p99.99 | misses |
|---|---|---|---|---|---|---|
| back-to-back (tight loop) | 3.882 | 3.978 | 4.205 | — | — | 0 |
| 100 Hz periodic, default governor | 6.333 | 6.680 | 6.726 | 6306.6 | 45.0 | 0 |
| 100 Hz periodic, `jetson_clocks` | 3.905 | 3.982 | 4.017 | 3893.6 | 22.9 | 0 |

(latencies in ms except compute/jitter in µs)

Driving the same model at its real 100 Hz period costs **62%** over the
back-to-back number — and it is GPU dynamic clocking, not the scheduler
(jitter stays under 45 µs). Locking clocks with `jetson_clocks` fully recovers
deterministic ~3.9 ms.

![the idle tax](docs/chart_idle_tax.png)
![what the latency is made of](docs/chart_whodunit.png)

Full writeup: [cleinsoft.com/dk](https://www.cleinsoft.com/dk/posts/back-to-back-benchmarks-lie)

> Stress-matrix results (CPU / memory / cache / IO / IRQ / thermal) and the
> tail-distribution plots are part 2 — run `experiments/run_matrix.sh`, then
> `analysis/analyze.py` and `analysis/plot.py`.

## Layout

```
harness/
  rt_utils.py      RT primitives: abs clock_nanosleep, SCHED_FIFO, affinity, mlockall
  backends.py      pluggable inference (onnxruntime CUDA reference + baseline slot)
  infer_bench.py   periodic loop, jitter/compute/response accounting, JSON+CSV out
experiments/
  profiles.yaml    stress matrix definition
  run_matrix.sh    orchestration (stress-ng + harness, untuned/tuned)
analysis/
  analyze.py       aggregate -> markdown table + summary.csv
  plot.py          tail-by-profile, jitter-vs-compute, tail CDF
```

## Caveats

Single board, single model, single kernel build — this measures *this* system,
not Jetson RT in general. Numbers depend on power mode, kernel config, isolation
setup, and stress intensity, all of which are recorded in each result's
`meta` block so a run is self-describing.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The
MobileNetV2 ONNX model is not redistributed here; fetch it from the
[ONNX Model Zoo](https://github.com/onnx/models).

---

*Built by Daniel Kang (dk). Background: ARM Linux mach #184, iriver Clix kernel +
graphics stack, now real-time ML for robots @ Cleinsoft. Writeup:
[cleinsoft.com/dk](https://cleinsoft.com/dk).*
