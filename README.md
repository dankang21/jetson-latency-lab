# jetson-latency-lab

**When does 100 Hz inference actually start missing its deadline on a Jetson Orin Nano Super?**

A reproducible study of real-time inference latency under load on a
`PREEMPT_RT` Jetson. The unstressed case is boringly clean. The interesting
question is the one a deployed robot actually faces: *what happens when the rest
of the system is busy?*

---

## TL;DR

On an idle, tuned board, a 100 Hz (10 ms period) MobileNetV2 loop is a
cleanroom — sub-4 ms response, zero deadline misses over 100k cycles. That
number tells you almost nothing about production. This repo stresses the board
along independent axes (CPU, memory bandwidth, cache, IO, timer/IRQ, thermal)
and measures **where the deadline starts to break and which layer breaks it**.

The harness separates three quantities per cycle so the failure mode is
attributable, not just visible:

| quantity | meaning | what it blames |
|---|---|---|
| `release_jitter` | wake-up − scheduled release | scheduler / IRQ latency |
| `compute` | inference done − wake-up | GPU + framework |
| `response` | inference done − scheduled release | end-to-end, vs deadline |

The recurring finding in RT systems — and the thing this repo is built to show
cleanly — is that tail misses come from **late wake-ups, not slow compute**.

## Test bed

| | |
|---|---|
| Board | Jetson Orin Nano Super |
| Power | `MAXN_SUPER`, clocks locked (`jetson_clocks`) |
| Kernel | `PREEMPT_RT` |
| Runtime | PyTorch 2.9.1 + CUDA 12.6 (baseline stack) |
| Model | MobileNetV2, ONNX, input `1×3×224×224` |
| Loop | SCHED_FIFO, pinned to an isolated core, `mlockall` |

Baseline (100k cycles, idle): mean **3.882 ms**, std **0.011 ms**,
p99.99 **3.978 ms**, **0** deadline misses.

## Method

The harness ([`harness/infer_bench.py`](harness/infer_bench.py)) runs a
fixed-rate loop using absolute `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)`
so scheduling error does not accumulate across cycles. The loop runs under
`SCHED_FIFO` pinned to an isolated core with locked memory, so the *measurement*
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
sudo CPU=5 MODEL=models/mobilenetv2.onnx ./experiments/run_matrix.sh
sudo CPU=5 MITIGATIONS=1 ./experiments/run_matrix.sh

# 4. tables + plots
python3 analysis/analyze.py     # markdown table + results/summary.csv
python3 analysis/plot.py        # tail / jitter-vs-compute / CDF PNGs
```

> Note: the baseline stack needs the documented `libcudss` `LD_LIBRARY_PATH`
> workaround in the venv activation; the harness inherits whatever env you
> launch it from.

## Results

> Filled from your run. `analysis/analyze.py` prints this table ready to paste.

| profile | resp p50 | resp p99 | resp p99.99 | resp max | compute p99.99 | jitter p99.99 | misses | miss % |
|---|---|---|---|---|---|---|---|---|
| baseline | … | … | … | … | … | … | … | … |
| … | | | | | | | | |

![tail by profile](results/tail_by_profile.png)
![jitter vs compute](results/jitter_vs_compute.png)
![tail CDF](results/cdf_tail.png)

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
