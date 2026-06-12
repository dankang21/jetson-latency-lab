# pilot_emc — G0-② EMC sensitivity pilot

Half-day pre-campaign check for the paper's RQ1 ("the EMC clock domain is a
variable that frequency-aware latency estimators absorb into a constant").
Decides whether RQ1 survives before committing to the full 100k campaign.

## What it does

CPU+GPU pinned (`part2/set_domain.sh all`), then EMC swept across
{min, mid, max} via BPMP debugfs, running two workloads per point:

| workload | model | role |
|---|---|---|
| `cnn` | mobilenetv2 (same as Part 1–3) | compute-heavy control |
| `proxy` | GEMV stack, 12×4096×4096 fp16 (~400 MiB/inference) | memory-bound, LLM-decode-shaped |

Each cell: 1000 iters + 100 warmup through `harness.infer_bench`, with a
tegrastats trace and a debugfs rate read-back proving the EMC lock held.

## Kill criteria (pre-registered)

- **RQ1 dead**: p50 compute shift max↔min EMC < 5% on *both* workloads.
- **RQ1 strongest**: small shift on `cnn`, large on `proxy` — the
  workload-dependent error of treating memory time as a frequency-independent
  constant.

## Run

```bash
# 1. generate the proxy model (once, no root)
/home/dk/mobile/rt-infer-bench/vit_export/bin/python3 pilot_emc/make_decode_proxy.py

# 2. sweep (~15 min, needs root for EMC debugfs + SCHED_FIFO)
sudo ./pilot_emc/run_emc_pilot.sh

# 3. verdict
python3 pilot_emc/analyze_emc_pilot.py
```

Restores EMC to dynamic and CPU/GPU to `free` on exit (including on failure).
