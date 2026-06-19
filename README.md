# jetson-latency-lab

**Artifact for _"Edge-Inference Governors Need Memory-Clock State"_** (Jaehoon Kang)
— [arXiv:2606.16106](https://arxiv.org/abs/2606.16106).

The core result: a CPU×GPU-only latency estimator cannot see the memory clock
(EMC), so a deployed deadline governor picks an **infeasible** operating point;
only an EMC-aware model finds the **feasible** side of the energy frontier. The
failure is measured on two integrated SoCs (Orin Nano Super, Orin NX) across
three workload classes (MobileNetV2, ViT-Small, Qwen2.5 token decode), with
calibrated module-rail energy.

This repository holds the measurement harness, the per-run data behind the
paper's tables, and a community replication kit. Bulk raw traces that exceed
GitHub's file-size limits are archived on Zenodo:
[10.5281/zenodo.20694687](https://doi.org/10.5281/zenodo.20694687)
(concept DOI — always resolves to the latest version).

---

## Test bed

| | |
|---|---|
| Boards | Jetson Orin Nano Super 8 GB; Jetson Orin NX 16 GB |
| Shared axis | four lockable EMC points {204, 665.6, 2133, 3199} MHz on **both** SKUs |
| Kernel / L4T | `PREEMPT_RT`, JetPack 6 / L4T R36 |
| Runtimes | ONNX Runtime (CUDA EP) + TensorRT fp16; `llama.cpp` for LLM decode |
| Loop | fixed-rate `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)`, `SCHED_FIFO`, pinned core, `mlockall` |

The exact board/software state the paper was measured on is captured under
[`repro/`](repro/): `kernel.txt`, `l4t_version.txt`, `package_versions.txt`,
`nvpmodel_active.txt`, and `emc_lockable_set.txt`.

> The lockable EMC set {204, 665.6, 2133, 3199} MHz is **measured, not from the
> docs** — JetPack snaps off-grid requests **up** to the next table entry, so
> always confirm the *actual* locked rate, not the requested one.

## Clock-lock preflight — read before any EMC measurement

EMC pinning goes through BPMP debugfs via
[`pilot_emc/emc_ctl.sh`](pilot_emc/emc_ctl.sh):

```bash
sudo ./pilot_emc/emc_ctl.sh points            # list lockable EMC rates
sudo ./pilot_emc/emc_ctl.sh lock 2133000000   # pin EMC to 2133 MHz
sudo ./pilot_emc/emc_ctl.sh status            # confirm rate= and locked=
sudo ./pilot_emc/emc_ctl.sh unlock            # ALWAYS unlock when done
```

> **`bwmgr_halt` is mandatory (L4T R36).** `mrq_rate_locked=1` alone is *not*
> enough — the BPMP bandwidth manager silently re-pulls EMC back to the
> demand-driven rate (e.g. the 204 MHz floor under light load) unless you also
> set `/sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt=1`. `emc_ctl.sh` sets and
> clears it for you. A board that has run `jetson_clocks` locks fine (it sets the
> flag implicitly); a fresh one silently does not. Always `unlock` — a locked-low
> EMC under GPU load can stall the board.

Per-domain CPU/GPU/EMC pinning used by the campaign runners is in
[`part2/set_domain.sh`](part2/set_domain.sh) (`free` / `gpu_only` / `cpu_only` / `all`).

## Reproducing the deployed-governor tables

| paper artifact | committed data | data-gen → analysis |
|---|---|---|
| Table II — MobileNetV2 governor | `results/govrep/` (miss), `results/energy_cal/` (energy), `results/nx_validation/` (fits) | `govrep.sh` + `energy_cal_gov.sh`; governor compute mirrors [`analyze_vit_gov.py`](campaign/analyze_vit_gov.py) |
| Table III — ViT-Small governor | `results/govrep_vit/` | `vit_deploy.sh` → `analyze_vit_gov.py` |
| Table IV — Qwen2.5 decode governor | `results/qwen_deploy/` | `qwen_deploy.sh` → `analyze_qwen_deploy.py` |
| Table V — offset-calibrator estimator | `results/nx_validation/` | `run_estimator_break.sh` → `analyze_estimator.py` |

RQ1/RQ2/RQ3 figure data is in `results/campaign_g2/` (Part-A sweep + Part-B
contention matrix) and `results/pilot_trans/` (actuation-lag probes). Energy is
calibrated 1 ms-sampled onboard-INA `VDD_IN` integration
([`campaign/ina_log.py`](campaign/ina_log.py),
[`campaign/power_probe.py`](campaign/power_probe.py)); `analyze_energy_cal.py`
converts power traces to per-inference joules.

## Raw vs processed data

Under `results/<campaign>/`: `*.csv` are **raw per-cycle / per-token traces**
(`response_us`, `dur_ns`); `*.json` are **per-run summaries**
(`p50` / `p99` / `compute_us`); `*.meta.json` records the clock state, power
mode, and command, so every run is self-describing. Bulk probe traces
(`*.probe.csv`) that exceed GitHub limits live on Zenodo (above).

## Community replication (~20 min on your own Orin)

[`repro/replicate.sh`](repro/replicate.sh) reproduces the two most-scrutinized
RQ1 results — the EMC latency shift (MobileNetV2, 2133 vs 3199 MHz) and the
L2-GEMM inversion — on your board, writing `repro/replication_result.json`:

```bash
sudo apt-get install stress-ng
pip install -r requirements.txt
sudo PY=/path/to/python-with-ort GENPY=/path/to/python-with-onnx ./repro/replicate.sh
```

## Layout

```
harness/      periodic RT loop + pluggable backends (infer_bench.py, backends.py, rt_utils.py)
campaign/     deployed-governor runners (govrep/vit/qwen/energy_cal) + analysis scripts
pilot_emc/    EMC clock control (emc_ctl.sh, bwmgr_halt) + compute-proxy generator
pilot_trans/  frequency-transition / actuation-lag probes (RQ3)
part2/        set_domain.sh -- per-domain CPU/GPU/EMC clock pinning (used by the runners)
tools/        membw memory-bandwidth adversary; slm_decode_bench LLM token bench
repro/        community replication kit + environment snapshot
results/      per-run CSV/JSON/tegra data behind the tables (bulk *.probe.csv on Zenodo)
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The
inference models (MobileNetV2, ViT-Small, Qwen2.5) are not redistributed here;
fetch them from their original sources.

---

*Built by Daniel Kang (dk) @ Cleinsoft — real-time ML for robots.
[cleinsoft.com/dk](https://cleinsoft.com/dk).*
