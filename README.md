# jetson-latency-lab

**Artifact for _"When Memory-Clock State Changes Deadline Admission: Evidence
from Jetson Orin"_** (Jaehoon Kang, 2026).

On integrated SoCs the memory controller has its own DVFS domain. On two Jetson
Orin SKUs this memory clock (EMC) is a deployment state that CPU×GPU latency
estimators omit: their "frequency-independent" term depends on it. On an
EMC-locked Orin NX, a GPU-only MobileNetV2 fit profiled at the deployment
release rate but at another memory point violates a 2 % miss budget at 7/7,
7/9 and 6/9 of the pre-stated deadlines evaluated in three sessions; a two-cell
refit at the deployment memory point admits eight of the nine and violates at
none. Decode horizon and GPU co-tenancy add two further state layers, handled
by bounded probes in a first-order admission prototype whose transfer was
tested in a 14-condition study on both boards. The findings are confined to
the two SKUs.

**Earlier version.** The extended technical report
[arXiv:2606.16106](https://arxiv.org/abs/2606.16106) (v3, July 2026, titled
"Edge-Inference Governors Need Memory-Clock State") is an earlier report of
this study with further characterization material. Its deployment tables and
headline figures, and anything else in it that differs from the article, are
superseded by the article and its supplement ("Supp." below).

**Contents:** [How to cite](#how-to-cite) ·
[What is released](#what-is-released) ·
[Test bed, as measured](#test-bed-as-measured) · [Models](#models) ·
[Clock-lock preflight](#clock-lock-preflight--read-before-any-emc-measurement) ·
[Where each result comes from](#where-each-result-comes-from) ·
[Campaigns E1–E8](#the-septemberoctober-2026-campaigns-e1e8) ·
[Data layout](#data-layout-and-formats) ·
[Zenodo](#bulk-traces-on-zenodo) · [Running the code](#running-the-code) ·
[Layout](#layout) · [License](#license)

---

## How to cite

Please cite the article; the technical report is its earlier version.

```bibtex
@misc{kang2026memclock,
  author = {Jaehoon Kang},
  title  = {When Memory-Clock State Changes Deadline Admission:
            Evidence from {Jetson Orin}},
  year   = {2026},
  note   = {Artifact: https://github.com/dankang21/jetson-latency-lab}
}

@misc{kang2026tr,
  author       = {Jaehoon Kang},
  title        = {Edge-Inference Governors Need Memory-Clock State},
  howpublished = {arXiv:2606.16106v3},
  month        = jul,
  year         = {2026},
  note         = {Extended technical report}
}
```

The bulk traces have their own DOI,
[10.5281/zenodo.21236659](https://doi.org/10.5281/zenodo.21236659)
([details](#bulk-traces-on-zenodo)).

## What is released

| | |
|---|---|
| `harness/` | the periodic-release measurement loop and its ONNX Runtime backends |
| `campaign/` | the board runners (`*.sh`), the deployed pick rules (`admit_pick.py`, `banded_pick.py`, `fresh_pick.py`, `joint_gov_policy.py`) and the analysis scripts (`*.py`) |
| `docs/protocols/` | the protocol logs of the September–October 2026 campaigns, with their freeze and amendment dates as recorded |
| `results/` | the per-run data: per-cycle (per-token) traces, run summaries, INA3221 power logs, EMC lock readbacks and session logs |
| Zenodo | the June–July 2026 bulk traces that exceed GitHub's file limits |

Not released: the manuscript sources, the generators of the manuscript's
LaTeX tables (`campaign/make_tab_*.py`: main-text Tables II and III, Supp.
Tables S3, S4 and S11; they typeset numbers that released printers compute)
and the manuscript figure scripts (`make_fig_tmc.py`, `make_fig_layers.py`,
`make_figs_cal.py`, `make_figs_v2.py`, `make_frontier_fig.py`,
`make_fig_mech2.py`); all are listed in `.gitignore`. Some released analysis
scripts can also write their supplement table with `--tex`. The
[result map](#where-each-result-comes-from) names, for every table, the
released scripts that print its numbers and marks the values that no released
script prints.

## Test bed, as measured

| | Orin Nano Super 8 GB | Orin NX 16 GB (third-party carrier) |
|---|---|---|
| Device-tree model string | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super | NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super |
| Host name in run metadata | `jetson-orin` | `Pandora` |
| L4T / JetPack | R36.5.0 / 6.2.2 (`nvidia-jetpack 6.2.2+b24`) | R36.4.3 / not recorded |
| Kernel | `5.15.185-tegra #1 SMP PREEMPT` (not PREEMPT_RT) | `5.15.148-tegra #3 SMP PREEMPT` (not PREEMPT_RT) |
| CUDA | 12.6 (`cuda-toolkit-12-6 12.6.11`, nvcc 12.6.68) | 12.6 (CUDA SDK 12.6.11) |
| Power mode | MAXN_SUPER (id 2) of `nvpmodel_p3767_0003_super.conf` = [`repro/nvpmodel.conf`](repro/nvpmodel.conf) (its default mode set to MAXN_SUPER) | MAXN_SUPER (id 0) of the stock `nvpmodel_p3767_0000_super.conf`, sha256 `7d7e83807e95e68a934a7a33f61cce80b8a02b170de37be286c617e43f826461` |
| CPU / GPU | 6 cores ≤ 1728 MHz / GPU ≤ 1020 MHz | 8 cores ≤ 1984 MHz / GPU ≤ 1173 MHz |
| Lockable EMC points | 204, 665.6, 2133, 3199 MHz ([`repro/emc_lockable_set.txt`](repro/emc_lockable_set.txt)) | the same four ([`results/nx_validation/discovery.txt`](results/nx_validation/discovery.txt)) |
| Python of the runs | 3.10.12 | 3.10.12 |

[`repro/`](repro/) is a snapshot of the Nano taken on 2026-06-12 (kernel, L4T,
package versions, active power mode and its nvpmodel file, the EMC lockable-set
probe). Every profiling and deployment cell held the power mode at MAXN_SUPER,
pinned the CPU (cpufreq governor `performance`) and the GPU (devfreq
`min_freq` = `max_freq`), and set the EMC alone through the BPMP rate lock;
Supp. Tables S16 and S17 list every knob and every power mode of both boards.

**Runtimes.**
- ONNX Runtime 1.23.0 (`onnxruntime-gpu`, CUDA execution provider) for every
  vision run: models in fp32, the GEMV/GEMM proxies in fp16, graph optimization
  `ORT_ENABLE_ALL`, one intra- and one inter-op thread. The harness stops with an
  error if the session does not resolve to the requested provider. The version is
  recorded for the Nano (`repro/package_versions.txt`); the NX's was not recorded.
- TensorRT 10.3.0 (`10.3.0.30`), fp16, through the ONNX Runtime TensorRT
  execution provider (`--backend onnxruntime-trt`; engine cache in `$TRT_CACHE`,
  default `/tmp/trt_cache`). Used only for the Nano runtime-robustness sweep
  (Supp. §S8, `results/e5_tensorrt`).
- llama.cpp tag `b9592` (commit `ac4cddeb0dbd778f650bf568f6f08344a06abe3a`,
  2026-06-10), CUDA backend, driven by
  [`tools/slm_decode_bench.cpp`](tools/slm_decode_bench.cpp): greedy sampling,
  batch 1, all layers on the GPU, a 64-token prompt (4,096 in the context-axis
  cell of Supp. Table S13) and 40 warm-up tokens before the timed tokens. The
  Nano checkout's `build/CMakeCache.txt` records `CMAKE_BUILD_TYPE=Release`, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=87`,
  `LLAMA_CURL=OFF`, `BUILD_SHARED_LIBS=ON`, `GGML_NATIVE=ON`, `GGML_CUDA_FA=ON`,
  `GGML_CUDA_GRAPHS=ON`; the build command itself was not recorded, and an
  equivalent one is
  `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`.
  The commit and flags of the NX build were not recorded.

**The loop.** Every deployment, admission and transfer run uses
[`harness/infer_bench.py`](harness/infer_bench.py): a periodic loop that
releases one inference at each absolute time t0 + iP
(`clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)`), so an overrun never delays
later releases. Response time is measured from the scheduled release, and every
inference reuses one fixed random input tensor (seed 42). The runners start it
as the invoking user, pinned to one core (NX core 7, Nano core 5; co-runners on
core 5 and 4), with RT throttling off (`sched_rt_runtime_us = -1`) for the
session. Its requests for `SCHED_FIFO` 80 and `mlockall` fail without root, so
these runs are **user-mode `SCHED_OTHER` without memory locking**. Every run JSON records this in `meta.rt_warnings`:
`SCHED_FIFO/affinity denied (need sudo) -> ran SCHED_OTHER` and
`mlockall failed: [Errno 12] Cannot allocate memory`. The affinity is set
before the refused call, so the pin holds. The decode bench prints the same on
stderr (`slm_bench: warning: SCHED_FIFO denied (need root)`,
`slm_bench: warning: mlockall failed`), kept in
`results/horizon_ext/nx/H*/t<k>.log`; its JSON `meta.prio` is the requested
priority, not the effective policy. Runs with an empty `rt_warnings` list come
from the root characterization harness (`SCHED_FIFO` 80 with `mlockall`), for
example the June 2026 Nano campaign (`results/campaign_g2`,
`results/estimator_break`) and the NX 8 Hz profiling cells in
`results/nx_validation`. One user-mode set differs: the Nano energy-calibration
runs `results/energy_cal/inv_*` (`energy_cal_inv.sh`) asked for core 7, which
the 6-core Nano lacks, so the affinity call itself failed and they ran
unpinned, as `SCHED_OTHER` without memory locking; their `rt_warnings` read
`set_realtime failed: [Errno 22] Invalid argument` and the same `mlockall`
line.

**Energy.** [`campaign/ina_log.py`](campaign/ina_log.py) polls the INA3221
`VDD_IN` rail (with `CPU_GPU_CV` and `SOC`) as fast as the driver allows, on
core 0. The median sample interval is 3.6 ms on the NX and 3.7 ms on the Nano
(3.9 ms in the Nano GEMM cells at EMC 3199 MHz); no log is sampled at 1 kHz. In
the interleaved sessions the log stops with the run or at 40 s (NX) / 30 s
(Nano). Energy per inference of a periodic run is the mean `VDD_IN` power over
the log, less its first and last 10 % of samples, divided by the release rate
(`energy()` in [`campaign/e1b_stats.py`](campaign/e1b_stats.py)), so it
includes each period's idle part and the module's baseline draw. For saturated
decode, energy per token is that mean power times the median per-token latency.

## Models

No model file is redistributed. Checksums are of the copies on the Nano; the
NX copies were not hashed.

| Model | File | Source | sha256 |
|---|---|---|---|
| MobileNetV2 | `mobilenetv2-12.onnx` (opset 12, 13,964,571 B) | ONNX Model Zoo, `validated/vision/classification/mobilenet/model/` (URL in [`repro/replicate.sh`](repro/replicate.sh)) | `c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5` |
| ResNet-50 | `resnet50-v1-12.onnx` (ResNet-50 v1, opset 12) | ONNX Model Zoo file name; the download URL was not recorded | `3f03fdef724b22947eed826f1eef1dc5c34151bb4c37d634f1db89dfa2dd1526` |
| YOLOv8s | `yolov8s.onnx` (640×640, batch 1, opset 17, no NMS) | Ultralytics 8.4.62 export of `yolov8s.pt` (COCO), 2026-06-09; the export arguments are in the ONNX metadata, and an equivalent command is `yolo export model=yolov8s.pt format=onnx imgsz=640 batch=1 opset=17 simplify=True` | `f820fa58865b9b05d212c232cb1b7fb744b683f8970c41c9726be5d6e04e02ef` (`yolov8s.pt`: `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`) |
| ViT-Small | `vit_small_patch16_224.onnx` + `vit_small_patch16_224.onnx.data` | timm `vit_small_patch16_224` (the DeiT-S configuration) with `pretrained=False`, i.e. random weights; no accuracy is evaluated. Exported as below (timm 1.0.27, torch 2.12.0, onnx 1.21.0) | `675cec44a264633387049a4d77dd50071996753066dc56a1fcd542a4a005a8ae` / `3298f5c11a74f4d4f94fee9e00477f7ef834065f3ac6ab75da2e0c27b8b6e91e`; the export below reproduces the `.onnx` graph file byte for byte, but not the `.onnx.data` weights file (unseeded random weights) |
| Qwen2.5-1.5B-Instruct Q4_K_M | `qwen2.5-1.5b-instruct-q4_k_m.gguf` (GGUF V3; `general.name` qwen2.5-1.5b-instruct, `general.version` v0.1, `file_type` 15) | Qwen2.5 Instruct GGUF; the download source was not recorded | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| Qwen2.5-0.5B / 3B-Instruct Q4_K_M, Qwen2.5-1.5B-Instruct Q8_0 | `qwen2.5-0.5b-instruct-q4_k_m.gguf`, `qwen2.5-3b-instruct-q4_k_m.gguf`, `qwen2.5-1.5b-instruct-q8_0.gguf` (names from the run metadata) | as above; used on the NX only | not available |
| GEMV decode proxy | `pilot_emc/models/decode_proxy_fp16.onnx` (12 × 4096², fp16) | `python3 pilot_emc/make_decode_proxy.py` | `f254029bc4bc2caa73d923050684238d3b760304961cc7c16dec4dc8bb46d181` |
| L2-resident GEMM | `pilot_emc/models/compute_proxy_fp16.onnx` (64 × 1024³, one shared weight) | `python3 pilot_emc/make_compute_proxy.py` | `6e400df5034b11e610b54d9abaf12840c3bc393b12c2cdaeb702dfd4b9aa7683` |
| GEMM 4 × 2048³ | `pilot_emc/models/compute_proxy_v1_gemm2048.onnx` | `python3 pilot_emc/make_compute_proxy.py --n 2048 --layers 4 --no-shared-weight --out pilot_emc/models/compute_proxy_v1_gemm2048.onnx` | `092ee77e5b1fb5b5ffc74b12aefe6f7a466284793c9e6a1da0fd37c6a2c43804` |

The proxy generators are seeded (42); with onnx 1.21.0 they reproduce the three
files above byte for byte. The runners expect the ONNX models in `$ONNX_DIR`
and the GGUF files in `$GGUF_DIR` ([below](#board-campaigns)). The ViT-Small
export (its script lives outside this repository):

```python
import torch, timm, onnx
m = timm.create_model("vit_small_patch16_224", pretrained=False).eval()
torch.onnx.export(m, torch.randn(1, 3, 224, 224), "vit_small_patch16_224.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=17, dynamic_axes=None)
onnx.checker.check_model(onnx.load("vit_small_patch16_224.onnx"))
```

## Clock-lock preflight — read before any EMC measurement

EMC pinning goes through BPMP debugfs via
[`pilot_emc/emc_ctl.sh`](pilot_emc/emc_ctl.sh) (root):

```bash
sudo ./pilot_emc/emc_ctl.sh status            # rate= locked= min= max=
sudo ./pilot_emc/emc_ctl.sh lock 2133000000   # halt bwmgr, lock, print requested= actual=
sudo ./pilot_emc/emc_ctl.sh unlock            # ALWAYS unlock when done
sudo ./pilot_emc/emc_ctl.sh points            # min, arithmetic midpoint, max: sweep targets, not the lockable set
```

> The lockable set {204, 665.6, 2133, 3199} MHz is **measured, not taken from
> the documentation**: the BPMP rounds an off-table request **up** to the next
> table entry while the write appears to succeed (the Nano probe is in
> `repro/emc_lockable_set.txt`). Always confirm the *actual* rate, not the
> requested one. The September–October 2026 runners lock with readback at the
> start of a session, log the `requested= actual=` line to the session's
> `emc_lock.txt` (`emc_lock_ref.txt` for an E3 reference sweep) and stop on a
> mismatch; the admission, launch and decode-horizon runners re-check the rate
> before every trial or launch and re-lock if it drifted, and the E5-B-live
> runner re-locks before every cell.

> **`bwmgr_halt` is mandatory (L4T R36).** `mrq_rate_locked=1` alone is *not*
> enough: the BPMP bandwidth manager silently re-pulls EMC back to the
> demand-driven rate (e.g. the 204 MHz floor under light load) unless you also
> set `/sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt=1`. `emc_ctl.sh` sets and
> clears it for you. A board that has run `jetson_clocks` locks fine (it sets the
> flag implicitly); a fresh one silently does not. The BPMP `rate` file reports
> the requested value, so it cannot reveal this override; check the tegrastats
> `EMC_FREQ` field under load. Always `unlock`: a locked-low EMC under GPU load
> can stall the board.

The September–October 2026 runners pin the other clocks themselves through
`sudo -n`: GPU devfreq `min_freq` = `max_freq`, cpufreq governor `performance`
on every core, `sched_rt_runtime_us = -1`. On exit they unlock the EMC and
restore the GPU range, `schedutil` and `950000`.
[`part2/set_domain.sh`](part2/set_domain.sh) (Orin Nano paths) pins CPU and GPU
for the June 2026 characterization runners, the pilots and the replication kit.

## Where each result comes from

Data paths are relative to `results/`; runner and analysis script names are
in `campaign/` unless a path is given (run an analysis script as
`python3 campaign/<script>`). The LaTeX tables belong to the manuscript
sources; for each table the last column names the released scripts that print
its numbers. **†** the figure is rendered by a manuscript-only script from the
listed data. **‡** no released script prints these values; the row says why.

### Main text

| Item | Data (`results/`) | Runner (`campaign/`) | Analysis (`campaign/`) |
|---|---|---|---|
| Table I — platforms; workload p50 at EMC 3199 MHz (Nano) | `campaign_g2/partA/emc3199_*.json` | `run_g2.sh A` (root) | `analyze_g2.py` (Part A) |
| Fig. 1 — overview: (a) solo failure, (c) joint stress | as Table II; `joint_hxt/` | as Table II; `joint_hxt.sh`, `tmc_reps.sh` | † `e5_han_baseline.py --model mobilenet` (a), `analyze_joint.py` (c) |
| Fig. 2 — deployed blind vs aware governor, NX MobileNetV2 | `govrep_rev/{nx,nx_mobilenet,nx_mobilenet_s3}`, `refit33/nx/`; dashed line: `nx_validation/nx_est_emc{2133,3199}_gpu{306,1173}_mobilenet.json` (8 Hz cells) | as Table II | `make_fig_govdeploy_rev.py --out FILE` |
| Table II — deployed MobileNetV2 governor | NX sessions `govrep_rev/{nx,nx_mobilenet,nx_mobilenet_s3}`; Nano sessions `govrep_rev/{nano,nano_mobilenet_s2}`; fits `refit33/nx/` (33 Hz, headline panel), `nx_validation/nx_est_emc{2133,3199}_gpu{306,1173}_mobilenet.json` (8 Hz, last panel), `nano_grid/mobilenet/est_emc{2133,3199}_gpu{306,1020}.json` (Nano, 50 Hz) | `govrep_rev.sh` (session 1 on each board); `govrep_rev2.sh` or `govrep_rev3.sh` (later sessions, `MODEL=mobilenet`); `MODE=refit govrep_rev2.sh` (33 Hz endpoint cells) | `govdeploy_stats.py` (every panel: picks, per-run misses, runs over τ per session, paired ΔE with its paired-t and bootstrap intervals; per session the median misses and the deadlines with a violating run; energy and p50/p99 by clock); also `e5_han_baseline.py --model mobilenet` (headline-panel picks and runs over τ per session), `selector_cmp.py` (paired ΔE means) |
| Table III — one cell per (workload, SKU) | as Table II and Supp. S3, S4; `qwen_deploy/`, `qwen3b_deploy/`, `boundary_stats/q15_*` | as those tables; `qwen_deploy.sh`, `qwen3b_deploy.sh`, `boundary_stats.sh` | `solo_stats.py` (the six vision rows, the figures of the table note and the Qwen-1.5B 2,000-token entry), `analyze_qwen_deploy.py` (Qwen-1.5B), `analyze_breadth.py` (Qwen-3B); also `e5_han_baseline.py` (MobileNetV2, NX), `e1b_stats.py` (ViT-S, ResNet-50, YOLOv8s, NX) |
| Table IV — EMC-free replay (E5-B) | `nx_validation_r33/` (latency fits), `govrep_rev/{nx,nx_mobilenet,nx_mobilenet_s3}` (EMC 2133 cells), `nx_pairs/` (EMC 3199 and 665.6 cells), `e5b_live/nx/` (re-run) | `sweep_rate.sh`, `govrep_emcfree.sh` | `e5b_emcfree.py` |
| Fig. 3 — the three state layers | none of its own (conceptual; numbers from Tables II, V, VI and, in the occupancy row, the §V decode runs `boundary_stats/q15_g1173_r*`, `joint_hxt/q15co_g1173_r*`) | — | † `tenancy_stats.py --section decode` (occupancy row: the all-token median shift and the plateau's share and location) |
| Fig. 4 — horizon and tenancy as contract state | `boundary_stats/q15_*`, `horizon_4k/`, `joint_hxt/` | `boundary_stats.sh`, `tmc_reps.sh`, `horizon_4k.sh`, `joint_hxt.sh` | † `analyze_joint.py` (drift, H\*, collision cliff), `tenancy_stats.py --section decode` (panel b: p50, plateau share and location), `qwen3b_drift.py` (the caption's drift range: pooled drift of the 2,000-token runs at 918, 1020, 1122 and 1173 MHz) |
| Table V — Qwen2.5-1.5B decode governor | `qwen_deploy/` (sweeps, 300-token cells, power logs), `boundary_stats/q15_g{918,1020,1173}_r*` (2,000-token runs) | `qwen_deploy.sh`, `boundary_stats.sh`, `tmc_reps.sh` | `analyze_qwen_deploy.py` (picks, 300-token miss, energy ordering), `qwengov_2k.py` (the 2,000-token column with its run-level bound, and the first- and last-500-token misses) |
| Table VI — admission outcomes | `joint_gov/`, `boundary_stats/q15_g1173_r*`, `horizon_4k/`, `joint_hxt/` | `joint_gov.sh`, `jg_extra.sh`, `boundary_stats.sh`, `tmc_reps.sh`, `horizon_4k.sh`, `joint_hxt.sh` | `analyze_joint.py`, `admission_runs.py`; safety-multiplier replay `replay_rsens.py` |
| Table VII — admission evidence by setting | `fresh_loop/`, `banded_loop/`, `launch1020/`, `launch_ext/`, `e3_nx/`, `e3_nano/`, `e3_nx_r33/` | `fresh_loop.sh`, `banded_loop.sh`, `launch1020.sh`, `launch_ext.sh`, `admit_loop.sh`, `admit_refsweep.sh` | `admit_pick.py --selftest` (re-derives the 18 recorded held-out and fresh-probe picks), `e3_repick.py` (round-1 and held-out picks and misses; re-run rows), `launch_ext_stats.py` (one-bin headroom), `e3_stats.py --board nx` and `--board nano`, `e3_tally.py` (sums by setting), `fresh_series.py` (the round-1, held-out and one-bin-headroom rows), `e3_table.py` (the transfer rows: verdict counts by setting under both wordings) |
| Table VIII — where the policy applies | summary (per-condition endpoints: Supp. S11) | — | — |

Other numbers in the main text:

| Result | Data (`results/`) | Analysis (`campaign/`) |
|---|---|---|
| §III EMC of the Nano's stock power modes (tegrastats samples per mode) | `e1_governor/`, `replication_25w/`, `replication_15w/` | `powermode_trace.py` |
| §III thermal gate | every September–October `session.log` | `thermal_record.py` (Supp. S18) |
| §III one-hour saturated-decode soak | `soak_qwen/` (power trace on Zenodo) | `deployment_stats.py --section soak` (per-window token miss, medians, tegrastats temperature and clocks; the power log once gunzipped to `results/soak_qwen/soak.power.csv`) |
| §III energy per inference by clock, NX MobileNetV2 | as Table II | `govdeploy_stats.py` |
| §IV admission (36 of 37 NX vision deadlines), compliance and frontier position of the refit | as Table II and Supp. S3 | `nx_vision_frontier.py` |
| §IV safety multiplier r = 1.035: solo p99/p50 dispersion of the profiling cells (with Supp. §S2) | `campaign_g2/partA/`; the endpoint cells of the deployed fits (`nx_validation/nx_est_*`, `refit33/`, `nano_grid/`) | `analyze_g2.py` (p99/p50 at 3199 MHz), `profile_dispersion.py` |
| §IV GEMM sign test of the monotone power law, and the Han-fit errors on the 24-cell Nano grid (Supp. §S3) | `estimator_break/` | `han_fit.py --loco-table --key compute_us` |
| §IV cross-SKU 2133→3199 MHz penalties | `campaign_g2/partA/`, `nx_validation/nx_emc*` | `emc_axis_stats.py --section xsku` (response-time p50 of each run JSON; `analyze_g2.py` prints the Nano compute medians) |
| §IV two-cell refits from every pair of the eight GPU cells | `estimator_break/emc2133_gpu*` | `emc_axis_stats.py --section pairs` (compute p50, as `analyze_estimator.py`) |
| §IV energy of the maximum clock against the aware picks; deployed medians of the blind breadth violations | `govrep_rev/{nx,nx_mobilenet,nx_mobilenet_s3}`; `govrep_rev/nx_{vit,yolov8s,resnet50}`, `refit33/nx_*` | `deployment_stats.py --section maxclock`, `--section breadth` |
| §IV tail state of the June session: slow-mode share and offset, the same p50, the full-sweep picks at 4.6–5.0 ms (blind misses per run and their mean; the aware arm's top-clock fallback), the later sessions unimodal | `govrep/` (2026-06-17); `govrep_rev/{nx,nx_mobilenet,nx_mobilenet_s3}`; fits `nx_validation/nx_est_emc{2133,3199}_gpu*_mobilenet.json` | `deployment_stats.py --section june` (slow mode of a run: its cycles more than 80 µs above the run's response-time p50; share = their fraction, offset = their median minus that p50) |
| §V co-tenancy characterization; the co-tenant decode distribution quoted in §VI | `tenancy/`; solo runs `govrep/`, `nx_validation/`; decode `boundary_stats/q15_g1173_r*`, `joint_hxt/q15co_g1173_r*` | `tenancy_stats.py --section victim` (MobileNetV2 under the 10 Hz co-runner; the EMC samples of `pin_check.tegra`), `--section decode` (solo and co-tenant medians over the same tokens, all 2,000 and the first 300; the collision plateau's share, location and 1 ms bins; Figs. 3 and 4b) |
| §V Qwen2.5-3B drift | `boundary_stats/q3_*` | `qwen3b_drift.py` |
| §VI, §VII decode drift under the co-tenant; §VI decode-probe autocorrelation | `joint_hxt/`, `boundary_stats/q15_*` | `tenancy_stats.py --section drift` (pooled and per-run), `--section probe` |
| §VII EWMA level calibrator; Supp. §S2 online calibrator | `govrep/`, `govrep_rev/nx_mobilenet/`, `live_gov/` | `replay_ewma.py` (escape and first switch per gain and cadence; after the first switch, cycles on an infeasible clock and the excursions above the landing clock; `--per-run` for every run) |
| §VII band-width sensitivity | `joint_gov/probe_co_g1173.csv`, `fresh_loop/vprobe_t*`, `banded_loop/probe_t*` | `replay_bandwidth.py` |
| §VIII-A rate-matched sensitivity | as Supp. S12 | `e3_repick.py` |
| §VIII-A summed utilization u and the ≈ 2T̂(f) stretch | `e3_nano/` | `e3_util_offline.py` |
| §VIII-A tenant GPU duty, inference-occupied fraction | `refit33/nx*`, `govrep_rev/nx*`, `govrep/*.tegra` | `tenancy_stats.py --section duty`, `--section occupancy` (against the June tegrastats GPU load) |

Table VI's measured miss of the 2,000-token decode contract (1173 MHz,
D = 28 ms) is 84 of 16,000 tokens, exactly 0.525 %: `analyze_joint.py` prints
it as 0.53 % and `replay_rsens.py` as 0.52 %, two roundings of the same tie.

### Supplement

| Item | Data (`results/`) | Runner (`campaign/`) | Analysis (`campaign/`) |
|---|---|---|---|
| S2 directed profile pairs | `nx_pairs/`, `govrep/`, `nx_validation/` | `nx_wave2.sh`, `govrep.sh`, `govrep_hi.sh`, `run_nx_validation.sh` | `deployment_stats.py --section pairs` (full-sweep fit of the response-time p50 at the source EMC point; a declined pick is marked and runs at 1173 MHz) |
| S3 NX breadth (ViT-S, YOLOv8s, ResNet-50) | `govrep_rev/nx_{vit,yolov8s,resnet50}`, `refit33/nx_{vit,yolov8s,resnet50}`; grids from `nx_validation/nx_est_*` | `govrep_rev2.sh` or `govrep_rev3.sh` with `MODEL=vit`, `yolov8s`, `resnet50`; refits `MODE=refit govrep_rev3.sh` | `e1b_stats.py` |
| S4 Nano ViT-Small grid | `govrep_rev/nano_vit`, `refit33/nano_vit`, `nano_grid/vit` | `MODEL=vit govrep_rev.sh`, `MODE=refit MODEL=vit govrep_rev3.sh` | `nano_vit_grid.py` (the rows and the caption's figures) |
| S5 Han-style baseline (E5) | `nx_validation_r33/`; the sessions of Table II and S3; `refit33/nx*` | `sweep_rate.sh` | `e5_han_baseline.py` (fits: `han_fit.py`) |
| S6 selector comparison | as S5 | — | `selector_cmp.py` |
| S7 contrast fits | as S5 and S8 | — | `e5b_contrast.py` (several minutes) |
| S8 EMC-free replay (E5-B) | as Table IV | `govrep_emcfree.sh` | `e5b_emcfree.py` |
| S9 interleaved re-run (E5-B-live) | `e5b_live/nx/` | `govrep_emcfree.sh` | `e5b_emcfree.py --json F` then `e5b_live_table.py F` |
| S10 decode horizon under co-tenancy (E8) | `horizon_ext/nx/H{1000,3000}`, `joint_hxt/` (2,000 tokens), `boundary_stats/` (solo γ; Qwen2.5-3B model axis) | `horizon_ext.sh` | `horizon_ext_stats.py` (the H = 1,000 and 3,000 launches and, by default, the nine 2,000-token runs of `joint_hxt/`; `--no-h2000` leaves those out); model axis `qwen3b_drift.py` |
| S11 transfer study (E3) | `e3_nx/{C0,N1…N9}`, `e3_nano/{A1…A4,A5r,A6r,A7}` | `admit_loop.sh`, `admit_refsweep.sh`, `nano_e3_*.sh` | `e3_stats.py --board nx`, `--board nano`; `e3_table.py` (the rows with the probe-margin correlation, the verdict under both wordings and the footnote figures) |
| S12 rate-matched sensitivity, "Nano rows" paragraph | `e3_nx/`, `e3_nano/A1…A4`, `refit33/nx/`, `refit33/nano/`, `e3_nx_r33/` | `results/e3_nx_r33/run_r33.sh`; `MODE=refit govrep_rev2.sh` | `e3_repick.py [--board nx or nano]` |
| S13 EMC exposure of LLM decode | `llm_axes/` | `c10_llm_axes.sh` | `emc_axis_stats.py --section llm` (median of the three runs' per-token p50; energy per token) |
| S14 held-out error with an EWMA offset | `estimator_break/` | `run_estimator_break.sh` | `analyze_estimator.py` (blind, EMC table), `replay_ewma.py` (offset) |
| S15 co-tenancy ablation ladder | `joint_gov/`, `joint_hxt/` | `joint_gov.sh`, `jg_extra.sh`, `tmc_reps.sh`, `joint_hxt.sh` | `analyze_joint.py`, `admission_runs.py` |
| S16 platform and harness knobs | run metadata, `emc_lock.txt`, the runners, `repro/` | — | — |
| S17 nvpmodel power modes | `repro/nvpmodel.conf` (Nano); the NX file named in the test-bed table | — | `nvpmodel_modes.py [CONF[:LABEL] ...]` |
| S18 thermal-gate record | every `session.log` from September 2026 on; `*.tegra` traces | — | `thermal_record.py` (its pre-September summary needs `e4_replication/`, `partb_ext/` and `e4_nx/` from Zenodo; a WARN line names a missing one) |
| S19 safety-multiplier sensitivity | as Table VI | — | `replay_rsens.py` |
| S20, S21 fresh-probe trials | `fresh_loop/` (vision and decode arms, `decisions.csv`), `banded_loop/`, `launch1020/`, `launch_ext/` | `fresh_loop.sh`, `banded_loop.sh`, `launch1020.sh`, `launch_ext.sh` | `fresh_series.py` (every trial and aggregate row; the decode arm's decisions re-derived), `e3_repick.py` (vision picks and misses), `replay_bandwidth.py` (slack), `launch_ext_stats.py` |
| S22 band-width sensitivity | as §VII | — | `replay_bandwidth.py` |
| S23 the EMC axis, consolidated | `campaign_g2/partA/` (Nano, ORT), `e5_tensorrt/` (Nano, TensorRT), `nx_validation/nx_emc*`, `nx_validation/nx_est_emc*_gpu1173_vit.json` (NX) | `run_g2.sh`, `run_nx_validation.sh`, `vit_deploy.sh`, harness `--backend onnxruntime-trt` | `emc_axis_stats.py --section s23` (response-time p50 of each run JSON, per-token compute time for the Qwen rows; with the compute p50, run length, release period and a BACKLOG flag where the compute time exceeds the release period); `analyze_g2.py` prints the Nano ONNX Runtime compute p50 |
| S24 EMC under the Nano's stock power modes | `e1_governor/`, `replication_25w/`, `replication_15w/` | `run_estimator_break.sh`, `run_25w_replication.sh`, `run_15w_trace.sh` | `powermode_trace.py`, `analyze_replications.py` |
| S25 held-out estimator error | `estimator_break/` | `run_estimator_break.sh` | `analyze_estimator.py` |
| S27 TensorRT vs ONNX Runtime penalty | `e5_tensorrt/`, `campaign_g2/partA/` | harness `--backend onnxruntime-trt`, `run_g2.sh` | `emc_axis_stats.py --section s27` (response-time p50) |
| §S2 margin provenance (r = 1.035) | `campaign_g2/partA/`; the endpoint cells of the deployed fits | — | `analyze_g2.py` (p99/p50@3199 column), `profile_dispersion.py` |
| §S2 thermal figures of the deployment sessions and the E5-B-live re-run | `govrep_rev/*/session.log`, `e5b_live/nx/session.log` | — | `deployment_stats.py --section thermal` |
| §S2 acceptance and occupancy; phase-locking detail | `govrep_rev/nx*`, `govrep/*.tegra`; `e3_nx/`, `e3_nano/` | — | `tenancy_stats.py --section occupancy`, `--section phase` |
| §S3 batch axis, session repeats, low-clock collapse and two-cell pairs | `batch_axis/`, `parta_reps/`, `parta_reps_nx/`, `campaign_g2/partA/`, `estimator_break/` | `c9_batch.sh`, `c6_reps.sh` | `emc_axis_stats.py --section batch` (compute p50), `--section reps` (response-time p50), `--section xsku`, `--section pairs` |
| §S4 energy and Pareto dominance | `energy_cal/` | `energy_cal_inv.sh`, `energy_cal_gov.sh` | `analyze_energy_cal.py` |
| §S5 tails and bursts | `campaign_g2/partB/`; replication cells `e4_replication/` (the NX cells also in `e4_nx/`), `partb_ext/` (Zenodo) | `run_g2.sh B`, `e4_replication.sh`, `nano_c7.sh` | `analyze_g2.py`, `analyze_bursts.py`, `analyze_evt_validation.py` (GPD validation and threshold spread), `tails_mech_stats.py --section worst`, `--section nx` (the replication cells) |
| §S6 shared stall signature | `mech_dose/`, `mech_nx/`; `partb_ext/` (Zenodo) | `mech_dose.sh`, `nano_wave2a.sh`, `nx_wave2.sh` | `mech_kdur_extract.py`, `mech_kdur_ts.py` (kernel durations from Nsight Systems exports, which the runners delete after extraction and which are not released; their outputs `mech_dose/kdur_*.txt`, `mech_dose/ts_ts_*.csv` and `mech_nx/kdur_*.txt` are), `tails_mech_stats.py --section ext`, `--section cycles`, `--section slow`; † Fig. S9; ‡ the Nsight Compute kernel-replay figures (8 % faster at 3199 MHz in isolation, L2 hit rate ≈ 83 %) and those of the fine-grained GPU-clock, temperature and supply-current trace (1013–1017 MHz, 51 vs 52 °C, +9 to 15 %): their data are not in this release |
| §S7 actuation lag | `pilot_trans/` (probe traces on Zenodo); live first-miss upshift `live_gov/emcsw_*` | `pilot_trans/run_trans_pilot.py`, `nx_live.sh` | `pilot_trans/analyze_trans.py`, `tails_mech_stats.py --section live` (the live upshift) |
| §S9 independent burst replication | `p3_*`, `p3hi_*` | `part3/run_part3.sh`, `part3/run_part3_highload.sh` (stress-ng, June 2026) | `part3/analyze_part3_bursts.py` (P[miss \| miss] at the p99.9-anchored deadline, per cell with its stressor) |

Tables S1, S26 and S28 have no data of their own. Supp. Figs. S1, S2, S6–S8 and
S10–S12 are rendered by `python3 campaign/make_figs.py --out DIR` (from
`campaign_g2/` and `estimator_break/`; Fig. S10 plots values from
`results/pilot_trans/analysis_trans.txt`; Fig. S12 plots
`results/estimator_break/govsim_curves.json`, which
`python3 campaign/analyze_governor_sim.py --json PATH` regenerates byte for
byte, while without `--json` the script only prints its table); Figs. S3, S4,
S5 and S9 (†) by manuscript-only scripts, from `campaign_g2/partA` and
`nx_validation/nx_emc*` (S3), `energy_cal/inv_*` (S4), `govrep/`,
`govrep_vit/`, `energy_cal/` and `qwen_deploy/` under the full-sweep fit (S5)
and `mech_dose/kdur_*` (S9).

## The September–October 2026 campaigns (E1–E8)

The campaign codes appear in script headers, folder names and the protocol
logs. They are unrelated to the older folder names `e1_governor`,
`e2_gpucap`, `e3_governor`, `e4_replication` and `e5_tensorrt`
([glossary](#data-layout-and-formats)).

| Code | What | Runner (`campaign/`) | Data (`results/`) | Analysis | Protocol (`docs/protocols/`) |
|---|---|---|---|---|---|
| E1 | balanced, interleaved blind/aware replication of Table II: every clock once per pair index, 10 pairs × 1,000 cycles, thermally gated, EMC locked at 2133 MHz with readback; three NX and two Nano sessions, 2026-09-14 to 09-16 | `govrep_rev.sh`, `govrep_rev2.sh`, `govrep_rev3.sh` | `govrep_rev/{nx,nx_mobilenet,nx_mobilenet_s3,nano,nano_mobilenet_s2}`; 33 Hz endpoint cells `refit33/nx/` | as Table II | `E1_E2_protocol.md` |
| E1b | the same design for ViT-Small, YOLOv8s and ResNet-50 on the NX and ViT-Small on the Nano, one session each | `govrep_rev2.sh`, `govrep_rev3.sh`, `govrep_rev.sh` (Nano ViT-Small) | `govrep_rev/{nx_vit,nx_yolov8s,nx_resnet50,nano_vit}`, `refit33/*_{vit,yolov8s,resnet50}` | `e1b_stats.py` | `E1_E2_protocol.md` (E1b) |
| E2 | deadline grids fixed before the E1/E1b runs | — | — | grids in `govdeploy_common.py` and `e1b_stats.py` | `E1_E2_protocol.md` |
| E3 | transfer study of the frozen dispersion-banded admission rule: 14 transfer conditions on both boards, two 7 Hz mechanism conditions (N9, A7) added by a dated amendment, and the calibration control C0 | `admit_loop.sh`, `admit_refsweep.sh`, `nano_e3_*.sh` | `e3_nx/`, `e3_nano/`; post-hoc rate-matched re-run `e3_nx_r33/` | `e3_stats.py`, `e3_table.py`, `e3_tally.py`, `e3_repick.py`, `e3_util_offline.py` | `E3_protocol.md` (frozen 2026-09-14; amendment log §8), `E3_runner_notes.md` |
| E4 | 16 more fresh launches at the one-bin-headroom design point (1020 MHz, D = 12 ms), 24 with the eight of July 2026 | `launch_ext.sh --launches 16` | `launch_ext/nx/mobilenet_vit10_g1020_D12/`, `launch1020/` | `launch_ext_stats.py`, `fresh_series.py` | — |
| E5 | Han-style memory-aware latency model (H1: deployment point held out; H2: in-sample with a free offset) inside the deployed pick rule, scored offline on the E1/E1b sessions | `sweep_rate.sh` (EMC {665.6, 2133, 3199} × 10 GPU clocks, 800 cycles at each model's release rate) | `nx_validation_r33/` | `han_fit.py`, `e5_han_baseline.py`, `selector_cmp.py`, `e5b_contrast.py` | `E1_E2_protocol.md` (E5, 2026-09-16) |
| E5-B | EMC-free replay of a Han-derived joint-scaling selector (energy per period), with an interleaved re-run of its picks (E5-B-live: seven cells × ten rounds, EMC re-locked with readback before every cell, 2026-10-28) | `govrep_emcfree.sh` | `e5b_live/nx/` | `e5b_emcfree.py`, `e5b_live_table.py` | `E1_E2_protocol.md` (E5-B, E5-B-live, 2026-10-28) |
| E6 | paired intervals for every energy difference | — | the E1/E1b sessions | `govdeploy_stats.py`, `solo_stats.py`, `nano_vit_grid.py`, `e1b_stats.py`, `selector_cmp.py`; June–July pairs `energy_pairs_ci.py` | — |
| E7 | knob-by-knob and power-mode tables (Supp. S16, S17) | — | run metadata, `repro/` | `nvpmodel_modes.py` | — |
| E8 | decode horizon under the 10 Hz ViT-Small co-tenant at H = 1,000 and 3,000 tokens, five fresh launches each (Qwen2.5-1.5B Q4_K_M, GPU 1173 MHz, EMC 2133 MHz, 2026-09-16) | `horizon_ext.sh --H 1000 --launches 5` (and `--H 3000`) | `horizon_ext/nx/` | `horizon_ext_stats.py` | `E8_protocol.md` (frozen 2026-09-15) |

E1 and E1b runners, by session: the first session on each board (`nx`,
`nano`) and the Nano ViT-Small session ran `govrep_rev.sh`; the later sessions
used `govrep_rev2.sh` / `govrep_rev3.sh`, which run the same protocol for any
model (`rev3` adds `OUTDIR`, `HZ_OVERRIDE` and a per-model refit folder). To
reproduce the layout:

```bash
PAIRS=10 ./campaign/govrep_rev.sh                          # session 1 -> results/govrep_rev/<board>
MODEL=mobilenet CLOCKS="612 714 816 918 1020 1122 1173" \
  OUTDIR=results/govrep_rev/nx_mobilenet_s3 ./campaign/govrep_rev3.sh
MODEL=mobilenet HZ_OVERRIDE=50 CLOCKS="408 510 612" \
  OUTDIR=results/govrep_rev/nano_mobilenet_s2 ./campaign/govrep_rev3.sh   # Nano, 50 Hz
MODEL=vit CLOCKS="612 714 816 918 1020 1122" ./campaign/govrep_rev3.sh  # -> govrep_rev/nx_vit
MODE=refit ./campaign/govrep_rev2.sh                       # MobileNetV2 33 Hz cells -> results/refit33/<board>
MODE=refit MODEL=vit ./campaign/govrep_rev3.sh             # -> results/refit33/<board>_vit
./campaign/sweep_rate.sh                                   # E5 sweeps -> results/nx_validation_r33
```

The session logs record at their start the board, host, checkout commit and
parameters of the run (the sweep log of `nx_validation_r33/` has no start
line).

### Transfer study (E3)

| Board | Primary (rate, D) | Conditions (tenant at rate; launches) |
|---|---|---|
| NX | MobileNetV2 (33 Hz, D = 12 ms) | C0 ViT-S 10 Hz (10, calibration control); N1 YOLOv8s 5 Hz, N2 YOLOv8s 10 Hz, N3 ResNet-50 5 Hz, N4 ResNet-50 10 Hz (20 each) |
| NX | ViT-Small (20 Hz, D = 32.5 ms) | N5 YOLOv8s 5 Hz, N6 YOLOv8s 10 Hz, N7 ResNet-50 5 Hz, N8 ResNet-50 10 Hz (20 each); N9 YOLOv8s 7 Hz (10, mechanism test) |
| Nano | MobileNetV2 (33 Hz, D = 13 ms) | A1 YOLOv8s 5 Hz, A2 YOLOv8s 10 Hz, A3 ResNet-50 5 Hz, A4 ResNet-50 10 Hz (10 each) |
| Nano | ViT-Small (20 Hz, D = 37.5 ms) | A5r YOLOv8s 5 Hz, A6r YOLOv8s 10 Hz (10 each; re-runs of A5, A6); A7 YOLOv8s 7 Hz (10, mechanism test) |

Per launch, `admit_loop.sh` passes the thermal gate, checks the EMC lock,
starts a fresh co-runner and waits (up to 120 s) for the harness to signal
readiness after its warm-up (`--ready-file`; added after the Nano A1–A6 trials,
so A5r, A6r, A7 and every NX condition ran with it), takes a 200-cycle probe at
the top GPU clock (NX 1173, Nano 1020 MHz), computes the point and banded picks
with `admit_pick.py` (written to `t<k>.pick.json`), and runs the 1,000-cycle
banded contract, plus the point contract when the band moved the pick.
`admit_refsweep.sh` runs one reference contract per grid clock under one
co-runner launch. The exact parameters of every condition are in its
`condition.txt`; the Nano chains are `nano_e3_*.sh`, for example:

```bash
./campaign/admit_refsweep.sh --primary mobilenet --co resnet50 --co-hz 5 --D 12.0 --cond N3
./campaign/admit_loop.sh --primary mobilenet --co resnet50 --co-hz 5 --D 12.0 --trials 20 --cond N3
python3 campaign/e3_stats.py --board nx --cond N3
```

Kept as recorded and not scored as band results: `e3_nano/A5` and `A6`,
protocol-execution failures (the probe had not seen the tenant) re-run as
A5r/A6r under the readiness check. The NX reference sweeps date from
2026-09-16 and the NX admission loops from their re-run on 2026-10-13. The
`WARN` cross-check lines that `e3_stats.py --board nx` prints come from the
2026-09-16 runs, whose log lines share each condition's `session.log`, and
affect no number: 180 `t<k> PICK FAIL` lines of the discarded first admission
series (its run-time pick failed, so it produced probes but no contracts, and
its probe files were discarded) and one `f_oracle=INFEASIBLE vs recomputed <f>`
line per condition, because the run-time scoring of each reference sweep
logged `m=FAIL` for every cell; `e3_stats.py` recomputes the frontier from the
recorded `ref_g<f>.csv` files.
`e3_nx_r33/{C0r33,N3r33}` is a post-hoc re-run of C0 and N3 with the 33 Hz
endpoint cells (`results/e3_nx_r33/run_r33.sh`, 2026-10-23), not a registered
condition.

### Thermal gate

Sessions from September 2026 on are thermally gated. Before every gated unit
(an interleaved pair, an endpoint-refit repeat, a rate-matched sweep cell, an
EMC-free re-run round, an admission trial, a whole reference sweep, a launch)
the runner reads the hottest of `/sys/devices/virtual/thermal/thermal_zone*/temp`
and waits in 20 s steps, for at most 60 checks (20 min), until it reads
≤ 55 °C, then proceeds regardless (`tgate()` in `govrep_rev*.sh`,
`sweep_rate.sh`, `govrep_emcfree.sh`, `admit_loop.sh`, `admit_refsweep.sh`,
`launch_ext.sh`, `horizon_ext.sh`). The `session.log` files hold what the
runners logged: the end-of-cell reading (`t=`, in millidegrees) of every
deployment, refit, sweep and re-run cell, and for admission trials, reference
sweeps and launches also the reading right after the gate (`tgate=`) and every
time-out (`WARN thermal gate timed out`). `campaign/thermal_record.py`
rebuilds Supp. Table S18 from them; its summary of the earlier, ungated
sessions also counts the tegrastats traces of `e4_replication/`,
`partb_ext/` and `e4_nx/`, so extract those Zenodo archives into `results/`
first (a WARN line names a missing one). It counts every `*.tegra` file under
`results/` except those of `e3_governor/`, whose four NX governor cells no
result of the article uses.

### Protocols

[`docs/protocols/`](docs/protocols/) holds the protocol logs:
`E1_E2_protocol.md` (E1/E2 frozen 2026-09-14; E1b 2026-09-15; the amendments;
E5 2026-09-16; E5-B and E5-B-live 2026-10-28), `E3_protocol.md` (frozen
2026-09-14; amendment log in §8, including the 2026-10-23 post-hoc entry),
`E3_runner_notes.md` (2026-09-14: how the runners implement E3) and
`E8_protocol.md` (frozen 2026-09-15). The dates are those recorded in the logs;
the files entered version control with this release, as each header states.

## Data layout and formats

**Per-run files** (`harness/infer_bench.py --out <tag>`):
- `<tag>.csv` — one row per cycle: `cycle, release_jitter_us, compute_us,
  response_us, deadline_miss`.
- `<tag>.json` — percentiles (p50, p90, p99, p99.9, p99.99, mean, std, min,
  max) of `response_us`, `compute_us` and `release_jitter_us`;
  `deadline_miss_count/rate` at the harness deadline, which is the period
  unless `--deadline-ms` was given (the analyses re-score the CSV at the
  article's deadlines); and `meta` = label, backend, model, hz, period and
  deadline, iters, warmup, cpu, prio, mlock, host, kernel, python, timestamp,
  `rt_warnings`.
- `<tag>.power.csv` — `t, vdd_in_mw, cpugpu_mw, soc_mw` from `ina_log.py`.
- Decode runs (`tools/slm_decode_bench`): `<tag>.csv` = `token, dur_ns`;
  `<tag>.json` = `compute_us` percentiles and `meta` (model path, `ngl`,
  `prompt_tokens`, `warmup`, `tokens`, `prefill_ms`, `kv_range`, `cpu`, `prio`).

**Session files** (September–October 2026 folders): `session.log` (start line
with board, host, commit and parameters; one line per cell, trial or launch with
its thermal readings; co-runner readiness; gate time-outs), `emc_lock.txt` (BPMP
lock readback, `requested= actual=`) and, for E3, E4 and E8, `condition.txt`.
E3 trial files: `t<k>.probe.{csv,json}` (200-cycle probe), `t<k>.pick.json` (the
pick computed at run time), `t<k>.contract_b.{csv,json}` (banded contract),
`t<k>.contract_pt.{csv,json}` (point contract, only when the band moved the
pick), `t<k>.done`, and `ref_g<f>.{csv,json}` (reference sweep). The clock state
of every run is in its file name (`g<f>`, `gpu<f>`: GPU MHz; `emc<E>`, `e<E>`:
EMC MHz), in `emc_lock.txt` where recorded, and in tegrastats traces (`*.tegra`)
where a campaign logged one. Only the actuation pilot (`results/pilot_trans`)
writes `*.meta.json` files.

**Folder glossary** (`results/`):

| Folders | Period | Role |
|---|---|---|
| `govrep_rev/`, `refit33/`, `nx_validation_r33/`, `e3_nx/`, `e3_nano/`, `e3_nx_r33/`, `launch_ext/`, `horizon_ext/`, `e5b_live/` | Sept.–Oct. 2026 | the campaigns E1–E8 above; `refit33/nano/` is the Nano 33 Hz re-profile of 2026-10-28 and `refit33/*/cells_mobilenet/` the median cells used by the re-run and the re-picks |
| `campaign_g2/`, `estimator_break/`, `nx_validation/`, `nano_grid/`, `nx_pairs/`, `joint_gov/`, `joint_hxt/`, `boundary_stats/`, `horizon_4k/`, `fresh_loop/`, `banded_loop/`, `launch1020/`, `qwen_deploy/`, `qwen3b_deploy/`, `llm_axes/`, `tenancy/`, `live_gov/`, `parta_reps/`, `parta_reps_nx/`, `batch_axis/`, `mech_dose/`, `mech_nx/`, `pilot_emc/`, `pilot_trans/`, `e1_governor/`, `replication_15w/`, `replication_25w/`, `e5_tensorrt/`, `soak_qwen/` | June–July 2026 | characterization and admission runs that the article and supplement still use (map above) |
| `govrep/`, `govrep_vit/`, `govrep_resnet50/`, `govrep_yolov8s/`, `energy_cal/`, `ab_energy/`, `nano_d9/` | June–July 2026 | **superseded** as deployment tables: the earlier report's governor tables and their energy columns, built from these runs, are replaced by the interleaved sessions in `govrep_rev/` (the paired intervals of those energy columns: `energy_pairs_ci.py`). `govrep/` and `govrep_vit/` still back Supp. Fig. S5 (full-sweep fit) and the June-trace EWMA replay; `energy_cal/` backs Supp. §S4 |
| `govrep_rev/nano_mobilenet_33hz_deviation/` | Sept. 2026 | set aside: a Nano MobileNetV2 session run at 33 Hz instead of the stated 50 Hz; not used |
| `e1_governor/`, `e2_gpucap/`, `e5_tensorrt/`, and on Zenodo `e3_governor/`, `e4_replication/` | June 2026 | older names, unrelated to the campaign codes E1–E8 |

## Bulk traces on Zenodo

The article cites
[10.5281/zenodo.21236659](https://doi.org/10.5281/zenodo.21236659), the
**version DOI of the third deposit** (2026-07-07, currently the latest). The
concept DOI [10.5281/zenodo.20694687](https://doi.org/10.5281/zenodo.20694687)
always resolves to the latest version. All three deposits date from June–July
2026: they hold the bulk traces of that period's characterization and
deployed-governor campaigns (those of the earlier technical report) that exceed
GitHub's file limits. **The September–October 2026 campaigns are small and are
in this repository, not on Zenodo**, including their per-cycle probe traces and
session logs.

Third deposit (10.5281/zenodo.21236659):

| File | Contents |
|---|---|
| `zenodo_v2_bulk.tar.gz` | repository-relative `results/pilot_trans/`, `results/pilot_trans_run1/` and `results/pilot_trans_run2_n40/` Nano frequency-transition probe traces (`*.probe.csv`), `results/e4_replication/`, `results/e5_tensorrt/`, `MANIFEST.txt`; the same archive as the second deposit |
| `partb_ext_nano.tar.gz` | 100k-cycle tail cells on the Nano (`results/partb_ext/`) |
| `partb_nx.tar.gz` | Part-B replication cells on the NX (`results/e4_nx/`; `e4_replication_nano.tar.gz` holds the same NX run files without their tegrastats traces, lock readbacks and adversary logs) |
| `e4_replication_nano.tar.gz` | root-protocol replication runs on the Nano, and the run files of the NX cells of `partb_nx.tar.gz` (`results/e4_replication/`) |
| `e3_governor_nx.tar.gz` | NX governor-cell auxiliary traces (`results/e3_governor/`) |
| `soak_power_1khz.csv.gz` | the `VDD_IN` log of the one-hour saturated Qwen2.5 decode soak (`results/soak_qwen/soak.power.csv`); despite the file name its median sample interval is 3.6 ms |
| `nx_probe_emc_3199_2133.csv.gz`, `nx_probe_gpu_1173_816.csv.gz` | NX EMC- and GPU-transition probe traces (not used in Supp. §S7, which reports only the Nano pairs; see below) |
| `README_v3.md`, `SHA256SUMS` | file table and checksums |

Earlier deposits: 10.5281/zenodo.20694688 (2026-06-15; the Nano probe traces
with their per-transition record files) and 10.5281/zenodo.20745228
(2026-06-18; `zenodo_v2_bulk.tar.gz`). `pilot_trans/analyze_trans.py` needs
all seven probe traces of `results/pilot_trans/`: the six Nano traces in
`zenodo_v2_bulk.tar.gz` (extract it at the repository root) and the NX GPU
trace `nx_probe_gpu_1173_816.csv.gz`, gunzipped to
`results/pilot_trans/gpu_1173000000_816000000.probe.csv`; without them it stops
with a one-line message that names the missing files. The NX GPU trace is needed
only for the script to run: the script prints that pair, but its data-quality
gate excludes it from the verdict (it prints `probe_rc=2`: the probe's exit
code, `probe_returncode` in `results/pilot_trans/gpu_1173000000_816000000.meta.json`), and Supp. §S7, like
the shipped `results/pilot_trans/analysis_trans.txt`, reports only the six Nano
pairs.
`campaign/thermal_record.py` scans `e4_replication/`, `partb_ext/` and `e4_nx/`;
`campaign/tails_mech_stats.py` reads `e4_replication/` (or `e4_nx/` for the
NX cells) and `partb_ext/`, and names the archive of every cell it skips;
`campaign/deployment_stats.py` summarizes the soak's power log once
`soak_power_1khz.csv.gz` is gunzipped to `results/soak_qwen/soak.power.csv`.
The `*.tar.gz` archives other than `zenodo_v2_bulk.tar.gz` hold folders
relative to `results/` (`tar -xzf ARCHIVE -C results`).

## Running the code

### Analysis

```bash
pip install -r requirements.txt               # numpy, scipy, matplotlib
python3 campaign/e3_stats.py --board nx       # any analysis script, from any directory
```

The analysis scripts read `results/` only and print to stdout; they write a
file only when given an output path (such as `--md`, `--tex`, `--save`, or the
`--json PATH` of `analyze_governor_sim.py` and `e5b_emcfree.py`; elsewhere
`--json` prints JSON to stdout). The table printers (`govdeploy_stats.py`,
`solo_stats.py`, `nano_vit_grid.py`, `qwengov_2k.py`, `fresh_series.py`,
`e3_table.py`, which runs `e3_stats.py` for both boards) and the number
printers (`emc_axis_stats.py`, `tenancy_stats.py`, `tails_mech_stats.py`,
`deployment_stats.py`, `profile_dispersion.py`) write nothing at all. The
first four number printers print one block per article item, or only the
blocks chosen with `--section NAME`; all five take `--results DIR`. The
figure scripts write PDF and PNG files only to the paths given: `make_figs.py`
into the folder `--out DIR`, `make_fig_govdeploy_rev.py` to `--out FILE` (plus
a PNG in `--png DIR` if given); both require `--out` and without it stop with
a usage error, writing nothing. Paths given to `--results`, `--root` or
an output option are taken relative to the current directory. The scripts need
Python 3 with numpy and scipy (the table and number printers need numpy only);
matplotlib is needed only for the figures. Tested with the L4T system Python
(3.10.12, numpy 1.21.5, scipy 1.8.0, matplotlib 3.5.1).

Without scipy the Han-model fits of `han_fit.py` lose their scipy polish and
some printed values change in the last digits. `e5_han_baseline.py` then prints
every fit's coefficients slightly differently (for MobileNetV2, H1 A = 1867.0,
B = 5243.5 instead of 1867.9, 5242.4; H1c α = 1.208, C = 1898.0 instead of
1.209, 1898.5; H2 C = 1920.2 instead of 1919.9), the ViT-Small H1 held-out
maximum error as 16.9 % instead of 17.0 % (Supp. Table S5), and in its
L2-resident GEMM section H2's T(2133) − T(3199) as +0.38 ms instead of
+0.21 ms and its 23.5 ms pick as 816 instead of 714 MHz; `e5b_contrast.py`
stops at its check of the 44–61 % error reduction stated for Supp. Table S7
(it computes 45–61 % without scipy); `e1b_stats.py` falls back to tabulated
t quantiles, which move one steady-window interval bound by 0.01 mJ. The NX
picks and admission counts are unchanged, and
`han_fit.py --loco-table --key compute_us`, `selector_cmp.py` and
`e5b_emcfree.py` print the same values (the `--json` file of the latter
differs only beyond the printed digits). `energy_pairs_ci.py` then computes its
t quantiles from the exact finite-series form of the t distribution and prints
the same values; only the `scipy=` flag of its first line changes.

### Board campaigns

- **Boards.** Orin Nano Super or Orin NX under L4T R36, set to MAXN_SUPER
  beforehand (only `run_15w_trace.sh` and `run_25w_replication.sh` change the
  power mode, using the Nano's mode ids, and restore MAXN_SUPER). The runners
  detect the board from the GPU devfreq table (top clock 1173 MHz: NX, victim
  core 7, co-runner core 5; 1020 MHz: Nano, core 5 and core 4) and put the INA
  logger on core 0.
- **Paths.** The runners default to the authors' layout and take overrides from
  the environment: `PY` (a Python with `onnxruntime-gpu`; default
  `/home/dk/mobile/rt-infer-bench/.venv/bin/python3`), `ONNX_DIR` (ONNX models;
  default `/home/dk/mobile/rt-infer-bench/models`), `GGUF_DIR` (GGUF models;
  default `/home/dk/mobile/models`) and `LLAMA_CPP` (llama.cpp tree, for
  `llama-bench`; default `/home/dk/mobile/llama.cpp`); each runner's header
  names the ones it uses. The repository root is taken from the script's own
  location. Example:
  `PY=$HOME/ort/bin/python3 ONNX_DIR=$HOME/models ./campaign/govrep_rev.sh`.
  Exceptions: the root-run `e4_replication.sh` needs `PY` and `MOBILENET` set,
  and `run_nx_validation.sh` needs `PY` (its `SLM_MODEL` defaults to
  `$HOME/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`);
  `campaign/banded_pick.py` and `fresh_pick.py` are kept as deployed and name
  their original checkout on their `LAB` line, which must be edited to run
  `banded_loop.sh` or `fresh_loop.sh` elsewhere (`admit_pick.py --selftest`
  rebinds that line by itself); `results/e3_nx_r33/run_r33.sh` changes into
  the original checkout path on its third line.
- **sudo.** The inference processes run as the invoking user. The
  September–October runners make every clock write with `sudo -n`, so they need
  passwordless sudo for those commands; the June–July user-mode runners call
  plain `sudo`; `run_g2.sh`, `run_estimator_break.sh`, `run_nx_validation.sh`,
  `run_15w_trace.sh`, `run_25w_replication.sh`, `e4_replication.sh` and
  `nano_c7.sh` run entirely as root, and `nx_live.sh` runs its controller as
  root.
- **Native tools.** Build `tools/slm_decode_bench` against a CUDA build of
  llama.cpp with the `g++` line in its header, `tools/membw` with
  `gcc -O2 -pthread -o membw membw.c`, and the transition probes with the lines
  in `pilot_trans/freq_probe.c` and `pilot_trans/gpu_probe.cu`. Generate the
  proxy models with `pilot_emc/make_*.py` (a Python with `onnx`). The June 2026
  burst-replication runners `part3/run_part3.sh` and `run_part3_highload.sh`
  need `stress-ng`.

### Community replication kit

[`repro/replicate.sh`](repro/replicate.sh) (about 20 min) re-measures, on your
own board, the two Nano results of Supp. §S3 that another unit can check most
directly: the 2133-vs-3199 MHz median shift of MobileNetV2 and the L2-resident
GEMM inversion at the top GPU clock. It targets the **Orin Nano** (CPU and GPU
pinned through `part2/set_domain.sh`, victim on core 5), runs as root,
downloads `mobilenetv2-12.onnx` from the ONNX Model Zoo (compare it with the
sha256 [above](#models)), builds the GEMM proxy if it is missing, and writes
`repro/replication_result.json`:

```bash
sudo PY=/path/to/python-with-onnxruntime-gpu GENPY=/path/to/python-with-onnx ./repro/replicate.sh
```

On the authors' Nano the 1,000-iteration cells of Supp. §S3 give +11.7 %
(MobileNetV2) and −9.1 % (GEMM); the 300-iteration cells of
`results/estimator_break` at GPU 1020 MHz, run like the kit's, give +12.3 % and
−10.6 %.

## Layout

```
harness/         periodic-release loop + ONNX Runtime backends (infer_bench.py, backends.py, rt_utils.py)
campaign/        board runners (*.sh), pick rules, analysis, printer and figure scripts (*.py)
docs/protocols/  protocol logs of the September-October 2026 campaigns
pilot_emc/       EMC clock control (emc_ctl.sh, bwmgr_halt) + proxy-model generators
pilot_trans/     frequency-transition / actuation-lag probes (Supp. S7)
part3/           stress-ng burst-replication runners and analysis (Supp. S9, June 2026)
part2/           set_domain.sh -- CPU/GPU pinning on the Orin Nano (June campaign, pilots, replication kit)
tools/           slm_decode_bench (llama.cpp per-token decode bench), membw (memory-bandwidth adversary)
repro/           board snapshot (Orin Nano, 2026-06-12) + community replication kit
results/         per-run data (folder glossary above; bulk June-July traces on Zenodo)
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE); the
Zenodo deposits carry the same license. The models are not redistributed:
MobileNetV2 and ResNet-50 come from the ONNX Model Zoo, YOLOv8s from
Ultralytics (AGPL-3.0, per the exported model's metadata), ViT-Small is
exported locally from timm with random weights, and the Qwen2.5 GGUF files are
covered by their model cards.

---

*Built by Daniel Kang (dk) @ Cleinsoft — real-time ML for robots.
[cleinsoft.com/dk](https://cleinsoft.com/dk).*
