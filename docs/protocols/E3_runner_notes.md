> **Public copy.** This is the public copy of the E3 runner notes kept with
> the manuscript's E3 protocol log (`E3_protocol.md` in this folder), for
> "When Memory-Clock State Changes Deadline Admission: Evidence from Jetson
> Orin". The date of writing is the one recorded in the note. It is
> self-attested: the file was not under version control until this release.
> Script line numbers refer to the versions the note was written against;
> later commits shifted some of them. Passages that quoted or cited non-public
> documents are replaced by a neutral description of what they address; the
> non-public planning document is called "the campaign plan", without its
> section numbers, and internal working references (host names, a
> user-specific path) are removed; where the note referred to the manuscript
> in preparation, it now says "the article". Nothing else is changed.

---

# E3 runner notes — how `admit_pick.py` / `admit_loop.sh` / `admit_refsweep.sh` implement `E3_protocol.md`

Written 2026-09-14 on the Orin Nano while both boards were running E1
(`govrep_rev.sh`). **Nothing in this note was measured**: the three scripts
were syntax-checked (`bash -n`, `py_compile`), dry-run (`--dry-run` prints
every command and touches neither clocks nor `results/`), and
`admit_pick.py --selftest` was run offline against the committed probes.
No benchmark, co-runner or clock change was executed.

Files (all under the repository root):

| file | role |
|---|---|
| `campaign/admit_pick.py` | pure decision function + CLI: probe CSV + two solo cells + D + G → p50, max, M_g, per-f slack, f_pt (h = 0), f_b (h = 400 µs); `--score` re-scores a contract CSV; `--selftest` reproduces the committed picks |
| `campaign/admit_loop.sh` | one condition, K independent trials, protocol §4 steps 1–7, board-agnostic, resumable |
| `campaign/admit_refsweep.sh` | protocol §5 reference sweep: one 1000-cycle contract at every f ∈ G under the same co-runner; logs f_oracle |

## 1. Self-test result (offline, committed data)

`python3 campaign/admit_pick.py --selftest` (also with the venv interpreter
that runs on-device) → **18 probes, 0 mismatches, PASS**:

* `results/banded_loop/probe_t1..10.csv`: f_b = 918 on every trial, equal to
  the `g918` in the committed `dep_t<t>_g918.json` file names, and
  `"{Mg:.0f} {pick}"` byte-identical to `campaign/banded_pick.py`'s stdout
  (subprocess). Point picks f_pt on the same probes: 816 on t1, t4, t8, t10
  (band active on 4/10), 918 otherwise.
* `results/fresh_loop/vprobe_t1..8.csv`: (M_g, f_pt) equal to the committed
  `results/fresh_loop/decisions.csv` vision rows and to `fresh_pick.py vision`
  stdout (816 on t3, t6; 918 otherwise).
* Two-cell refit reproduced: NX MobileNetV2 cells 306/1173 at EMC 2133
  → k = 3056.6092 µs·GHz, b = 2134.4331 µs (the campaign plan quotes
  3056.6 / 2134.4).

Passing the two endpoint files explicitly (what `admit_loop.sh` does) gives
the same k, b as `banded_pick.py`'s 10-file glob, because the refit uses only
min(gpu) and max(gpu) of whatever set it is given.

## 2. Protocol §1 (frozen elements) → code

| §1 element | value | where it lives now | source it copies |
|---|---|---|---|
| admission rule | lowest f with feasibility | `admit_pick.py` `evaluate()` lines 88–98 | `banded_pick.py:20-23`, `fresh_pick.py:17-18` |
| r | 1.035 | `admit_pick.py:42` `R_MULT` (CLI `--R`, never passed by the loop) | `banded_pick.py:18` |
| T̂(f) | k/f + b, endpoint refit at EMC 2133 | `fit_2cell()` lines 71–76; cells `CELLS` lines 47–55, passed explicitly by `admit_loop.sh` lines 46/49 (`--cells gpu306 gpu{1173,1020}`) | `banded_pick.py:14-17` |
| M_g | max − p50 of the probe | line 83 | `banded_pick.py:13` |
| clock scaling | M_g·f_probe/f | lines 90–92 | `banded_pick.py:21` |
| h | 400 µs, fixed | `admit_pick.py:43` `H_US`; `admit_loop.sh:30` `H_US=400` → `--h-us 400` | `banded_pick.py:10` |
| probe | 200 cycles at f_probe, warm-up 60, co-runner 20 s earlier | `admit_loop.sh:123-127` (`--iters 200 --warmup 60`), `co_start` lines 94–98 (`sleep 20`) | `banded_loop.sh:24-31` |
| contract | 1000 cycles at admitted f, same co-runner | `contract()` lines 99–105 (`--iters 1000 --warmup 60`) | `banded_loop.sh:33-36` |
| τ | 2 % | `admit_refsweep.sh:20` `TAU=0.02` (f_oracle only; false-admission counting is offline) | paper §III |
| EMC | 2133 locked via BPMP | `admit_loop.sh:110-112` lock + `actual=2133000000` check; `emc_check` line 89 re-verifies per trial | `govrep_rev.sh:37-38`, `emc_ctl.sh` |
| γ | 0 | no horizon term in `evaluate()` | paper §VI |
| G | NX 510…1173 / Nano 306…1020 | `admit_pick.py:44-45` and `admit_loop.sh:45-49`, passed as `--grid` so both agree | `fresh_pick.py:17`; Nano devfreq table |
| f_probe | NX 1173 / Nano 1020 | `admit_pick.py:46`, `admit_loop.sh:45-49`, asserted equal to the devfreq maximum (line 51) | protocol §1 |

`admit_pick.py --brief` prints `Mg_us f_pt f_b` with `0` for INFEASIBLE
(the `banded_pick.py` convention the shell tests against); the JSON written
by `--out` carries the strings `"INFEASIBLE"`, `band_active`, and for every
f ∈ G: `T_hat_us`, `s_us` (protocol slack), `mstar_us`, `margin_mg_us`
(= m*(f) − M_g, what `banded_pick.py` thresholds), `h_in_s_us`, `ok_h0`,
`ok_hb`.

## 3. Protocol §4 (per-trial procedure) → `admit_loop.sh`

Session set-up (once, before the trial loop): lines 108–112 —
`sched_rt_runtime_us=-1`, all CPU governors `performance`, EMC lock with
the `actual=2133000000` grep (verbatim `govrep_rev.sh:35-38`); line 113
writes a `session` line with host, commit, every frozen constant and the
cell files to `session.log`. Restore trap lines 76–82 = `govrep_rev.sh:27-31`
plus the co-runner/INA kill from `fresh_loop.sh:15`.

| §4 step | `admit_loop.sh` | notes |
|---|---|---|
| 1 thermal gate ≤ 55 °C; EMC 2133; GPU at f_probe | 120–121: `tgate` (lines 86–88, verbatim `govrep_rev.sh:33-34` + timeout WARN), `emc_check` (89–92), `setclk $FPROBE; sleep 0.4` | gate temperature logged as `tgate=` in the trial line |
| 2 start co-runner on its CPU at rate R, wait 20 s | 123 → `co_start` 94–98: `infer_bench --model $CO_MODEL --hz $CO_HZ --iters 100000 --warmup 30 --cpu $CCPU --label co_$CO --out /tmp/co_e3_<cond>_t<k>` & ; `sleep 20` | verbatim `fresh_loop.sh:26-28` with model/rate/core parametrised |
| 3 200-cycle probe at f_probe; p50, max, M_g | 125–128 probe → `t<k>.probe.{json,csv}`; statistics computed in step 4 | `--label e3_<board>_<cond>_probe_t<k>`; PROBE FAIL → co-runner stopped, trial logged, next trial |
| 4 f_pt (h = 0) and f_b (h = 400) from the same probe; s(f) for every f | 133–136: one `admit_pick.py` call writes `t<k>.pick.json` and returns `Mg f_pt f_b` | both picks come from one evaluation of one probe |
| 5 banded contract at f_b, 1000 cycles, m_b vs D | 140: `contract $FB t<k>.contract_b`; `m_b` = `admit_pick.py --score` (response_us > D) | skipped if f_b = INFEASIBLE (declined trial) |
| 6 if f_pt ≠ f_b, point contract at f_pt, m_pt | 142: runs iff `FPT != FB` and f_pt feasible | includes the case f_b INFEASIBLE, f_pt feasible |
| 7 stop co-runner; restore; log to `results/e3_<board>/<cond>/t<k>.…` | 144–147: `co_stop` (kill, wait, `sleep 2` = `banded_loop.sh:37`), `t<k>.done`, `session.log` line `t<k>: Mg= f_pt= f_b= m_b= m_pt= tgate= tend= <ISO time>`; clocks restored by the EXIT trap after the last trial | |
| "independent launches … nothing reused" | 117–118: a trial is skipped only if `t<k>.contract_b.json` or `t<k>.done` exists; otherwise every partial file of that index is deleted and the trial restarts with a fresh co-runner and fresh probe | |

## 4. Protocol §5 reference sweep → `admit_refsweep.sh`

Lines 90–91 list the cells still missing (`ref_g<f>.json` absent) and log
the session line; 93–95 gate once, verify EMC, pin f_probe, launch **one**
co-runner (same model/rate/core as the loop) and wait 20 s; 96–105 for each
missing f ascending: `setclk f; sleep 0.4`, 1000-cycle contract
`--label e3_<board>_<cond>_ref_g<f> --out ref_g<f>` with `--deadline-ms D`,
log `ref g<f>: m=<miss>`; 106 kill the co-runner; 110–112 compute and log
`f_oracle` = lowest f ∈ G with m ≤ 0.02 (`INFEASIBLE` if none).

## 5. Choices the protocol does not fix (and what I chose)

1. **Order of "pin f_probe" and "start co-runner".** §4 lists pin (step 1)
   before launch (step 2); the committed loops do the opposite
   (`banded_loop.sh:24-28`: launch, sleep 20, then `setclk 1173`), so their
   co-runner warmed up at whatever clock the previous trial left. I follow
   the protocol text: pin first, then launch. The probe itself runs at
   f_probe in both orders.
2. **Board detection** uses the devfreq `available_frequencies` maximum, not
   `max_freq` as `govrep_rev.sh:18` does. While writing this, `max_freq` on
   the Nano read `408000000` because E1 had the GPU pinned — `govrep_rev.sh`'s
   idiom would have classified the board wrongly and its restore would have
   left `max_freq` at 408 MHz. Everything else in the detection block
   (`≥ 1100 → nx`, cpu 7/5, restore to `${GMAX}000000`) is the
   `govrep_rev.sh` idiom.
3. **Thermal gate timeout.** `govrep_rev.sh:34` polls 60 × 20 s and then
   silently proceeds; I keep the same loop but log a `WARN thermal gate
   timed out` line and always record the gate temperature per trial, so a
   hot session is visible in `session.log`. Gate is per trial (protocol:
   "separated by the thermal gate"); the reference sweep gates once before
   its single co-runner launch.
4. **EMC verification per trial.** The committed loops lock once per
   session. I lock once (with the `actual=` check) and additionally read the
   BPMP rate before every trial (`emc_check`); if it is not 2133000000 the
   script re-locks and logs it. This is a read plus a conditional re-lock,
   not a protocol change.
5. **`--deadline-ms D` on contracts** (not on the probe). The committed
   contracts did not pass it (their JSON `deadline_us` is the period and every
   miss rate was re-scored offline from `response_us > D`). Passing it makes
   `contract_*.json` self-describing (`deadline_us = D`,
   `deadline_miss_rate = m`) and does not touch the timed loop
   (`infer_bench.py:61-63` uses it only for the miss count). The logged
   `m_b`/`m_pt` are nevertheless computed offline by `admit_pick.py --score`
   exactly as the Table-II convention, so the two numbers can be
   cross-checked.
6. **INA power logging is off by default.** `fresh_loop.sh` and
   `banded_loop.sh` log no power; `--ina` adds `taskset -c 0 ina_log.py
   <tag>.power.csv 120` around each contract (the `govrep_rev.sh:46` /
   `c3_tenancy.sh:53` idiom) if per-contract energy is wanted.
7. **Co-runner trace is discarded** (`--out /tmp/co_e3_…`, process killed
   before its 100000 iterations, so no JSON is ever written) — exactly as
   `fresh_loop.sh`. Consequence: the co-runner's own miss rate / GPU duty
   claimed in protocol §3 is not recorded by E3. (The campaign plan's suggestion
   to check `/tmp/co_admit_*.json` cannot work for the same reason.) Not
   changed, to keep the tenant identical to the calibration runs; if duty
   evidence is wanted, a `tegrastats` side-log is the non-invasive option.
8. **Declined trials** (f_b = INFEASIBLE): no banded contract, `m_b = -`,
   `.done` written so the trial is not re-run; the point contract still runs
   if f_pt is feasible (literal reading of step 6). `.done` is needed because
   the resume rule "skip if contract_b.json exists" alone would re-run
   declined trials forever.
9. **Condition ids.** §3 lists "YOLOv8s / ResNet-50 | 5 / 10" without
   assigning N1–N4, A1–A4. I enumerate co-runner-major, rate-minor:
   N1 = YOLOv8s@5, N2 = YOLOv8s@10, N3 = ResNet-50@5, N4 = ResNet-50@10
   (same for N5–N8 with the ViT primary and A1–A4 on the Nano).
10. **Output naming** follows the protocol (`t<k>.{probe,contract_b,contract_pt}.{json,csv}`)
    rather than the campaign plan's `probe_t<t>` / `dep_t<t>_g<pick>` draft; labels
    are `e3_<board>_<cond>_{probe,contract_b,contract_pt}_t<k>[_g<f>]`.
    `analyze_joint.py`-style globs will not match these — E3 gets its own
    analyser.
11. **Resume/consistency guards** not in any committed script: `condition.txt`
    (board, primary, rates, D, f_probe, G, h, R, cell files) is written on
    first start and must match on resume; the loop refuses to start while
    another `harness.infer_bench` is running (`--force` overrides) — this is
    what protects the E1 session running right now.
12. **Reference sweep**: one co-runner for the whole sweep (§5 "same
    co-runner"; §7 budget "8 clocks × 40 s" implies no per-clock relaunch),
    ascending f (the `c3_tenancy.sh` / `joint_gov.sh` order). A resumed sweep
    launches a fresh co-runner for the remaining cells; the session line lists
    which cells were run under which launch.
13. **Two cells passed explicitly** (`gpu306` and `gpu1173`/`gpu1020` JSONs)
    instead of the 10-file glob: identical k, b (selftest), and it removes the
    NX-only glob path from the code.
14. **`--h-us`, `--R`, `--grid`, `--fprobe`, `--mg-us` are CLI-exposed** in
    `admit_pick.py` for the selftest and for planning; `admit_loop.sh` always
    passes the frozen values. `--mg-us` (evaluate the rule for a hypothetical
    M_g without a probe) is a planning aid for §3's D_ViT choice only.
15. `chown -R` in the restore trap dropped (runs are user-mode, as in
    `govrep_rev.sh`). `sleep 0.4` after every clock write, `sleep 2` after
    the co-runner kill, `--warmup 30` for the co-runner: unchanged.
16. In `--dry-run` output the co-runner command is echoed slightly after the
    probe command (background subshell); the real order is co-runner first.

## 6. Estimated wall-clock (from the campaign plan's measured 62 s/trial and the harness arithmetic)

Per MobileNetV2 trial: gate (0 s if cool) + 0.4 + 20 (co warm-up) + ≈ 11
(probe: ≈ 3 s session start + 260/33 Hz) + ≈ 35 (contract: 3 + 1060/33) + 2
≈ **70 s**; +≈ 35 s when the band is active (step 6). Per ViT-Small trial
(20 Hz): 20 + ≈ 16 + ≈ 56 + 2 ≈ **95 s**; +≈ 56 s when active. Band-active
fraction on the calibration probes was 4/10 (banded_loop) and 2/8
(fresh_loop); I budget 1/3.

| block | trials | base | +step 6 (1/3) | ≈ total |
|---|---|---|---|---|
| NX C0 | 10 × 70 s | 12 min | +2 | 14 min |
| NX N1–N4 | 80 × 70 s | 93 min | +16 | 1.8 h |
| NX N5–N8 | 80 × 95 s | 127 min | +25 | 2.5 h |
| NX reference sweeps | 5 × (20 + 8×36 s) + 4 × (20 + 8×57 s) | 26 + 32 min | | 1.0 h |
| **NX total** | | | | **≈ 5.3 h** + thermal-gate waits |
| Nano A1–A4 | 40 × 70 s | 47 min | +8 | 55 min |
| Nano A5–A6 | 20 × 95 s | 32 min | +6 | 38 min |
| Nano reference sweeps | 4 × 5 min + 2 × 8 min | | | 36 min |
| **Nano total** | | | | **≈ 2.2 h** + gate waits |

Thermal-gate waits are the unknown: the gate runs before every trial, 2 s
after a 35–90 s co-tenanted contract. If the board sits above 55 °C at idle
(the Nano read 52–54 °C during E1) every trial pays 20 s multiples.

## 7. Exact command lines (protocol §3; run from the repository root)

D_ViT (NX) and D_ViT,Nano are **not yet in §8** — the placeholders below must
be replaced by the values appended there before N5–N8 / A5–A6 start.

NX:

```bash
./campaign/admit_loop.sh --primary mobilenet --co vit      --co-hz 10 --D 12 --trials 10 --cond C0
./campaign/admit_loop.sh --primary mobilenet --co yolov8s  --co-hz 5  --D 12 --trials 20 --cond N1
./campaign/admit_loop.sh --primary mobilenet --co yolov8s  --co-hz 10 --D 12 --trials 20 --cond N2
./campaign/admit_loop.sh --primary mobilenet --co resnet50 --co-hz 5  --D 12 --trials 20 --cond N3
./campaign/admit_loop.sh --primary mobilenet --co resnet50 --co-hz 10 --D 12 --trials 20 --cond N4
./campaign/admit_loop.sh --primary vit --co yolov8s  --co-hz 5  --D <D_ViT> --trials 20 --cond N5
./campaign/admit_loop.sh --primary vit --co yolov8s  --co-hz 10 --D <D_ViT> --trials 20 --cond N6
./campaign/admit_loop.sh --primary vit --co resnet50 --co-hz 5  --D <D_ViT> --trials 20 --cond N7
./campaign/admit_loop.sh --primary vit --co resnet50 --co-hz 10 --D <D_ViT> --trials 20 --cond N8
# reference sweeps (same arguments, no --trials)
./campaign/admit_refsweep.sh --primary mobilenet --co vit      --co-hz 10 --D 12 --cond C0
./campaign/admit_refsweep.sh --primary mobilenet --co yolov8s  --co-hz 5  --D 12 --cond N1
./campaign/admit_refsweep.sh --primary mobilenet --co yolov8s  --co-hz 10 --D 12 --cond N2
./campaign/admit_refsweep.sh --primary mobilenet --co resnet50 --co-hz 5  --D 12 --cond N3
./campaign/admit_refsweep.sh --primary mobilenet --co resnet50 --co-hz 10 --D 12 --cond N4
./campaign/admit_refsweep.sh --primary vit --co yolov8s  --co-hz 5  --D <D_ViT> --cond N5
./campaign/admit_refsweep.sh --primary vit --co yolov8s  --co-hz 10 --D <D_ViT> --cond N6
./campaign/admit_refsweep.sh --primary vit --co resnet50 --co-hz 5  --D <D_ViT> --cond N7
./campaign/admit_refsweep.sh --primary vit --co resnet50 --co-hz 10 --D <D_ViT> --cond N8
```

Nano:

```bash
./campaign/admit_loop.sh --primary mobilenet --co yolov8s  --co-hz 5  --D 12 --trials 10 --cond A1
./campaign/admit_loop.sh --primary mobilenet --co yolov8s  --co-hz 10 --D 12 --trials 10 --cond A2
./campaign/admit_loop.sh --primary mobilenet --co resnet50 --co-hz 5  --D 12 --trials 10 --cond A3
./campaign/admit_loop.sh --primary mobilenet --co resnet50 --co-hz 10 --D 12 --trials 10 --cond A4
./campaign/admit_loop.sh --primary vit --co yolov8s --co-hz 5  --D <D_ViT,Nano> --trials 10 --cond A5
./campaign/admit_loop.sh --primary vit --co yolov8s --co-hz 10 --D <D_ViT,Nano> --trials 10 --cond A6
./campaign/admit_refsweep.sh --primary mobilenet --co yolov8s  --co-hz 5  --D 12 --cond A1
./campaign/admit_refsweep.sh --primary mobilenet --co yolov8s  --co-hz 10 --D 12 --cond A2
./campaign/admit_refsweep.sh --primary mobilenet --co resnet50 --co-hz 5  --D 12 --cond A3
./campaign/admit_refsweep.sh --primary mobilenet --co resnet50 --co-hz 10 --D 12 --cond A4
./campaign/admit_refsweep.sh --primary vit --co yolov8s --co-hz 5  --D <D_ViT,Nano> --cond A5
./campaign/admit_refsweep.sh --primary vit --co yolov8s --co-hz 10 --D <D_ViT,Nano> --cond A6
```

Overnight session pattern (as E1 was launched): chain the lines of one board
in a single `nohup bash -c '…' > results/e3_<board>/run.log 2>&1 &`; every
script is resumable, so a crashed chain is simply re-issued. Preflight per
board, in this order and only after E1 has printed `GOVREP_REV DONE`:
`pgrep -af infer_bench` empty → `--dry-run` of one condition →
`--trials 1 --cond SMOKE` (then `rm -r results/e3_<board>/SMOKE`) →
the chain. ResNet-50 has never been executed on the Nano; the SMOKE trial
for A3 doubles as its CUDA-EP load check.

Filling D_ViT before N5/A5 (planning aid, no measurement):
`python3 campaign/admit_pick.py --mg-us <assumed M_g> --D-ms <candidate D> --board nx --primary vit`
prints f_pt / f_b and the per-f slack under the frozen rule; see §8 item 3
for why an assumed M_g is unavoidable.

## 8. Inconsistencies found between the protocol and what the pipeline actually does

1. **The band is not "not clock-scaled" in the protocol's own notation.**
   `banded_pick.py:21-22` tests `Mg <= mstar - H` with
   `mstar = (D − T̂·r)/(f_probe/f)`, i.e. h is subtracted from the
   *probe-scale* margin. In the §1 slack notation
   s(f) = D − [T̂(f)·r + M_g·f_probe/f] this is **s(f) ≥ h·f_probe/f**, not
   s(f) ≥ h: the band is worth 400 µs at f_probe, 511 µs at 918 (NX),
   920 µs at 510, 1333 µs at 306 (Nano). `admit_pick.py` reproduces the code
   (that is what the committed validation used) and reports both `s_us` and
   `margin_mg_us` per f. §1's rule row and the "not clock-scaled" remark
   should be re-worded before E3 data exist ("h applies to the probe-scale
   margin m*(f) − M_g, equivalently s(f) ≥ h·f_probe/f").
2. **§4 step order vs the committed loops** (choice 1 above): the submitted
   pipeline launched the co-runner *before* pinning f_probe. The runner
   follows §4; note it in §8 of the protocol if the committed C0 is to be
   called "identical".
3. **§3 "computed from the existing solo estimator sweeps only" cannot place
   a knife-edge by itself.** With M_g = 0 the two-cell model admits 510 MHz
   on NX MNv2 at D = 12 ms (T̂·r = 8.4 ms) — the knife-edge at 816–918 only
   appears once a co-runner margin ≈ 4.3 ms (the calibration probes' M_g) is
   assumed; the campaign plan's Nano statement "knife-edge at 714–816" also used
   M_g ≈ 4.4 ms. §8 must therefore record the assumed M_g together with
   D_ViT and D_ViT,Nano, or the choice is not reproducible.
4. **The calibration condition already violates §5's primary endpoint as
   worded.** Scoring the committed `results/banded_loop/dep_t*_g918.csv`
   per trial at τ = 2 % gives 1.8 / **2.9** / **2.2** / 0.8 / 0.8 / 1.7 / 0.0 /
   0.1 / 0.0 / 1.6 % → 2 of 10 banded admissions exceed τ. Under §6 the
   positive control C0 would be "not supported" unless the paper's τ was
   applied to a pooled/mean miss rate. Decide (and write into §8) whether the
   endpoint is per-trial m_b > τ before C0 is re-run; the runner records
   per-trial m_b either way.
5. **§7 counts 8 NX reference sweeps; §3 has 9 NX conditions** (C0 + N1–N8).
   The command list above includes a C0 sweep; the existing
   `results/joint_gov/dep_co_g*_r*` (ViT@10 co, 3 reps at 510–1122, no 1173)
   is a near-substitute but not "once per condition, all f ∈ G".
6. **Fit-cell protocols differ across boards/primaries** although §1 calls
   them all "the estimator sweep": NX MNv2 cells are 300 iters at 8 Hz
   (`run_nx_validation.sh` P2), NX ViT 800 at 20 Hz (`vit_deploy.sh`), Nano
   MNv2 800 at **50 Hz** and Nano ViT 800 at **10 Hz** (`nano_wave2a.sh`),
   i.e. not at the E3 primary rates 33/20 Hz. The campaign plan estimates the
   MNv2 p50 offset at ≈ 0.1 ms. Frozen as-is per §1; worth one sentence in
   the article.
7. **`govrep_rev.sh` board detection via `max_freq`** (item 2 in §5) is wrong
   whenever a clock is pinned at start-up; it happened to be right for E1
   because both boards started unpinned. Patch it to
   `available_frequencies` after the current E1 sessions finish (do not
   touch the running copies).
8. **Campaign-plan draft** (`/tmp/co_admit_$$.json` duty check, `dep_t<t>_g<pick>`
   naming, `D_US=24000` default for ViT) is superseded by the protocol's
   file names and by §8's pending D_ViT; the draft's co-runner JSON never
   exists (choice 7).
9. **Co-runner ViT `--warmup 30` and `sleep 20`** are in the scripts, not in
   §1's probe row (which says only "co-runner started 20 s earlier"); kept
   verbatim.
10. **§2 says C0 uses the "same workload against the same co-runner"** but
    the E3 run of C0 will differ from the submitted 10 trials in the three
    documented ways above (pin-before-launch, per-trial thermal gate,
    `--deadline-ms` in the contract JSON). None touches the timed loop.
