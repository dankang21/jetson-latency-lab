> **Public copy.** This is the public copy of the E3 protocol log kept with
> the manuscript "When Memory-Clock State Changes Deadline Admission: Evidence
> from Jetson Orin". The freeze, pre-statement and amendment dates are those
> recorded in the log. They are self-attested: the file was not under version
> control until this release. Passages that quoted or cited non-public
> documents are replaced by a neutral description of what they address;
> internal working references (finding labels, names and paths of internal
> notes) are removed; where the log referred to the manuscript in preparation,
> it now says "the article". Nothing else is changed.

---

# E3 — Pre-registered prospective validation of the dispersion-banded admission guard

**Status: FROZEN 2026-09-14 (before any E3 data collection).** Repo commit at
freeze: see `git log -1` at the time this file is committed. No fresh-probe
trial under any condition listed below has been run before this document was
written. Any deviation from this protocol must be recorded in §8 with the date
and reason, and reported in the article.

Addresses: whether the dispersion band, set and validated only on the
calibration condition (§2), transfers to other primaries, co-runners,
co-runner rates and boards; how the band/headroom values were set and
whether that choice depends on the held-out data; acceptance rate and
utilization; more independent repetitions; policy applicability when the
co-runner, its rate, the workload, or the device changes.

## 1. What is frozen (taken verbatim from the submitted pipeline)

| element | value | source |
|---|---|---|
| admission rule | admit lowest f in G with s(f) = D − [T̂(f)(1+γH) r + M_g·f_probe/f] ≥ h | fresh_pick.py, banded_pick.py |
| safety multiplier r | 1.035 | fresh_pick.py:9, banded_pick.py:18 |
| latency model T̂(f) | two-cell endpoint refit (k/f + b) at the deployment EMC point from the two extreme GPU cells of the estimator sweep | banded_pick.py:14–17 |
| maximum guard M_g | max − p50 of the 200-cycle occupancy probe at f_probe, scaled by f_probe/f | banded_pick.py:13 |
| band h | 400 µs in probe-clock (M_g) space: admit lowest f with M_g + h ≤ (D − T̂(f)(1+γH)r)·f/f_probe, i.e. s(f) ≥ h·f_probe/f (effective band 511 µs at NX 918 MHz, 800 µs at Nano 510 MHz). This is what banded_pick.py deploys and what the held-out validation ran; the submitted text's "not clock-scaled" wording is corrected in the article. h = 2σ of the round-1 probe M_g on the calibration condition. | banded_pick.py:21–22 |
| probe | 200 cycles at f_probe (NX 1173 MHz; Nano 1020 MHz), warm-up 60, co-runner started 20 s earlier | fresh_loop.sh, banded_loop.sh |
| contract | 1000 periodic cycles at the admitted f, same co-runner running | fresh_loop.sh |
| QoS budget τ | 2 % | paper §III |
| EMC | locked 2133 MHz (BPMP) on both boards | emc_ctl.sh |
| horizon term | γ = 0 for vision contracts | paper §VI |
| grid G | NX {510,612,714,816,918,1020,1122,1173}; Nano {306,408,510,612,714,816,918,1020} | fresh_pick.py; Nano per devfreq available_frequencies |

**Nothing above is re-tuned per condition.** In particular h stays 400 µs on
every condition, including the Nano, where the probe dispersion has never
been measured; if the band turns out to be too narrow or too wide there,
that is a finding, not a reason to re-fit.

## 2. Calibration condition (excluded from transfer claims)

Primary MobileNetV2 (33 Hz, NX), co-runner ViT-Small at 10 Hz, D = 12 ms,
EMC 2133. This is the only condition on which h was set and on which the
submitted held-out validation (10 trials) ran. It is re-run in this campaign
only as a positive control (§4, C0).

## 3. Transfer conditions (pre-stated)

Factors: primary P ∈ {MobileNetV2, ViT-Small}; co-runner C ∈ {YOLOv8s,
ResNet-50}; co-runner rate R ∈ {5 Hz, 10 Hz}; board B ∈ {NX, Nano}.

| id | board | primary | co-runner | rate | D (ms) | trials |
|---|---|---|---|---|---|---|
| C0 (control) | NX | MNv2 33 Hz | ViT-S | 10 | 12 | 10 |
| N1–N4 | NX | MNv2 33 Hz | YOLOv8s / ResNet-50 | 5 / 10 | 12 | 20 each |
| N5–N8 | NX | ViT-S 20 Hz | YOLOv8s / ResNet-50 | 5 / 10 | D_ViT | 20 each |
| A1–A4 | Nano | MNv2 33 Hz | YOLOv8s / ResNet-50 | 5 / 10 | 13.0 | 10 each |
| A5–A6 | Nano | ViT-S 20 Hz | YOLOv8s | 5 / 10 | 37.5 | 10 each |

(N5–N8 use D_ViT = 32.5 ms.)

Deadlines are chosen **before the first trial** by one rule applied to every
condition, using only the existing solo estimator sweeps: with f_k the grid
clock three bins below f_probe (NX 918 MHz, Nano 714 MHz),
D = round_0.5ms( 2.12 × T̂_2cell(f_k) · r ), where 2.12 is the ratio the
submitted NX MobileNetV2 condition already has (D = 12 ms against
T̂(918)·r = 5.66 ms). Values (computed 2026-09-14, see §8): NX MNv2 12.0 ms;
NX ViT-S 32.5 ms; Nano MNv2 13.0 ms; Nano ViT-S 37.5 ms. The rule leaves
the same proportional room for the co-tenant margin in every condition, so
the knife-edge lands at a comparable place in each grid.

Co-runner rates: 5 Hz and 10 Hz. YOLOv8s ≈ 30 ms and ResNet-50 ≈ 12–15 ms
per inference at the top GPU clock, so 10 Hz keeps GPU duty ≤ ~55 % for
either; the ViT-S calibration co-runner ran at 10 Hz (≈13 % duty), so the
new conditions are strictly heavier tenants.

## 4. Per-trial procedure (identical to the submitted pipeline)

1. Thermal gate ≤ 55 °C; EMC locked 2133; GPU pinned to f_probe.
2. Start co-runner (pinned to its CPU, rate R). Wait 20 s.
3. Run the 200-cycle probe at f_probe; compute p50, max, M_g = max − p50.
4. Compute the **point-policy** pick f_pt (h = 0) and the **banded** pick f_b
   (h = 400 µs) from the same probe. Record both, with s(f) for every f ∈ G.
5. Deploy the banded contract: pin GPU to f_b, run 1000 cycles, record miss
   rate m_b against D.
6. **If f_pt ≠ f_b**, also deploy the point-policy contract (1000 cycles at
   f_pt), record m_pt. This measures whether the band was *needed* on that
   trial, at the cost of one extra contract.
7. Stop co-runner; restore clocks. Log everything to
   `results/e3_<board>/<cond>/t<k>.{probe,contract_b,contract_pt}.{json,csv}`.

Trials are independent launches (fresh process, fresh probe) separated by the
thermal gate; nothing from a previous trial is reused.

## 5. Pre-stated endpoints

Endpoints follow the submitted paper's separation of QoS criteria
(reference-probe feasibility / fresh-launch aggregate feasibility /
per-launch compliance), so that the calibration condition is judged by the
same yardstick as the new ones.

Primary endpoints (per condition):
- **P1 — low-clock false admission (layer 2, the band's job)**: number of
  trials whose banded pick f_b lies below the condition's aggregate-feasible
  frontier f_oracle (lowest f whose reference contract, §5 sweep, meets τ).
  This is exactly the failure the band was introduced to remove (the two
  816-MHz admissions of the submitted stress test).
- **P2 — aggregate compliance of banded admissions**: pooled miss rate over
  all banded contracts of the condition ≤ τ = 2 %, with its 95 % upper
  bound (run-level, as in the paper).
Reported with one-sided 95 % Clopper–Pearson upper bounds (the convention
used for every zero-event count in the article; 0/20 → 13.9 %, 0/10 →
25.9 %, 0/24 → 11.7 %); no pooling across conditions into a single
statement (conditions are not exchangeable).

Layer-3 statistic (reported, not a pass/fail of the band): **per-launch
violation count** — trials with m_b > τ — with min–max of m_b. The submitted
held-out set already had 2/10 such launches (2.2 %, 2.9 %) under an
aggregate of 1.19 %; the article reports this statistic for every condition
and does not relabel it as a band failure.

Secondary endpoints:
- **Band necessity**: number of trials with f_pt ≠ f_b, and among those, how
  many point-policy contracts exceeded τ (from step 6). This is the direct
  test, under new conditions, of the concern the +0.08 probe → realization correlation raises: the band is
  useful only if it removes admissions that would have failed.
- **Acceptance rate**: fraction of trials that admitted any f (vs. declined
  as infeasible). **Utilization cost**: f_b − f_oracle in grid bins, where
  f_oracle is the lowest f whose measured contract miss is ≤ τ, taken from the
  reference sweep run once per condition (all f ∈ G, one 1000-cycle contract
  each, same co-runner) — answers whether compliance comes only from
  declining or over-provisioning.
- **Probe → realization correlation**: Pearson/Spearman between probe M_g and
  contract miss per condition (the +0.08 statistic, now on 15 conditions).
- **Launch-level spread**: per-condition min–max of m_b across trials.

## 6. Pre-stated interpretation rules

- Transfer is **supported** for a condition if P1 = 0 and P2 holds, and the
  band-necessity test shows either no f_pt ≠ f_b trials (band idle) or that
  every removed point-policy admission would have failed.
- Transfer is **not supported** for a condition if P1 > 0 or P2 fails; we
  report it as such, with the trial's probe statistics, and do not widen h
  post hoc. A follow-up with a re-derived h would be labelled as a new
  calibration, not as validation.
- If the band is idle everywhere (f_pt = f_b on every trial) the honest
  statement is that the transfer conditions did not exercise the band; the
  knife-edge selection of D in §3 is designed to avoid this, but we will not
  move D after seeing results.

## 7. Budget

NX: C0 (10) + N1–N8 (160) = 170 trials × ≈ 70 s (+ step-6 extras, ≤ 30 %
more) ≈ 4–5 h plus 9 reference sweeps (8 clocks × 40 s) ≈ 50 min.
Nano: A1–A6 (60) ≈ 1.5 h plus 6 reference sweeps ≈ 35 min. One overnight
session per board. Boards run concurrently (no shared resources).

## 8. Amendments log

- 2026-09-14 — protocol frozen (no E3 trial run yet).
- 2026-09-14 (same day, before any trial) — deadline rule fixed as one
  formula (§3) instead of per-condition judgement; computed from
  `results/nx_validation/nx_est_emc2133_gpu*_{mobilenet,vit}.json` and
  `results/nano_grid/{mobilenet,vit}/est_emc2133_gpu*.json` with two-cell
  endpoints (NX 306/1173, Nano 306/1020): T̂(f_k)·r = 5.66 (NX MNv2), 15.40
  (NX ViT), 6.23 (Nano MNv2), 17.62 ms (Nano ViT) → D = 12.0 / 32.5 / 13.0 /
  37.5 ms. Nano MNv2 therefore uses 13.0 ms, not the 12 ms first pencilled in.
- 2026-09-15 (before any trial; from the runner self-test against the
  committed pipeline) — (i) §1 band row corrected: the deployed code applies
  h in probe-clock space (s(f) ≥ h·f_probe/f); the runner reproduces the
  committed picks exactly (18/18 probes). (ii) §5 endpoints restated in the
  paper's three-criteria form (P1 low-clock false admission vs the reference
  frontier; P2 aggregate compliance; per-launch violations reported as the
  layer-3 statistic) — the earlier per-trial wording would have counted the
  submitted held-out set's own 2/10 launch exceedances as band failures,
  which is not how the paper scores them. (iii) §4 pins f_probe before
  launching the co-runner (the committed loops did the reverse); the timed
  loop is unaffected. (iv) The knife-edge interpretation in §3 assumes the
  calibration M_g ≈ 4.3 ms; the deadline formula itself does not depend on
  it. (v) §7: 9 NX sweeps, not 8. No numerical parameter changed.

- 2026-09-15 — bookkeeping: 15 conditions (C0 + N1–N8 + A1–A6), not 16; zero-event bounds stated as one-sided 95 % Clopper–Pearson throughout the article (E3, E4, E1 violation counts).
- 2026-09-15 (after Nano A1–A6 trials and A1 reference sweep; disclosed as a
  post-data amendment) — (i) **Probe-validity precondition.** In A6 (ViT-S
  primary, YOLOv8s 10 Hz) 7/10 probes, and in A5 some, were indistinguishable
  from the solo cell (p50 ≈ solo, M_g 184–520 µs vs solo dispersion 246 µs):
  the co-runner was not seen by the probe (root cause not established: with
  the readiness gate the co-runner reports ready after 2 s, so a start-up
  delay alone does not explain it; killed co-runners left no output), so the guard admitted 306 MHz and every
  contract missed. This is an execution failure of step 2, not a band
  outcome. Fix: the harness now touches a ready-file after its warm-up and
  the runner waits for it (timeout 120 s, wait logged) before the probe; the
  co-runner's log is kept. A5/A6 are re-run as A5r/A6r under the fixed step
  2; the original A5/A6 are retained and reported as protocol-execution
  failures, not as band results. The NX conditions run only with the fix.
  (ii) **§6 wording.** The "supported" clause required every removed
  point-policy admission to have failed; that made the band's conservative
  removals (which cost utilization but no compliance) read as a failure of
  support. Restated before the NX conditions run: transfer is supported iff
  P1 = 0 and P2 holds; band necessity is reported as (removed admissions
  that failed, removed admissions that passed) and enters the utilization
  cost, not the verdict. A1–A3 under this wording: supported (P1 0/10, P2
  0.35 %/0.11 %/0.04 %; band removed 1/6, 1/4, 0/3 failing admissions).
- 2026-09-15 (after A5r/A6r; mechanism test, pre-stated before running) —
  **A7 / N9 (incommensurate tenant rate).** ViT-S 20 Hz primary vs YOLOv8s at
  7 Hz (period 142.9 ms, not a multiple of 50 ms), D = 37.5 ms (Nano) /
  32.5 ms (NX), 10 trials + reference sweep, readiness-verified runner.
  Prediction if (a) phase-locked interference or (b) probe blindness is the mechanism: no all-or-nothing launches
  (per-launch miss neither 0 nor ≥ 50 % with cycle-parity structure), every
  probe sees the tenant (M_g ≫ solo dispersion), lag-1 autocorrelation of
  the miss indicator near 0. Prediction for (c) fair-share stretching is unchanged (saturation
  at low clocks does not depend on phase). Verdict rules of §6 apply.
  Offline analysis (no new runs): a utilization guard
  u(f) = T̂(f)·rate_primary + T̂_tenant(f)·rate_tenant ≤ 1 evaluated on the
  A5r/A6r picks, reported as a proposed extension only.
- 2026-09-15 — **A7 result (pre-stated mechanism test met its prediction).**
  ViT-S 20 Hz vs YOLOv8s 7 Hz (incommensurate): every probe saw the tenant
  (M_g 15.1–16.5 ms in 10/10 trials vs 0.15–15.5 ms, phase-dependent, under 5/10 Hz), pick 816
  MHz in 10/10, P1 = 0/10, P2 = 0.00 % (0/10 000 cycles), per-launch
  violations 0/10, f_oracle = 714 (one bin below the pick). A5r/A6r with
  their own reference sweeps: P1 3/10 and 10/10 (A6r picks all below
  f_oracle = 816, utilization −3 bins), P2 15.1 % / 71.3 %. Conclusion
  recorded for the applicability section: the occupancy probe is valid when
  tenant interference is phase-mixing within the probe window and invalid
  (blind or partial) when the tenant's period is commensurate with the
  primary's; the band cannot repair a probe that did not observe the
  tenant. E3 Nano complete (A1–A7).
- 2026-09-15 — Nano MobileNetV2 E1 session 2 pre-stated (multi-session
  evidence for the Nano row as for the NX headline): clocks {408, 510,
  612}, 10 pairs, grid unchanged → results/govrep_rev/nano_mobilenet_s2.
- 2026-10-13 — execution failure logged: the NX conditions C0, N1–N9 run on 2026-09-16 produced probes but no contracts because campaign/admit_pick.py had not been deployed to the NX (every trial "PICK FAIL"); the reference sweeps completed and are kept. The admission loops are re-run with the full campaign directory synchronised and admit_pick --selftest passing on the NX; the 2026-09-16 probe files are discarded (nothing reused, per §4).

## Result — NX conditions (admission loops 2026-10-13 02:03–20:06 UTC = 2026-10-13 10:03 to 2026-10-14 04:07 board time +08:00; reference sweeps 2026-09-16; results/e3_nx; readiness-verified runner)
| cond | primary / tenant | P1 | P2 (pooled) | launches > τ | band removed (failed/passed) | verdict |
|---|---|---|---|---|---|---|
| C0 | MNv2 / ViT-S 10 Hz (control) | 0/10 | 0.46 % | 0/10 | 2/1 | supported |
| N1 | MNv2 / YOLOv8s 5 Hz | 6/20 (816 vs f_oracle 918; ref 816 = 3.2 %) | 0.55 % | 3/20 | 7/4 | not supported (P1) |
| N2 | MNv2 / YOLOv8s 10 Hz | 20/20 (f_oracle 1020; ref 816 = 2.1 %, 918 = 2.7 %) | 1.32 % | 6/20 | 5/2 | not supported (P1) |
| N3 | MNv2 / ResNet-50 5 Hz | 0/20 | 0.93 % | 4/20 | 4/7 | supported |
| N4 | MNv2 / ResNet-50 10 Hz | 2/20 (816 vs 918; ref 816 = 3.0 %) | 1.33 % | 8/20 | 5/3 | not supported (P1) |
| N5 | ViT-S / YOLOv8s 5 Hz | 18/20 | 33.8 % | 16/20 | idle | not supported (phase-locked; 15/20 probes ≈ solo) |
| N6 | ViT-S / YOLOv8s 10 Hz | 20/20 | 82.5 % | 17/20 | 1/0 | not supported (phase-locked; 13/20 probes ≈ solo) |
| N7 | ViT-S / ResNet-50 5 Hz | 9/20 | 7.4 % | 6/20 | 0/4 | not supported (phase-locked; 8/20 probes ≈ solo) |
| N8 | ViT-S / ResNet-50 10 Hz | 0/20 | 19.2 % | 8/20 | 0/1 | not supported (P2; all-or-nothing launches) |
| N9 | ViT-S / YOLOv8s 7 Hz (mechanism test) | 0/10 | 0.00 % | 0/10 | 0/4 | supported |
Reading (pre-stated rules applied as written): aggregate compliance (P2) held on every incommensurate-rate condition on both boards (Nano A1–A4, A7; NX C0, N1–N4, N9: 10/10); the low-clock criterion P1 failed on N1, N2, N4 (and Nano A4) where the reference frontier cell itself measures within ~1 point of τ (2.7–3.2 % for the cell one bin below the oracle; N2's next cell 2.1 %) — P1 is scored against a single 1,000-cycle reference contract per clock, so at such frontiers it is not resolved by the design; we report the verdicts as pre-stated and add this interpretation. Commensurate-period tenants (N5–N8, A5r/A6r) defeat the occupancy probe on both boards (phase-locked interference); the incommensurate mechanism test (A7, N9) is supported on both.
- Runner note: admit_refsweep.sh logged "f_oracle=INFEASIBLE" in every NX session.log while e3_stats recomputes the frontier correctly from the same CSVs (cosmetic logging bug; the scored values are the recomputed ones).

- 2026-10-23 (post-hoc sensitivity experiment, stated before running; does NOT replace any registered verdict) —
  **Rate-matched cells (r33).** The offline re-pick shows that
  with the 33 Hz endpoint cells (results/refit33/nx/emc2133_g{306,1173}_r{1,2,3}, median per-run p50:
  306 MHz 11,634.24 µs, 1173 MHz 4,384.82 µs) instead of the frozen 8 Hz cells, 27 of 90 NX MobileNetV2 banded picks
  move one bin down. To measure the consequence directly, two representative conditions are re-run fresh on the NX
  with the 33 Hz cells and everything else as frozen (rule, r = 1.035, h = 400 µs, f_probe = 1173, grid, D = 12.0 ms
  kept, not re-derived; the protocol's deadline formula would give 11.18 ms with the 33 Hz cells):
  C0r33 = MobileNetV2 33 Hz vs ViT-S 10 Hz (the calibration condition), 10 trials; N3r33 = MobileNetV2 33 Hz vs
  ResNet-50 5 Hz (the one condition supported under the original-timing verdict), 20 trials. Each condition first gets
  a same-session reference sweep (one 1,000-cycle contract per grid clock), so its frontier is contemporaneous with
  its contracts. Endpoints and verdict rule as registered (P1 vs this session's frontier, P2 pooled banded miss with
  run-level bootstrap UB, launches > τ). Output results/e3_nx_r33/{C0r33,N3r33}. Reported as a post-hoc sensitivity
  experiment next to, never instead of, the registered C0 and N3 rows.
