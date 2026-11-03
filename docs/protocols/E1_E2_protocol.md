> **Public copy.** This is the public copy of the E1/E2 replication protocol
> log, with its E5, E5-B and E5-B-live sections, kept with the manuscript
> "When Memory-Clock State Changes Deadline Admission: Evidence from Jetson
> Orin". The freeze, pre-statement and amendment dates are those recorded in
> the log. They are self-attested: the file was not under version control
> until this release. Where the E5-B and E5-B-live sections call
> `campaign/e5b_emcfree.py` or its scoring "committed", the script was in fact
> kept in the author's working tree outside version control; it first entered
> version control in this release (2026-11-03), so the times the log gives for
> its writing and its 2026-10-28 scoring are, like all of the log's time
> stamps, self-attested. Passages that quoted or cited non-public documents
> are replaced by a neutral description of what they address; internal working
> references (internal process notes, names and paths of internal notes and
> outputs, host names) are removed, and the non-public planning document is
> called "the campaign plan". Nothing else is changed.

---

# E1/E2 — Balanced interleaved replication of Table II over a pre-stated deadline grid

Frozen 2026-09-14 before analysis of results/govrep_rev/ (stated in the
campaign plan before the campaign finished).

- Rule for BOTH arms: the deployed two-cell endpoint refit T̂(f)=k/f+b with
  r=1.035; pick = lowest f in G with T̂(f)·r ≤ D; if none, the policy
  declares the deadline infeasible and the deployment falls back to the
  maximum clock (reported as such, not as a pick).
- Blind arm: two-cell fit at EMC 3199 (profiling point); aware arm: two-cell
  fit at EMC 2133 (deployment point). Deployment always at EMC 2133.
- Cells: NX 306/1173; Nano 306/1020 endpoints from the committed solo sweeps.
- Deadline grid (E2): NX MobileNetV2 D ∈ {4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6,
  5.8, 6.0} ms; Nano MobileNetV2 D ∈ {8.5, 9.0, 9.5, 10.0} ms. The submitted
  cells (NX 4.6/4.8/5.0, Nano 9.0) are members of the grid.
- Data: results/govrep_rev/{nx,nano}/g<f>_p<1..10>.csv (1000 cycles each),
  clocks NX {816,918,1020,1122,1173}, Nano {408,510,612}; every clock run
  once per pair index inside one thermally gated session (≤55 °C), EMC 2133.
- Reported per (arm, D): the picked clock, per-run miss % (10 values),
  median and min–max, and the number of runs violating the 2 % budget
  (k/10). Headline statements are made on violation counts and the
  miss-vs-D curves, not on a single mean.

- Nano ViT-Small (20 Hz) addendum, frozen 2026-09-15 before the campaign: clocks {510,612,714,816,918,1020}, 10 pairs; deadline grid = T̂_aware(f)·r rounded UP to 0.5 ms for f ∈ {612,714,816,918,1020} plus each +0.5 ms → D ∈ {14.0, 14.5, 15.0, 15.5, 16.5, 17.0, 18.0, 18.5, 20.0, 20.5} ms.

## E1b — breadth deployment cells (NX: ViT-Small 20 Hz, YOLOv8s 10 Hz, ResNet-50 20 Hz), frozen 2026-09-15 before running
- Same interleaved design (10 pairs, one thermally gated session, EMC 2133, INA log) over every clock the two-cell rule can pick across each model's deadline grid, so Table S2/S3 (tab:vitgov, tab:breadthgov) are regenerated under the deployed rule with balanced arms.
- Deadline grid per model, by the same formula used for Nano ViT: D = T̂_aware(f)·r rounded UP to 0.5 ms for f ∈ {714, 816, 918, 1020, 1122} plus each +0.5 ms (values computed from results/nx_validation two-cell 306/1173 fits and appended below before the runs start).
- Clocks run: the union of blind and aware two-cell picks over the grid (typically {816, 918, 1020, 1122, 1173}).
- Energy: ΔE per grid point from interleaved pairs, paired 95 % CI (E6 method); relative ΔE against the blind pick's energy.

- NX ViT-S 20 Hz: two-cell k=8283856 b=5858; D grid = {14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 17.5, 18.5, 19.0} ms
- NX YOLOv8s 10 Hz: two-cell k=17627925 b=12354; D grid = {29.5, 30.0, 31.0, 31.5, 33.0, 33.5, 35.5, 36.0, 38.5, 39.0} ms
- NX ResNet-50 20 Hz: two-cell k=9130786 b=4941; D grid = {14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 17.5, 18.5, 19.0} ms

## Amendment 2026-09-15 (after the first E1 NX session, before any refit) — profiling rate
The two-cell fit cells in results/nx_validation were swept at 8 Hz (300
iters) while every deployment loop releases at 33 Hz; the fit cells sit
~400 µs above the deployment p50 at every clock (918: 5.36 vs 4.95 ms). The
release rate is therefore a knob that moved between profiling and
deployment (E7 list). Fix, pre-stated: re-profile the four two-cell
endpoints (GPU 306 and 1173 at EMC 2133 and 3199) at the deployment rate
(33 Hz, 1000 cycles, 3 repeats each, same user-mode loop) and refit both
arms from those cells; re-evaluate the SAME interleaved E1 data under the
rate-matched fits, reporting both the original-fit and the rate-matched
results side by side. The E2 grid is unchanged. Prediction: both arms
pick lower clocks; the blind arm's EMC error is unchanged in sign.

## Amendment 2026-09-15 (after the rate-matched re-evaluation of session 1)
- Result of the pre-stated refit (results/refit33/nx, 33 Hz, 3 repeats): endpoint cells are 283–489 µs faster than the 8 Hz cells. Under the rate-matched two-cell fits the aware arm picks the lowest feasible clock at every grid deadline with 0/10 violations (declining 4.4 ms); the blind arm violates the budget at 7 of the 9 grid deadlines (100 % at 4.4/4.6/4.8/5.2; 6/10 and 5/10 at 5.0/5.4; 1/10 at 5.6). Under the 8 Hz fits both arms are inflated by ≈400 µs and the blind arm passes at ≥4.6 ms in this session. Both evaluations are reported.
- Session 1 lacks the 714 MHz clock the rate-matched blind fit picks at 5.8/6.0 ms. **E1 session 2** (pre-stated, different day/session): clocks {612, 714, 816, 918, 1020, 1122, 1173}, 10 pairs, same protocol → results/govrep_rev/nx_mobilenet/. A third session is planned to give three independent sessions for the multi-session violation counts.
- Fit-cell rates found: NX ViT-S 20 Hz (matches deployment), YOLOv8s 5 Hz (deployment 10), ResNet-50 10 Hz (deployment 20), Nano MNv2 50 Hz (matches), Nano ViT 10 Hz (deployment 20). Rate-matched endpoint refits are queued for YOLOv8s/ResNet-50 (and ViT as a same-rate replicate) on the NX (results/refit33/nx_<model>) and will be run for Nano ViT; E1b/E1-Nano data are evaluated under both the original and the rate-matched fits.
- E3 keeps the submitted (8 Hz-cell) two-cell model by design — it is the frozen, validated rule; the rate-matched model's picks are computed offline on the same probes for comparison only.
- 2026-09-15 — deviation logged: the first Nano MobileNetV2 "session 2" ran at 33 Hz (runner default) instead of the pre-stated 50 Hz; it is kept as results/govrep_rev/nano_mobilenet_33hz_deviation (numbers happened to match session 1: 408 MHz p50 9.09–9.11 ms, 2/10 violations at 9.5 ms) but is excluded from pooling. Session 2 re-run at 50 Hz → results/govrep_rev/nano_mobilenet_s2.

## E5 — memory-aware controller baseline (pre-stated 2026-09-16, before scoring)
- Baseline estimator: the Han et al. monotone power-law T = A·f_gpu^-α + B·f_emc^-β + C (campaign/han_fit.py), fitted per model on the rate-matched sweeps results/nx_validation_r33 (three EMC levels × 10 GPU clocks, deployment release rates).
- Two protocols: (H1, realistic) fit on the profiling points {665.6, 3199} only and predict the deployment point 2133 (held-out, like the blind arm's situation but with a memory axis); (H2, upper bound) fit on all three levels in-sample.
- Pick rule identical to both arms: lowest grid clock with T̂_Han(f, 2133)·r ≤ D, r = 1.035, else declines; scored offline on the same interleaved sessions and pre-stated grids as Table II (NX MobileNetV2, two sessions) and the breadth table (ViT-S/YOLOv8s/ResNet-50).
- Reported per model and deadline next to blind/aware: pick, per-run miss median [min–max], runs > τ per session, pooled miss; summary = deadlines with ≥ 1 violating run and deadlines failing the pooled budget. Also the picks for the L2-resident GEMM proxy where the paper's sign test lives (inversion).
- 2026-10-13 — session 3 (nx_mobilenet_s3, run 2026-09-16) analysed: blind violates at 6/9 deadlines (clean at 5.4, 5.6, 6.0 — its tightest tail state, 816 MHz p99 5.40 ms) vs 7/9 in sessions 1–2; pooled union 8/9; the five 100 % cells (4.4, 4.6, 4.8, 5.2, 5.8) and the aware arm's 0/9 hold in all three sessions. Headline wording: "six to seven of nine per session, eight of nine pooled, every run over budget at five". E4: 0/16 new launches (CP 17.1 %), 0/24 pooled (11.7 %).

## E5-B — EMC-free replay of the Han et al. joint scaling policy (pre-stated 2026-10-28 13:36 KST, before scoring)
- Why: to compare against a memory-aware policy that also sets the memory clock, a *static* selector is replayed exactly on recorded traces, with the EMC-fixed and EMC-free settings reported separately. Han et al. (arXiv 2509.17970, Sec. IV-A) define three static policies, no controller: the "joint scaling policy" adjusts memory and computing frequency together "to meet the deadline constraint". Under the paper's platform-fixed memory point that policy reduces to E5's H1/H2 (energy per period rises monotonically with the GPU clock at every EMC level); this section replays the policy with the memory point free.
- Disclosure: before this section was written, an exploratory prototype with its own fitter was computed on 2026-10-28 (scratch only, not in the repository). The endpoints below are fixed here regardless of that preview; the numbers reported in the paper come only from the committed script `campaign/e5b_emcfree.py`, written after this section.
- Setting: NX, MobileNetV2 at 33 Hz, the nine pre-stated deadlines 4.4–6.0 ms (0.2 ms steps), τ = 2 %, r = 1.035. Candidate cells: EMC ∈ {665.6, 2133, 3199} MHz × GPU ∈ {612, 714, 816, 918, 1020, 1122, 1173} MHz. Excluded with reason: EMC 204 MHz (MobileNetV2 p50 23.7 ms at 1173 MHz, results/nx_validation catalogue cell; infeasible for every deadline); GPU 306–510 MHz (p50 ≥ 6.5 ms at EMC 2133/612 and 7.3 ms at 3199/510 in the rate-matched sweep; infeasible for every deadline); EMC 1066/1600 MHz are not lockable (the BPMP rounds them up to 2133 MHz, Supp. §S1).
- Recorded runs that score a pick: EMC 2133 — the three interleaved E1 sessions results/govrep_rev/{nx, nx_mobilenet, nx_mobilenet_s3} (10 runs per clock per session; session 1 ran 816–1173 only); EMC 3199 and 665.6 — results/nx_pairs e{3199,665}_g{612..1173}_r{1,2,3} (2026-07-03, 1000 cycles at 33 Hz, warm-up 60, SCHED_OTHER, INA power logged, no thermal gate, EMC lock requested in the launcher but no readback file; their p50s sit within −3.7..+0.8 % of the readback-verified 2026-09-15 rate-matched sweep cells, and the same-day refit33 cell (3199, 1173) reads +2.3 % higher energy than the July cell). Every candidate cell therefore has recorded runs; n is stated per cell.
- Latency models: exactly E5's H1 (fit on {665.6, 3199} MHz, C = 0, abs weighting) and H2 (all three levels, C free), from results/nx_validation_r33 via the committed `han_fit.fit_han`; the script asserts that it reproduces the published held-out errors (H1 median/max 6.6/15.1 %, H2 2.3/6.3 %).
- Power model: Han et al. Eq. (2), p = k_mem·f_mem³ + k_com·f_com³ + σ (f in GHz, k ≥ 0, σ ≥ 0), fitted by non-negative least squares to the per-cell mean of the trimmed-mean VDD_IN power (campaign trim p[n//10 : −n//10], the Table II energy convention): H1-power on the 14 cells at {665.6, 3199} MHz (profiling points only), H2-power on all 21 cells.
- Objective: energy per period at 33 Hz = predicted power / 33 Hz (the Table II E/inf metric). Han et al.'s e = p·t (continuous-inference busy energy) is NOT derivable from the 3.6 ms INA logs (the trace is smoothed within ±1.4 % inside a run) and is not evaluated; the paper says so.
- Pick rule (per arm, per deadline): among candidate cells with T̂_arm(g, emc)·r ≤ D, choose the cell with the lowest predicted energy per period (ties: lower GPU clock, then lower EMC); decline if none. The aware arm's pick is Table II's per-EMC two-cell refit at 2133 MHz (results/refit33/nx).
- Endpoints, fixed here: per deadline and arm — the pick (emc, g), T̂·r, runs with miss > τ out of the recorded runs at that cell (n), pooled miss, measured mean E/inf at the cell, ΔE % versus the measured mean E/inf of the aware arm's pick (unpaired; different sessions when the EMC differs — stated); the EMC-free measured frontier = the lowest-energy candidate cell with no recorded run over τ at that deadline, and whether each arm's pick is that cell. Summary per arm: admitted/declined; picks with ≥ 1 violating run; picks at 3199 and at 665.6 MHz; ΔE range over admitted deadlines; cells identical to the aware pick.
- Outputs: a Markdown results file (tables, verbatim script output), a LaTeX table (compact supplement table, label tab:e5bemcfree), a JSON file. Interpretation rules: a pick "has a violating run" if any recorded run at its cell exceeds τ; "escapes to 3199 MHz" = picks a 3199 MHz cell where the aware arm admits a 2133 MHz clock; no claim about Han's own objective is made.

## E5-B-live — interleaved re-run of the EMC-free picks (pre-stated 2026-10-28 13:36 KST, before the run)
- Why: the July 3199/665.6 MHz cells are n = 3, non-interleaved and lack a readback file; this one session supplies readback-verified, thermally gated, same-session n = 10 cells for every 3199 MHz pick the replay reports, paired by round with the aware arm's 2133 MHz picks.
- Board: NX (kernel 5.15.148-tegra, idle at 51 °C on 2026-10-28 12:32 KST), the E1 harness and settings (MAXN_SUPER, CPU governors performance, user-mode SCHED_OTHER loop pinned to CPU 7, 1000 cycles, warm-up 120, 33 Hz, INA power log per run), runner `campaign/govrep_emcfree.sh` (a variant of govrep_rev2.sh that re-locks the EMC with BPMP readback before every cell and logs the readback line per cell).
- Cells (from the prototype picks at 4.6–5.2 ms; amended here in writing before the run if the committed E5-B scoring picks other cells): 3199 MHz × {918, 1020, 1122, 1173} and 2133 MHz × {1020, 1122, 1173}; 10 rounds; round-robin order within a round 2133/1173, 3199/1173, 2133/1122, 3199/1122, 2133/1020, 3199/1020, 3199/918; thermal gate ≤ 55 °C before each round. Output results/e5b_live/nx/e{emc}_g{g}_r{round}.{json,csv,power.csv}, emc_lock.txt (one line per cell), session.log.
- Endpoints, fixed here: per cell — runs with miss > τ out of 10 for each deadline the replay assigns to it, pooled miss, mean E/inf; per deadline — the replay's pick versus the aware pick, paired by round (10 pairs): ΔE mJ and % with a paired-t 95 % interval (E6 method) and the violation counts of both; per 3199 MHz cell — live versus July p50 and E/inf. Rule: where a live cell exists, the paper reports the live numbers for that pick and keeps the July replay as the pre-stated source in the supplement; a pick whose live cell has ≥ 1 run over τ is reported as violating regardless of the July cells.
- Not run: ViT-S/YOLOv8s/ResNet-50 (their 3199 MHz deployment cells are not recorded), Han's e = p·t objective (not measurable here).
- Run record (added after the run): started 2026-10-28T12:41:35+08:00, finished 16:20:15+08:00 (3 h 39 min). Thermal gate: the ≤55 °C condition was met only before round 1; before every round 2–10 the gate waited its 20-min maximum (nine between-round gaps of 1233–1234 s) and released the round with the board at 55–58 °C (post-run maxima 53.2–58.1 °C, mean 57.1 °C), exactly as the E1 runner's tgate() behaves. Caveat: the within-round order is fixed, so cell position and a ≈1.9 °C in-round gradient are confounded; within-cell E/inf spreads are 0.2–1.1 mJ against paired premiums of 33–38 mJ, so this does not affect the verdicts, and it is stated with the results; 70/70 runs OK, 0 FAIL; emc_lock.txt 70/70 lines `requested=actual`; board restored (EMC unlocked, GPU 306–1173 MHz, governors schedutil). Results results/e5b_live/nx (212 files). Scored by campaign/e5b_emcfree.py (its Markdown, LaTeX and JSON outputs regenerated 2026-10-28 17:2x KST).
