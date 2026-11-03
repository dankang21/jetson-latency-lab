> **Public copy.** This is the public copy of the E8 protocol log kept with
> the manuscript "When Memory-Clock State Changes Deadline Admission: Evidence
> from Jetson Orin". The freeze and amendment dates are those recorded in the
> log. They are self-attested: the file was not under version control until
> this release. Passages that quoted or cited non-public documents are
> replaced by a neutral description of what they address; an internal working
> reference (a user name) is removed. Nothing else is changed.

---

# E8 — decode-horizon coverage under co-tenancy

**Status: APPROVED 2026-09-15; FROZEN before any run.**

Answers whether the decode-horizon term, shown under co-tenancy only at the
submitted 2,000-token horizon, holds at other horizons. Drafted 2026-09-15 before any run; frozen on approval (same day).

- Board NX; primary Qwen2.5-1.5B Q4_K_M decode (llama.cpp driver
  tools/slm_decode_bench), GPU 1173 MHz, EMC 2133, co-tenant ViT-Small 10 Hz
  (the submitted joint-stress co-runner), user-mode loop as in
  results/joint_hxt.
- Horizons H ∈ {1,000, 3,000} tokens (submitted: 2,000; solo 4,000 exists in
  results/horizon_4k). Deadline: the submitted joint cell's first accepted
  D = 44 ms and the tenancy-blind D = 28 ms, both replayed at each H.
- 5 independent launches per (H, D) cell → 20 contracts ≈ 20 × (H tokens ×
  ~27 ms) ≈ 1.5 h plus co-runner warm-up and thermal gates.
- Endpoints (pre-stated): per-launch token miss % vs D; the drift
  coefficient γ re-estimated from tokens 0–500 vs the last 500 of each H,
  compared with the submitted γ (from H = 2,000 solo runs); whether the
  admission rule's (1+γH) term, evaluated with the submitted γ, predicts the
  measured late-horizon median within the r margin at H = 1,000 and 3,000.
- Interpretation rule: if the H = 3,000 contract at D = 44 ms exceeds τ, the
  paper reports the horizon limit of the accepted contract explicitly
  (a contract is admitted for a stated H, not for arbitrary lengths).

## Amendment 2026-09-15 (before any run)
- D is a post-hoc analysis parameter of the per-token trace (as in every
  decode cell of the paper), so one 5-launch series per H serves both
  D = 28 and D = 44 ms: 10 contracts total (H = 1,000 and 3,000), each a
  fresh process with the ViT-Small 10 Hz co-runner started 20 s ahead, GPU
  1173, EMC 2133, same slm_decode_bench arguments as results/joint_hxt
  (warm-up 40 tokens, victim CPU 7, --prio 80). Thermal gate ≤ 55 °C.
- Output: results/horizon_ext/nx/H<H>/t<k>.csv (+ .json), session.log.

## Result (2026-09-16, results/horizon_ext/nx; 5 fresh launches per H, ViT-S 10 Hz co-tenant, GPU 1173, EMC 2133)
- H = 1,000: D = 28 ms → 5/5 launches > τ (pooled 32.30 %, last-500 32.5 %); D = 44 ms → 0/5 (pooled 0.060 %, last-500 0.08 %; one-sided 95 % CP UB 45.1 % on the launch exceedance).
- H = 3,000: D = 28 ms → 5/5 (32.95 %, last-500 33.7 %); D = 44 ms → 0/5 (0.127 %, last-500 0.20 %).
- Drift coefficient re-estimated per launch: pooled γ = 1.715e-5 (H = 1,000) and 1.763e-5 (H = 3,000) vs the submitted 1.683e-5 (ratio 1.02 / 1.05); the rule's (1+γH) prediction of the tail-window median errs by −0.81 % / −0.61 % — inside the r = 1.035 margin at both horizons (all 10 launches within 3.5 %).
- Rule replay with the submitted parameters: (H = 1,000, D = 44) accepted (D* = 43.35 ms) and measured feasible; (H = 3,000, D = 44) declined (D* = 44.26 ms) although the measured contract met the budget → a conservative decline by 0.26 ms, consistent with the paper's stance that refusals err conservative; D = 28 is infeasible under co-tenancy at any H (measured 32–33 %).
- Interpretation rule outcome: the H = 3,000 contract at D = 44 did not exceed τ, so no horizon limit needs to be declared beyond the rule's own decline; the horizon term generalises across 1,000–3,000 tokens within the stated margin.
