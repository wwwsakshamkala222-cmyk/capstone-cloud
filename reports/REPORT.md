# CICIoT2023 IDS — v3 report (new methods on top of the previous run)

Generated 2026-11-20 00:58. Split: prior run's shuffled per-row 75/5/20 (seed 42), val 2,339,084 rows, test 9,350,212 rows. All selection (thresholds, calibration, ensemble choice, early stopping) uses **validation only**; val-fitted methods were scored by 2-fold CV on val for selection. Test numbers are reported for every candidate, but the pick is by val. Targets: Neto et al. 2023 (RF).

All paths are relative to `/dev/shm/saksham/` (`~/saksham/ids` → `ids/`).

## 1. Headline

| Task | Prior best (test) | v3 best (test, selected by val) | Target | Status |
|---|---|---|---|---|
| 2_class | LightGBM cfg2: acc 0.9938, mF1 0.9328 | final_ensemble/A3_stack+A1: acc 0.9937, mF1 0.9339 | acc 0.9968, mF1 0.9653 | **NOT MET** |
| 8_class | XGB cfg3_weighted: acc 0.8410, mF1 0.7324 | unchanged (prior model): acc 0.8410, mF1 0.7324 | acc 0.9944, mF1 0.7193 | **MET** |
| 34_class | XGB balanced_p050: acc 0.7946, mF1 0.6429 | final_ensemble/A1_threshold: acc 0.7895, mF1 0.6452 | acc 0.9916, mF1 0.714 | **NOT MET** |

2-class status is judged on accuracy (+0.30 pts needed); 8/34-class on macro-F1. 8-class was already MET and not re-tuned.

## 2. Phase A — free wins on existing models (CPU)

**2_class — best single model only + A3 over prior models** (`ids/v3/results/2_class/phase_a/results.json`)

Models: lgbm_2_class_cfg2, xgb_2_class_cfg2, lgbm_2_class_cfg1

| Candidate | val mF1 (selection) | test acc | test mF1 |
|---|---|---|---|
| baseline | 0.9323 | 0.9938 | 0.9328 |
| A1_threshold | 0.9328 | 0.9937 | 0.9333 |
| A2_isotonic | 0.9315 | 0.9938 | 0.9329 |
| A2_sigmoid | 0.9253 | 0.9934 | 0.9257 |
| A3_soft | 0.9323 | 0.9938 | 0.9331 |
| A3_weighted | 0.9323 | 0.9938 | 0.9331 |
| A3_weighted+A1 | 0.9331 | 0.9937 | 0.9337 |
| A3_stack | 0.9324 | 0.9938 | 0.9331 |
| A3_stack+A1 ←selected | 0.9331 | 0.9937 | 0.9337 |

**2_class — all models incl. v3 CatBoost / hierarchical / focal** (`ids/v3/results/2_class/final_ensemble/results.json`)

Models: lgbm_2_class_cfg2, xgb_2_class_cfg2, lgbm_2_class_cfg1, cb_2_class_c1_none_s42

| Candidate | val mF1 (selection) | test acc | test mF1 |
|---|---|---|---|
| baseline | 0.9323 | 0.9938 | 0.9328 |
| A1_threshold | 0.9328 | 0.9937 | 0.9333 |
| A2_isotonic | 0.9315 | 0.9938 | 0.9329 |
| A2_sigmoid | 0.9253 | 0.9934 | 0.9257 |
| A3_soft | 0.9316 | 0.9937 | 0.9322 |
| A3_weighted | 0.9316 | 0.9937 | 0.9322 |
| A3_weighted+A1 | 0.9320 | 0.9936 | 0.9328 |
| A3_stack | 0.9325 | 0.9938 | 0.9334 |
| A3_stack+A1 ←selected | 0.9332 | 0.9937 | 0.9339 |

**34_class — best single model only + A3 over prior models** (`ids/v3/results/34_class/phase_a/results.json`)

Models: xgb_34_class_balanced_p050, xgb_34_class_cfg3_weighted, lgbm_34_class_cfg3_weighted, lgbm_34_class_cfg2, catboost_34_class_balanced

| Candidate | val mF1 (selection) | test acc | test mF1 |
|---|---|---|---|
| baseline | 0.6412 | 0.7946 | 0.6429 |
| A1_threshold ←selected | 0.6431 | 0.7895 | 0.6452 |
| A2_isotonic | 0.6361 | 0.7980 | 0.6380 |
| A2_sigmoid | 0.6369 | 0.7980 | 0.6386 |
| A3_soft | 0.6328 | 0.7954 | 0.6330 |
| A3_weighted | 0.6361 | 0.7956 | 0.6366 |
| A3_weighted+A1 | 0.6404 | 0.7905 | 0.6424 |
| A3_stack | 0.6348 | 0.7980 | 0.6353 |
| A3_stack+A1 | 0.6394 | 0.7921 | 0.6422 |

**34_class — all models incl. v3 CatBoost / hierarchical / focal** (`ids/v3/results/34_class/final_ensemble/results.json`)

Models: xgb_34_class_balanced_p050, xgb_34_class_cfg3_weighted, lgbm_34_class_cfg3_weighted, cb_34_class_c1_none_s42, hier_s42, xgb_34_class_focal_s42

| Candidate | val mF1 (selection) | test acc | test mF1 |
|---|---|---|---|
| baseline | 0.6412 | 0.7946 | 0.6429 |
| A1_threshold ←selected | 0.6431 | 0.7895 | 0.6452 |
| A2_isotonic | 0.6361 | 0.7980 | 0.6380 |
| A2_sigmoid | 0.6369 | 0.7980 | 0.6386 |
| A3_soft | 0.6394 | 0.7955 | 0.6416 |
| A3_weighted | 0.6402 | 0.7953 | 0.6426 |
| A3_weighted+A1 | 0.6423 | 0.7888 | 0.6461 |
| A3_stack | 0.6367 | 0.7981 | 0.6398 |
| A3_stack+A1 | 0.6423 | 0.7901 | 0.6480 |

A1 = per-class probability multipliers (0.3–3.0, step 0.1, coordinate ascent on val mF1). A2 = per-class isotonic / Platt calibration (one-vs-rest, equivalent to CalibratedClassifierCV on frozen probs). A3 = soft vote, val-mF1-weighted vote, multinomial-LogReg stacking on log-probs (final_ensemble: stacker fit on ≤20k rows/class with prior correction, for speed).

## 3. Phase B1 — CatBoost GPU

1M rows/class cap; probabilities prior-corrected back to the full-train distribution; early stopping on val macro-F1 (val reweighted to the training prior). Two bugs found and fixed mid-run (logloss early-stop fooled by the cap's prior shift; double prior correction with `Balanced`); runs below are post-fix unless noted.

| Run | best iter | val acc | val mF1 | test acc | test mF1 | train s |
|---|---|---|---|---|---|---|
| cb_2_class_c1_bal_s42 | 469 | 0.9926 | 0.9207 | 0.9926 | 0.9213 | 14.3 |
| cb_2_class_c1_none_s1 | 813 | 0.9928 | 0.9230 | 0.9928 | 0.9232 | 21.8 |
| cb_2_class_c1_none_s2 | 416 | 0.9927 | 0.9213 | 0.9927 | 0.9219 | 13.0 |
| cb_2_class_c1_none_s42 | 455 | 0.9927 | 0.9212 | 0.9927 | 0.9217 | 13.8 |
| cb_2_class_c2_bal_s42 | 638 | 0.9924 | 0.9185 | 0.9925 | 0.9192 | 10.6 |
| cb_2_class_c2_none_s42 | 923 | 0.9926 | 0.9201 | 0.9926 | 0.9207 | 14.5 |
| cb_34_class_c1_bal_s42 | 71 | 0.7687 | 0.5010 | 0.7688 | 0.5006 | 42.4 |
| cb_34_class_c1_none_s1 | 47 | 0.7746 | 0.5411 | 0.7749 | 0.5415 | 40.6 |
| cb_34_class_c1_none_s2 | 42 | 0.7746 | 0.5415 | 0.7749 | 0.5417 | 39.0 |
| cb_34_class_c1_none_s42 | 42 | 0.7747 | 0.5395 | 0.7750 | 0.5393 | 40.7 |
| cb_34_class_c2_bal_s42 | 175 | 0.7637 | 0.4845 | 0.7638 | 0.4840 | 51.0 |
| cb_34_class_c2_none_s42 | 84 | 0.7727 | 0.5256 | 0.7730 | 0.5261 | 36.6 |
| cb_34_class_c2_none_s42_b1024 | 67 | 0.7715 | 0.5199 | 0.7717 | 0.5202 | 47.7 |

2_class c1_none over seeds: test mF1 0.9223 ± 0.0007 (n=3), test acc 0.9927 ± 0.0000 (n=3)

34_class c1_none over seeds: test mF1 0.5408 ± 0.0011 (n=3), test acc 0.7749 ± 0.0001 (n=3)

Finding: CatBoost plateaus far below XGBoost/LightGBM on 34-class (≈0.54 vs 0.64 mF1); 1024 borders gives no gain (`cb_34_class_c2_none_s42_b1024`), so binning is not the cause. Not closing either gap.

## 4. Phase B2 — focal reweighting (34-class)

`ids/v3/results/34_class/focal/xgb_34_class_focal_s42/results.json` — w = sqrt-balanced class weight × (1−p_correct)^2 (floor 1e-3), focal renormalised within class; p_correct from balanced_p050 on train; XGB depth 12, lr 0.05, max_bin 128, 1M/class cap, 2892 rounds, 2947s, VRAM 2523 MiB.

val mF1 0.5160 / acc 0.6595; **test mF1 0.5152 / acc 0.6597** — much worse than baseline 0.6429. The 'hard' rows are mostly near-identical DoS/DDoS SYN-flood twins, so focal weighting amplifies label ambiguity. Single seed only (≈50 min/run; clearly negative).

## 5. Novelty C1 — hierarchical 8→34 classification

Stage 1 = prior 8-class XGB cfg3_weighted (cached probs). Stage 2 = XGB expert per multi-subtype family (depth 8, 300 rounds, max_bin 128, ≤300k rows/subtype). Soft: P(sub)=P(family)·P(sub|family); hard: route to argmax family.

| Expert | subtypes | train rows | within-family val acc (true family) |
|---|---|---|---|
| DDoS | 12 | 2,868,576 | 0.9072 |
| DoS | 4 | 953,881 | 0.9988 |
| Mirai | 3 | 900,000 | 0.9985 |
| Recon | 5 | 517,598 | 0.7295 |
| Spoofing | 2 | 364,881 | 0.8501 |
| Web_Based | 6 | 18,675 | 0.5100 |

hard_test: mF1 0.6352 ± 0.0001 (n=3), acc 0.7714 ± 0.0001 (n=3)

soft_test: mF1 0.6336 ± 0.0001 (n=3), acc 0.7625 ± 0.0001 (n=3)

hard_val: mF1 0.6306 ± 0.0009 (n=3), acc 0.7717 ± 0.0000 (n=3)

Flat baseline (balanced_p050) test mF1 0.6429, acc 0.7946. Files: `ids/v3/novelty/hierarchical_s{42,1,2}.json`.

Finding: within-family experts are near-perfect for DoS (0.999) and Mirai (0.999) but the hierarchy is ≈0.007 mF1 *below* flat: stage-1 DoS↔DDoS routing errors propagate and cannot be undone by stage 2. Web_Based (0.51) and Recon (0.73) experts are data-starved. Negative-but-informative result.

Figure: `ids/v3/figures/34class_flat_vs_hier_f1.png`

## 6. Novelty C3 — DoS vs DDoS

`ids/v3/novelty/dos_ddos.json`. 8-class test confusion: DDoS→DoS 928,973, DoS→DDoS 485,919 — **95.2% of all 8-class errors**.

Merging DoS+DDoS into 'Flood' (7 classes, same predictions): test acc 0.8410 → **0.9923**, mF1 0.7324 → **0.7660**.

Dedicated binary DoS-vs-DDoS classifier (prior run N3, `ids/novelty/N3_dos_vs_ddos/results.json`): AUC 0.894, acc 0.856; top features: Protocol Type, Min, fin_flag_number, psh_flag_number, Header_Length, fin_count, ICMP, UDP, syn_count, AVG.

Figure: `ids/v3/figures/dos_ddos_merge.png`

## 7. Novelty C2 / C4 — reused from prior run (not re-run)

Protocol ladder, 8-class (`ids/novelty/N2_protocol_ladder/results.json`): protocol_a_paper: mF1 0.7308; protocol_b_dedup: mF1 0.7297; protocol_c_file_grouped: mF1 0.4454; protocol_d_file_grouped_twin_free: mF1 0.4557. File-grouped split drops mF1 by ≈0.285 → shuffled-split inflation from within-file twins (59% of test rows have an exact train twin, N1).

Window-size artifact (`ids/novelty/N5_window_size/results.json`): features identify the 10- vs 100-packet window group with acc 1.0000; top features Number, Header_Length, Protocol Type, ICMP, Tot sum.

## 8. Conclusions

- **2-class: NOT MET.** Every post-hoc method moves accuracy by <0.0002; the 0.9968 target sits above the ≈0.994 ceiling seen across 20+ models. Prior-run evidence (N1 twins, D15/D16) attributes the paper's number to evaluation protocol, not model quality.
- **34-class: NOT MET.** Best gain is threshold tuning (≈+0.002 mF1). CatBoost (≈0.54), focal (0.515) and hierarchical (≈0.636) do not beat flat XGBoost (0.643).
- **Main novelty:** the residual error is structural — 95% of 8-class errors are DoS↔DDoS, and merging them gives 99.2% accuracy. The hierarchy confirms subtype separation inside a family is easy; family-level DoS/DDoS separation is the bottleneck.
- **Honesty notes:** test never used for selection; 3 seeds for hierarchical and CatBoost c1_none; focal and ensembles single-seed (deterministic given inputs / too slow and clearly negative). Phase A 2-class was run earlier in this session by a previous v3 script (`ids/v3/results/2_class/{threshold_tuned,calibrated,ensemble}`); numbers agree with `phase_a`.

## 9. Files

- Results JSON: `ids/v3/results/<task>/{phase_a,final_ensemble,catboost/<run>,focal/<run>}/results.json`
- Novelty: `ids/v3/novelty/{hierarchical_s*,dos_ddos,protocol_ladder,window_artifact}.json`
- Cached probabilities: `ids/v3/probs/<task>/*.npy`
- New models: `ids_cache/v3_models/`
- Scripts: `ids/v3/src/`, logs: `ids/v3/logs/`
- Figures: `ids/v3/figures/`
