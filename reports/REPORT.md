# CICIoT2023 IDS — v3 report (new methods on top of the previous run)

Generated 2026-11-20 00:58. Split: prior run's shuffled per-row 75/5/20 (seed 42), val 2,339,084 rows, test 9,350,212 rows. All selection (thresholds, calibration, ensemble choice, early stopping) uses **validation only**; val-fitted methods were scored by 2-fold CV on val for selection. Test numbers are reported for every candidate, but the pick is by val. Targets: Neto et al. 2023 (RF).

All paths are relative to `/dev/shm/saksham/` (`~/saksham/ids` → `ids/`).

## 1. Headline

| Task | Prior best single model (test; best on val) | v3 best (test, selected by val) | Target | Status |
|---|---|---|---|---|
| 2_class | LightGBM cfg2: acc 0.9938, mF1 0.9328 | final_ensemble/A3_stack+A1: acc 0.9937, mF1 0.9339 | acc 0.9968, mF1 0.9653 | **NOT MET** |
| 8_class | XGB balanced_deep: acc 0.8415, mF1 0.7308 | unchanged (prior model, best on val): acc 0.8415, mF1 0.7308 | acc 0.9944, mF1 0.7193 | **MET** |
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

`results/novelty/v3/dos_ddos_balanced_deep.json` (8-class model XGB balanced_deep). 8-class test confusion: DDoS→DoS 921,122, DoS→DDoS 489,145 — **95.1% of all 8-class errors**.

Merging DoS+DDoS into 'Flood' (7 classes, same predictions): test acc 0.8415 → **0.9923**, mF1 0.7308 → **0.7642**.

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


## 10. v4: 46-feature ablation (Kaggle `himadri07/ciciot2023` subset)

Added 2026-11-20. **Different data:** a 2.2 GB Kaggle subset (7,845,673 rows, the paper's 46 features + `label`), not
the full 46.8M-row release used in sections 1–9. The subset's own train/validation/test files were merged and
re-split with the paper's protocol: shuffled 80/20 train/test (seed 42), then 5% of train as validation for
selection only. Sizes: train 5,962,711, val 313,827, test 1,569,135. StandardScaler fit on train, no
de-duplication. Each model was trained on the same rows with 46 features, with 38 features (46 minus
flow_duration, Srate, Drate, urg_count, Magnitue, Radius, Covariance and Weight; this equals our 39-feature
release without IGMP, which the subset lacks), and with each of those minus IAT. Models: RF-200 (`n_jobs=16`, sklearn
defaults), XGBoost `xgb_best` (2-class cfg2, 8-class cfg3_weighted, 34-class balanced_p050) and `xgb_unw` (one unweighted
XGBoost per task; no-IAT runs used RF + xgb_best only). Early stopping and model selection used validation only
(2-class: accuracy; 8/34-class: macro-F1). Test was scored once per run.

**Findings**

1. **Paper beaten with or without the 8 extra features.** With 46 features: 2-class acc 0.9977 (paper 0.9968), 8-class mF1
   0.8925 (0.7193), 34-class mF1 0.8484 (0.7140). With 38 features: 0.9972 / 0.8799 / 0.8361. The 8 paper-only
   features add only +0.0005 acc and +0.012–0.013 mF1. **The v3-to-paper gap is mostly a data-version effect:** the
   same 38 features give 34-class mF1 0.645 on the full release and 0.836 here.
2. **IAT is a capture-time proxy.** IAT values cluster around 8.3e7 (floods) or 1.66e8 (all else), with a narrow band
   per class (e.g. DoS 8.293–8.303e7, DDoS 8.303–8.34e7). One depth-8 tree on IAT alone reaches 99.05% 8-class
   val accuracy (mF1 0.82; majority baseline 72.8%). IAT is the top 8-class feature (40% of gain).
   **Without IAT:** 2-class acc 0.9959 (NOT BEAT), 8-class mF1 0.7547 (BEAT), 34-class mF1 0.7143 (above 0.7140 by
   0.0003, effectively a tie). Accuracy falls to 0.915 (8-class) and 0.898 (34-class).
3. **DoS vs DDoS.** With IAT the confusion nearly disappears (285 test rows, 5% of errors). Without IAT it comes back
   (123,149 rows, 93% of errors), matching v3. **The 8 paper-only features do not fix it** (38f without IAT: 133,657).
4. **Twins.** 5.61% of test rows have an exact copy in train (the same for 38 and 46 features); 3.51% of
   all rows are duplicates. This is far below the full release's 59%, so twin memorisation is not what drives these numbers.

Figures: `figures/v4/v4_ablation_headline.png`, `figures/v4/v4_top15_importance_46f.png`, `figures/v4/v4_8class_confusion.png`.
Results: `results/v4_46feat/` (`summary.json`, `tables.md`, one JSON per run). Code: `src/v4/{prep_v4,train_v4,report_v4}.py`.
Models and arrays stay in `/dev/shm/saksham/v4/` (not committed).

### All features vs our 38

| Task | Judged on | v3: full release, 39f (75/5/20) | v4 38 features | v4 46 features | Neto et al. 2023 RF | 38f vs paper | 46f vs paper |
|---|---|---|---|---|---|---|---|
| 2-class | accuracy | 0.9937 / 0.9339 | 0.9972 / 0.9696 (rf) | 0.9977 / 0.9754 (rf) | 0.9968 / 0.9653 | **BEAT** | **BEAT** |
| 8-class | macro-F1 | 0.8415 / 0.7308 | 0.9961 / 0.8799 (xgb_best) | 0.9967 / 0.8925 (xgb_best) | 0.9944 / 0.7193 | **BEAT** | **BEAT** |
| 34-class | macro-F1 | 0.7895 / 0.6452 | 0.9951 / 0.8361 (xgb_best) | 0.9957 / 0.8484 (xgb_best) | 0.9916 / 0.7140 | **BEAT** | **BEAT** |

### Without IAT (capture-timestamp proxy removed)

| Task | Judged on | v3: full release, 39f (75/5/20) | v4 38 features, no IAT | v4 46 features, no IAT | Neto et al. 2023 RF | 38nf vs paper | 46nf vs paper |
|---|---|---|---|---|---|---|---|
| 2-class | accuracy | 0.9937 / 0.9339 | 0.9947 / 0.9436 (xgb_best) | 0.9959 / 0.9562 (xgb_best) | 0.9968 / 0.9653 | NOT BEAT | NOT BEAT |
| 8-class | macro-F1 | 0.8415 / 0.7308 | 0.9076 / 0.7346 (xgb_best) | 0.9154 / 0.7547 (xgb_best) | 0.9944 / 0.7193 | **BEAT** | **BEAT** |
| 34-class | macro-F1 | 0.7895 / 0.6452 | 0.8900 / 0.6979 (xgb_best) | 0.8976 / 0.7143 (xgb_best) | 0.9916 / 0.7140 | NOT BEAT | **BEAT** |

### DoS vs DDoS confusions (8-class test set; ✓ = validation-selected model)

| Features | Model | DoS→DDoS | DDoS→DoS | Total | Share of all 8-class errors |
|---|---|---|---|---|---|
| 38 features | rf | 53 | 48 | 101 | 1.6% |
| 38 features | xgb_best ✓ | 95 | 211 | 306 | 4.9% |
| 38 features | xgb_unw | 159 | 96 | 255 | 3.5% |
| 46 features | rf | 55 | 48 | 103 | 1.8% |
| 46 features | xgb_best ✓ | 93 | 192 | 285 | 5.5% |
| 46 features | xgb_unw | 145 | 82 | 227 | 3.7% |
| 38 features, no IAT | rf | 93,258 | 41,787 | 135,045 | 91.7% |
| 38 features, no IAT | xgb_best ✓ | 81,062 | 52,595 | 133,657 | 92.2% |
| 46 features, no IAT | rf | 88,310 | 31,320 | 119,630 | 91.7% |
| 46 features, no IAT | xgb_best ✓ | 75,432 | 47,717 | 123,149 | 92.8% |

### All runs

| Task | Features | Model | Val acc | Val mF1 | Test acc | Test mF1 | Test wF1 | Selected (val) |
|---|---|---|---|---|---|---|---|---|
| 2 | 38 features | rf | 0.9971 | 0.9685 | 0.9972 | 0.9696 | 0.9972 | ✓ |
| 2 | 38 features | xgb_best | 0.9965 | 0.9632 | 0.9967 | 0.9652 | 0.9968 |  |
| 2 | 38 features | xgb_unw | 0.9963 | 0.9602 | 0.9964 | 0.9618 | 0.9964 |  |
| 2 | 46 features | rf | 0.9976 | 0.9744 | 0.9977 | 0.9754 | 0.9977 | ✓ |
| 2 | 46 features | xgb_best | 0.9974 | 0.9720 | 0.9974 | 0.9723 | 0.9974 |  |
| 2 | 46 features | xgb_unw | 0.9972 | 0.9696 | 0.9972 | 0.9696 | 0.9972 |  |
| 2 | 38 features, no IAT | rf | 0.9944 | 0.9404 | 0.9945 | 0.9417 | 0.9946 |  |
| 2 | 38 features, no IAT | xgb_best | 0.9946 | 0.9429 | 0.9947 | 0.9436 | 0.9947 | ✓ |
| 2 | 46 features, no IAT | rf | 0.9954 | 0.9513 | 0.9955 | 0.9523 | 0.9956 |  |
| 2 | 46 features, no IAT | xgb_best | 0.9959 | 0.9566 | 0.9959 | 0.9562 | 0.9959 | ✓ |
| 8 | 38 features | rf | 0.9959 | 0.8806 | 0.9960 | 0.8758 | 0.9958 |  |
| 8 | 38 features | xgb_best | 0.9962 | 0.8823 | 0.9961 | 0.8799 | 0.9960 | ✓ |
| 8 | 38 features | xgb_unw | 0.9953 | 0.8219 | 0.9954 | 0.8138 | 0.9952 |  |
| 8 | 46 features | rf | 0.9962 | 0.8639 | 0.9963 | 0.8681 | 0.9962 |  |
| 8 | 46 features | xgb_best | 0.9968 | 0.8967 | 0.9967 | 0.8925 | 0.9966 | ✓ |
| 8 | 46 features | xgb_unw | 0.9960 | 0.8381 | 0.9960 | 0.8254 | 0.9959 |  |
| 8 | 38 features, no IAT | rf | 0.9064 | 0.7206 | 0.9061 | 0.7074 | 0.9022 |  |
| 8 | 38 features, no IAT | xgb_best | 0.9075 | 0.7419 | 0.9076 | 0.7346 | 0.9055 | ✓ |
| 8 | 46 features, no IAT | rf | 0.9170 | 0.7306 | 0.9168 | 0.7193 | 0.9130 |  |
| 8 | 46 features, no IAT | xgb_best | 0.9156 | 0.7603 | 0.9154 | 0.7547 | 0.9136 | ✓ |
| 34 | 38 features | rf | 0.9947 | 0.8082 | 0.9948 | 0.8011 | 0.9944 |  |
| 34 | 38 features | xgb_best | 0.9951 | 0.8291 | 0.9951 | 0.8361 | 0.9949 | ✓ |
| 34 | 38 features | xgb_unw | 0.9940 | 0.7780 | 0.9941 | 0.7819 | 0.9937 |  |
| 34 | 46 features | rf | 0.9949 | 0.7901 | 0.9950 | 0.7962 | 0.9947 |  |
| 34 | 46 features | xgb_best | 0.9957 | 0.8406 | 0.9957 | 0.8484 | 0.9956 | ✓ |
| 34 | 46 features | xgb_unw | 0.9946 | 0.7926 | 0.9947 | 0.7907 | 0.9944 |  |
| 34 | 38 features, no IAT | rf | 0.8844 | 0.6795 | 0.8842 | 0.6830 | 0.8810 |  |
| 34 | 38 features, no IAT | xgb_best | 0.8900 | 0.6969 | 0.8900 | 0.6979 | 0.8873 | ✓ |
| 34 | 46 features, no IAT | rf | 0.8951 | 0.6951 | 0.8950 | 0.6959 | 0.8922 |  |
| 34 | 46 features, no IAT | xgb_best | 0.8973 | 0.7153 | 0.8976 | 0.7143 | 0.8952 | ✓ |
