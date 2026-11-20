# CICIoT2023 Intrusion Detection — Capstone

Author: Saksham Kala, Bennett University (B.Tech CSE 2023–2027)

Gradient-boosted tree classifiers for IoT network-flow intrusion detection on the
[CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) dataset, benchmarked against
the Random Forest results of Neto et al. (2023) at three label granularities (2, 8 and 34 classes),
plus an analysis of *why* the published numbers are hard to reproduce honestly.

## Dataset

- **CICIoT2023** (Canadian Institute for Cybersecurity, UNB): 46.8M flow records, 39 numeric
  features, 33 attacks + benign, captured from a 105-device IoT testbed.
- Label hierarchies: 2-class (benign / attack), 8-class (Benign, DDoS, DoS, Mirai, Recon,
  Spoofing, Web-based, Brute-force), 34-class (individual attacks).
- Split: shuffled per-row 75 / 5 / 20 (train / val / test), seed 42, no de-duplication — this
  mirrors the paper's protocol so the comparison is like-for-like.
- See [`data/README.md`](data/README.md) for download.
  No data is stored in this repo.

## Results (test set, models selected on validation only)

| Task | Best model | Test accuracy | Test macro-F1 | Neto et al. 2023 (acc / mF1) | Status |
|---|---|---|---|---|---|
| 2-class | Stacked ensemble (LGBM+XGB+CatBoost) + threshold | 0.9937 | 0.9339 | 0.9968 / 0.9653 | Not met (judged on accuracy) |
| 8-class | XGBoost, sqrt class weights (cfg3_weighted) | 0.8410 | **0.7324** | 0.9944 / 0.7193 | **Met** (macro-F1) |
| 34-class | XGBoost balanced p=0.5 + per-class thresholds | 0.7895 | 0.6452 | 0.9916 / 0.714 | Not met (macro-F1) |

Multi-seed (42/123/456) standard deviations are ≤ 0.0002 for every headline number
(`results/*/BEST/multi_seed_results.json`). Full tables: [`reports/REPORT.md`](reports/REPORT.md)
and [`reports/ids_v3_report.html`](reports/ids_v3_report.html).

What did **not** help (v3): post-hoc calibration / stacking (≤ +0.002 mF1), CatBoost GPU
(34-class mF1 ≈ 0.54), focal reweighting (0.515), hierarchical 8→34 (0.636 vs flat 0.643).

## Key finding: DoS vs DDoS is the bottleneck

- **95% of all 8-class test errors are DoS ↔ DDoS confusions**
  (`results/novelty/v3/dos_ddos.json`).
- Merging the two into a single "Flood" class — re-scoring the *same* predictions, no retraining —
  lifts 8-class test accuracy from 0.841 to **0.992** (macro-F1 0.732 → 0.766).
- Single-source and distributed floods produce near-identical per-flow features; the distinction
  is a property of the *aggregate* traffic, which per-flow features can't see. A binary
  DoS-vs-DDoS model drops from 0.856 to 0.808 accuracy under file-grouped evaluation (N3).

## Leakage caveat

- **59.1% of test rows are exact duplicates ("twins") of training rows** under the shuffled
  per-row split (DDoS: 67.6%, Benign: 0.5%) — `results/novelty/N1_twin_analysis/results.json`.
- Under the strictest protocol tested (de-duplicated + file-grouped), macro-F1 falls by
  **0.275** relative to the paper protocol (N2, `results/novelty/N2_protocol_ladder/`).
- A classifier can identify a row's capture window with 99.9997% accuracy (N5), so window-level
  artifacts also leak class information.
- Takeaway: headline accuracies near 99% on CICIoT2023 (ours and the paper's) largely reflect
  memorisation of duplicated flows, not generalisation. Report them alongside a grouped split.

## v4: 46-feature ablation (Kaggle subset)

**Data caveat:** v4 uses the 2.2 GB Kaggle subset `himadri07/ciciot2023` (7.85M rows, the paper's 46
features plus `label`), not the full 46.8M-row release used above. Results from the two datasets are
not directly comparable. Protocol: merge the subset's train/val/test files, shuffle, 80/20
train/test split (seed 42), 5% of train as validation (used for model selection only), StandardScaler
fit on train, no de-duplication. Each model was trained on the **same rows** twice: once with all 46
features and once with the 38 features that overlap our 39-feature release (the 8 paper-only features
removed). Models: RF-200 (the paper's method), XGBoost with our best v3 configs, and one unweighted
XGBoost per task. Each cell below shows the model that scored best on validation.

| Task (judged on) | 38 features | 46 features | 46 features, **no IAT** | Neto et al. 2023 RF |
|---|---|---|---|---|
| 2-class (accuracy) | 0.9972 **BEAT** | 0.9977 **BEAT** | 0.9959 NOT BEAT | 0.9968 |
| 8-class (macro-F1) | 0.8799 **BEAT** | 0.8925 **BEAT** | 0.7547 **BEAT** | 0.7193 |
| 34-class (macro-F1) | 0.8361 **BEAT** | 0.8484 **BEAT** | 0.7143 BEAT by 0.0003 (a tie) | 0.7140 |

- **The 8 extra features add little:** +0.0005 accuracy (2-class) and +0.012 to +0.013 macro-F1
  (8 and 34-class). Most of the gap between v3 and the paper comes from the different data
  version, not from the missing features. With the same 38 features, 34-class macro-F1 is 0.645 on
  the full release and 0.836 on this subset.
- **IAT acts as a timestamp.** In this subset `IAT` takes values around 8.3e7 (floods) or 1.66e8
  (everything else), and each attack class sits in its own narrow band. A single depth-8 tree on
  IAT alone reaches 99.05% 8-class accuracy. With IAT dropped, 8-class accuracy falls from 0.997 to
  0.915 and 34-class from 0.996 to 0.898. The headline ~99.5% accuracies, ours and the paper's,
  depend heavily on this capture-time artifact.
- **DoS vs DDoS:** with IAT, only 285 test rows are DoS↔DDoS confusions (5% of 8-class errors). Without
  IAT this rises to 123,149 (93% of errors), the same bottleneck we found in v3. The 8 paper-only
  features don't fix it (38 features without IAT: 133,657).
- **Twins:** 5.6% of test rows have an exact copy in train (3.5% duplicate rows overall), much lower
  than the 59% on the full release.
- Full tables: [`results/v4_46feat/tables.md`](results/v4_46feat/tables.md). Figures:
  [`figures/v4/`](figures/v4/). Code: [`src/v4/`](src/v4/).

## Repository layout

```
src/common.py          shared paths, data loading, prediction helpers (v3 scripts)
src/preprocessing/     CSV → labelled parquet cache + 75/5/20 split; EDA profiling
src/training/          LightGBM / XGBoost / CatBoost / RF / MLP / focal training; GPU queue scripts
src/evaluation/        threshold tuning, calibration, ensembles, multi-seed eval, reports, figures
src/v4/                v4 46-feature ablation (prep, train, report, launchers)
src/novelty/           N1–N6 analyses (twins, protocol ladder, DoS vs DDoS, ablation, window,
                       class-weight sweep) and hierarchical 8→34
results/{2,8,34}_class/  per-model results.json (v3 runs under v3/)
results/novelty/       novelty-analysis JSONs
results/v4_46feat/     v4 per-run JSONs, summary.json, tables.md
reports/               REPORT.md, HTML report, run summaries, decision log
figures/               all plots (figures/v3, figures/eda)
logs/                  run logs
```

## How to run

```bash
python3 -m venv ids_env && source ids_env/bin/activate
pip install -r requirements.txt          # see note there for the CUDA torch wheel

# 1. Paths are set as constants at the top of each script (and in src/common.py):
#    DATA_ROOT  -> CICIoT2023 CSV folder
#    CACHE_ROOT -> parquet cache + trained models (needs ~40 GB)
#    IDS_ROOT   -> output folder for results/figures
#    Edit them to match your machine before running.

python src/preprocessing/phase1_data_pipeline.py      # CSVs -> labelled, split parquet cache
python src/training/phase2_train_v3.py                # LightGBM / XGBoost / CatBoost / MLP, all tasks
python src/training/phase2d_random_forest.py          # paper's RF baseline
python src/evaluation/multi_seed_eval.py              # 3-seed evaluation of the best models
python src/novelty/phase3_novelty_v2.py               # N1–N6 analyses
python src/evaluation/phase4_figures.py               # figures

# v3 (post-hoc tuning, CatBoost GPU, focal, hierarchical)
python src/evaluation/cache_probs.py <task> <model_file> val,test   # see src/evaluation/run_cache.sh
python src/evaluation/phase_a_free_wins.py
python src/training/phase_b1_catboost.py <task> <c1|c2> <bal|none> <seed>
python src/training/phase_b2_focal.py <seed>
python src/novelty/phase_c1_hier.py <seed>
python src/novelty/phase_c234.py
python src/evaluation/make_report.py
```

Hardware used: NVIDIA GPU for XGBoost (2.1.4, `gpu_hist`) and CatBoost; LightGBM on CPU
with 16 threads (more threads were slower on a 4-NUMA-node machine — see `reports/DECISIONS.md`).
Trained models and probability caches are not included (tens of GB).

## Reference

Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. A. (2023).
CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment.
*Sensors*, 23(13), 5941.
