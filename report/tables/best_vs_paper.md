**Best model per task against Neto et al. 2023 (RF, Table 6). 2-class is judged on accuracy, 8- and 34-class on macro-F1. Δ = ours minus paper.**

| Setting | Task | Model (selected on val) | Test acc | Test mF1 | Paper acc / F1 | Δ (judged metric) | Verdict |
|---|---|---|---|---|---|---|---|
| Full release, 39 features (v1-v3) | 2-class | Stacked ensemble + threshold (v3) | 0.9937 | 0.9339 | 0.9968 / 0.9653 | acc -0.0031 | NOT BEAT |
| Full release, 39 features (v1-v3) | 8-class | XGBoost cfg3_weighted (v1) | 0.8410 | 0.7324 | 0.9944 / 0.7193 | mF1 +0.0131 | **BEAT** |
| Full release, 39 features (v1-v3) | 34-class | XGBoost balanced_p050 + per-class thresholds (v3) | 0.7895 | 0.6452 | 0.9916 / 0.7140 | mF1 -0.0688 | NOT BEAT |
| Kaggle subset, 46 features (v4) | 2-class | Random Forest (200 trees) | 0.9977 | 0.9754 | 0.9968 / 0.9653 | acc +0.0009 | **BEAT** |
| Kaggle subset, 46 features (v4) | 8-class | XGBoost (prior best config) | 0.9967 | 0.8925 | 0.9944 / 0.7193 | mF1 +0.1732 | **BEAT** |
| Kaggle subset, 46 features (v4) | 34-class | XGBoost (prior best config) | 0.9957 | 0.8484 | 0.9916 / 0.7140 | mF1 +0.1343 | **BEAT** |
| Kaggle subset, 46 features, no IAT (v4) | 2-class | XGBoost (prior best config) | 0.9959 | 0.9562 | 0.9968 / 0.9653 | acc -0.0009 | NOT BEAT |
| Kaggle subset, 46 features, no IAT (v4) | 8-class | XGBoost (prior best config) | 0.9154 | 0.7547 | 0.9944 / 0.7193 | mF1 +0.0354 | **BEAT** |
| Kaggle subset, 46 features, no IAT (v4) | 34-class | XGBoost (prior best config) | 0.8976 | 0.7143 | 0.9916 / 0.7140 | mF1 +0.0003 | **BEAT** |
