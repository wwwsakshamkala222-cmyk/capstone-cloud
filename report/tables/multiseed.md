**Multi-seed results on the full release. Std is the population std over seeds; BEST rows from results/{task}_class/BEST/multi_seed_results.json, others computed from the per-seed JSONs.**

| Task | Model | Seeds | Test acc (mean ± std) | Test mF1 (mean ± std) |
|---|---|---|---|---|
| 2-class | lightgbm_cfg2 | 42, 123, 456 | 0.9938 ± 0.000003 | 0.9328 ± 0.000030 |
| 8-class | xgboost_cfg3_weighted | 42, 123, 456 | 0.8408 ± 0.000060 | 0.7317 ± 0.000056 |
| 34-class | xgboost_balanced_p050 | 42, 123, 456 | 0.7945 ± 0.000073 | 0.6434 ± 0.000157 |
| 34-class | hierarchical 8→34 (hard) | 1, 2, 42 | 0.7714 ± 0.000051 | 0.6352 ± 0.000102 |
| 2-class | CatBoost c1, no weights | 1, 2, 42 | 0.9927 ± 0.000045 | 0.9223 ± 0.000673 |
| 34-class | CatBoost c1, no weights | 1, 2, 42 | 0.7749 ± 0.000054 | 0.5408 ± 0.001091 |
