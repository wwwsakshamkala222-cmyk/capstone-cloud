**Every model trained in this project (v1-v4). Val = validation split, mF1 = macro-F1. n/a: no separate validation (train+val run, or test-only push run). n/r = not recorded during the original run.**

| Ver | Data | Task | Model | Config | Val acc | Val mF1 | Test acc | Test mF1 | Train s |
|---|---|---|---|---|---|---|---|---|---|
| v1 | full 39f | 2 | catboost_balanced | depth=10, lr=0.1, iters=3000, w=Balanced | 0.9863 | 0.8834 | 0.9864 | 0.8840 | 111 |
| v1 | full 39f | 2 | lightgbm_cfg1 | leaves=255, lr=0.1 | 0.9937 | 0.9314 | 0.9937 | 0.9319 | 1,810 |
| v1 | full 39f | 2 | lightgbm_cfg2 | leaves=512, lr=0.05 | 0.9937 | 0.9323 | 0.9938 | 0.9328 | 3,465 |
| v1 | full 39f | 2 | lightgbm_cfg3_weighted | leaves=512, lr=0.05, sqrt class weights | 0.9921 | 0.9229 | 0.9921 | 0.9236 | 7,975 |
| v1 | full 39f | 2 | mlp_v1 | layers=1024-512-256 | 0.9883 | 0.8962 | 0.9884 | 0.8967 | 2,008 |
| v1 | full 39f | 2 | xgboost_cfg1 | depth=12, lr=0.1 | 0.9934 | 0.9286 | 0.9934 | 0.9295 | 37 |
| v1 | full 39f | 2 | xgboost_cfg2 | leaves=1024, lr=0.05 | 0.9935 | 0.9303 | 0.9936 | 0.9313 | 115 |
| v1 | full 39f | 2 | xgboost_cfg3_weighted | leaves=1024, lr=0.05, sqrt class weights | 0.9903 | 0.9102 | 0.9903 | 0.9106 | 499 |
| v1 | full 39f | 8 | catboost_balanced | depth=10, lr=0.1, iters=3000, w=Balanced | 0.7454 | 0.6324 | 0.7451 | 0.6333 | 167 |
| v1 | full 39f | 8 | lightgbm_cfg1 | leaves=255, lr=0.1 | 0.8613 | 0.6629 | 0.8610 | 0.6672 | 818 |
| v1 | full 39f | 8 | lightgbm_cfg2 | leaves=512, lr=0.05 | 0.8634 | 0.6899 | 0.8633 | 0.6923 | 2,921 |
| v1 | full 39f | 8 | lightgbm_cfg3_weighted | leaves=512, lr=0.05, sqrt class weights | 0.8407 | 0.7283 | 0.8403 | 0.7305 | 22,199 |
| v1 | full 39f | 8 | mlp_v1 | layers=1024-512-256 | 0.8334 | 0.6940 | 0.8330 | 0.6976 | 1,807 |
| v1 | full 39f | 8 | xgboost_cfg1 | depth=12, lr=0.1 | 0.8645 | 0.7123 | 0.8642 | 0.7129 | 750 |
| v1 | full 39f | 8 | xgboost_cfg2 | leaves=1024, lr=0.05 | 0.8652 | 0.7166 | 0.8650 | 0.7168 | 9,577 |
| v1 | full 39f | 8 | xgboost_cfg3_weighted | leaves=1024, lr=0.05, sqrt class weights | 0.8415 | 0.7282 | 0.8410 | 0.7324 | 10,626 |
| v1 | full 39f | 34 | catboost_balanced | depth=10, lr=0.1, iters=3000, w=Balanced | 0.7425 | 0.5125 | 0.7423 | 0.5130 | 101 |
| v1 | full 39f | 34 | lightgbm_cfg1 | leaves=255, lr=0.1 | 0.7847 | 0.5458 | 0.7845 | 0.5460 | 2,362 |
| v1 | full 39f | 34 | lightgbm_cfg2 | leaves=512, lr=0.05 | 0.7886 | 0.5698 | 0.7886 | 0.5703 | 5,205 |
| v1 | full 39f | 34 | lightgbm_cfg3_weighted | leaves=512, lr=0.05, sqrt class weights | 0.7893 | 0.6137 | 0.7891 | 0.6158 | 7,866 |
| v1 | full 39f | 34 | mlp_v1 | layers=1024-512-256 | 0.7827 | 0.6111 | 0.7827 | 0.6140 | 1,886 |
| v1 | full 39f | 34 | xgboost_cfg1 | depth=12, lr=0.1 | 0.7959 | 0.6323 | 0.7958 | 0.6352 | 944 |
| v1 | full 39f | 34 | xgboost_cfg2 | leaves=1024, lr=0.05 | 0.7977 | 0.6349 | 0.7975 | 0.6369 | 2,158 |
| v1 | full 39f | 34 | xgboost_cfg3_weighted | leaves=1024, lr=0.05, sqrt class weights | 0.7939 | 0.6398 | 0.7939 | 0.6423 | 13,137 |
| v2 | full 39f | 2 | accpush_xgb_ensemble_threshold | - | n/a | n/a | 0.9934 | 0.9318 | n/r |
| v2 | full 39f | 2 | accpush_xgboost_deep_low_lr | depth=16, lr=0.02 | n/a | n/a | 0.9936 | 0.9314 | 126 |
| v2 | full 39f | 2 | accpush_xgboost_lossguide_wide | leaves=2048, lr=0.02 | n/a | n/a | 0.9936 | 0.9314 | 467 |
| v2 | full 39f | 2 | random_forest | trees=200 | n/r | n/r | 0.9933 | 0.9271 | 320 |
| v2 | full 39f | 2 | rf_balanced | trees=100, w=balanced_subsample | n/r | n/r | 0.9928 | 0.9219 | 284 |
| v2 | full 39f | 2 | xgboost_balanced_deep | depth=14, lr=0.05, sqrt class weights | 0.9927 | 0.9268 | 0.9927 | 0.9272 | 535 |
| v2 | full 39f | 2 | xgboost_deep | depth=16, lr=0.03 | 0.9935 | 0.9302 | 0.9936 | 0.9313 | 246 |
| v2 | full 39f | 2 | xgboost_trainval | leaves=1024, lr=0.05 | n/a | n/a | 0.9936 | 0.9314 | 1,901 |
| v2 | full 39f | 8 | random_forest | trees=200 | n/r | n/r | 0.8587 | 0.6978 | 768 |
| v2 | full 39f | 8 | rf_balanced | trees=100, w=balanced_subsample | n/r | n/r | 0.7759 | 0.6998 | 664 |
| v2 | full 39f | 8 | xgboost_balanced_deep | depth=14, lr=0.05, sqrt class weights | 0.8419 | 0.7295 | 0.8415 | 0.7308 | 1,414 |
| v2 | full 39f | 8 | xgboost_deep | depth=16, lr=0.03 | 0.8653 | 0.7156 | 0.8650 | 0.7159 | 2,054 |
| v2 | full 39f | 8 | xgboost_trainval | leaves=1024, lr=0.05 | n/a | n/a | 0.8649 | 0.7172 | 6,675 |
| v2 | full 39f | 34 | random_forest | trees=200 | n/r | n/r | 0.7891 | 0.6233 | 736 |
| v2 | full 39f | 34 | rf_balanced | trees=100, w=balanced_subsample | n/r | n/r | 0.7737 | 0.6244 | 661 |
| v2 | full 39f | 34 | xgboost_balanced_deep | depth=14, lr=0.05, sqrt class weights | 0.7945 | 0.6401 | 0.7944 | 0.6420 | 4,137 |
| v2 | full 39f | 34 | xgboost_balanced_p025 | depth=12, lr=0.05, w-power=0.25 | n/r | n/r | 0.7974 | 0.6412 | 7,419 |
| v2 | full 39f | 34 | xgboost_balanced_p050 | depth=12, lr=0.05, w-power=0.5 | n/r | 0.6412 | 0.7946 | 0.6429 | 4,440 |
| v2 | full 39f | 34 | xgboost_deep | depth=16, lr=0.03 | 0.7977 | 0.6345 | 0.7977 | 0.6379 | 5,643 |
| v2 | full 39f | 34 | xgboost_deep_balanced | depth=16, lr=0.03, w-power=0.5 | n/r | n/r | 0.7941 | 0.6416 | 4,589 |
| v2 | full 39f | 34 | xgboost_trainval | leaves=1024, lr=0.05 | n/a | n/a | 0.7980 | 0.6377 | 12,618 |
| v3 | full 39f | 2 | catboost_c1_bal_s42 | depth=10, lr=0.08, iters=1500, w=bal | 0.9926 | 0.9207 | 0.9926 | 0.9213 | 14 |
| v3 | full 39f | 2 | catboost_c1_none_s1 | depth=10, lr=0.08, iters=1500, w=none | 0.9928 | 0.9230 | 0.9928 | 0.9232 | 22 |
| v3 | full 39f | 2 | catboost_c1_none_s2 | depth=10, lr=0.08, iters=1500, w=none | 0.9927 | 0.9213 | 0.9927 | 0.9219 | 13 |
| v3 | full 39f | 2 | catboost_c1_none_s42 | depth=10, lr=0.08, iters=1500, w=none | 0.9927 | 0.9212 | 0.9927 | 0.9217 | 14 |
| v3 | full 39f | 2 | catboost_c2_bal_s42 | depth=8, lr=0.05, iters=2000, w=bal | 0.9924 | 0.9185 | 0.9925 | 0.9192 | 11 |
| v3 | full 39f | 2 | catboost_c2_none_s42 | depth=8, lr=0.05, iters=2000, w=none | 0.9926 | 0.9201 | 0.9926 | 0.9207 | 14 |
| v3 | full 39f | 34 | catboost_c1_bal_s42 | depth=10, lr=0.08, iters=1500, w=bal | 0.7687 | 0.5010 | 0.7688 | 0.5006 | 42 |
| v3 | full 39f | 34 | catboost_c1_none_s1 | depth=10, lr=0.08, iters=1500, w=none | 0.7746 | 0.5411 | 0.7749 | 0.5415 | 41 |
| v3 | full 39f | 34 | catboost_c1_none_s2 | depth=10, lr=0.08, iters=1500, w=none | 0.7746 | 0.5415 | 0.7749 | 0.5417 | 39 |
| v3 | full 39f | 34 | catboost_c1_none_s42 | depth=10, lr=0.08, iters=1500, w=none | 0.7747 | 0.5395 | 0.7750 | 0.5393 | 41 |
| v3 | full 39f | 34 | catboost_c2_bal_s42 | depth=8, lr=0.05, iters=2000, w=bal | 0.7637 | 0.4845 | 0.7638 | 0.4840 | 51 |
| v3 | full 39f | 34 | catboost_c2_none_s42 | depth=8, lr=0.05, iters=2000, w=none | 0.7727 | 0.5256 | 0.7730 | 0.5261 | 37 |
| v3 | full 39f | 34 | catboost_c2_none_s42_b1024 | depth=8, lr=0.05, iters=2000, w=none | 0.7715 | 0.5199 | 0.7717 | 0.5202 | 48 |
| v3 | full 39f | 34 | hierarchical_8to34_hard_s1 | stage1 xgb cfg3_weighted + 6 family experts | 0.7717 | 0.6304 | 0.7714 | 0.6352 | 38 |
| v3 | full 39f | 34 | hierarchical_8to34_hard_s2 | stage1 xgb cfg3_weighted + 6 family experts | 0.7718 | 0.6296 | 0.7715 | 0.6351 | 37 |
| v3 | full 39f | 34 | hierarchical_8to34_hard_s42 | stage1 xgb cfg3_weighted + 6 family experts | 0.7717 | 0.6317 | 0.7714 | 0.6354 | 36 |
| v3 | full 39f | 34 | xgboost_focal_s42 | depth=12, lr=0.05, focal-gamma=2.0, w=sqrt-bal | 0.6595 | 0.5160 | 0.6597 | 0.5152 | 2,947 |
| v4 | Kaggle | 2 | rf [38f-noIAT] | trees=200 | 0.9944 | 0.9404 | 0.9945 | 0.9417 | 181 |
| v4 | Kaggle | 2 | rf [38f] | trees=200 | 0.9971 | 0.9685 | 0.9972 | 0.9696 | 154 |
| v4 | Kaggle | 2 | rf [46f-noIAT] | trees=200 | 0.9954 | 0.9513 | 0.9955 | 0.9523 | 186 |
| v4 | Kaggle | 2 | rf [46f] | trees=200 | 0.9976 | 0.9744 | 0.9977 | 0.9754 | 177 |
| v4 | Kaggle | 2 | xgb_best [38f-noIAT] | leaves=1024, lr=0.05 | 0.9946 | 0.9429 | 0.9947 | 0.9436 | 251 |
| v4 | Kaggle | 2 | xgb_best [38f] | leaves=1024, lr=0.05 | 0.9965 | 0.9632 | 0.9967 | 0.9652 | 44 |
| v4 | Kaggle | 2 | xgb_best [46f-noIAT] | leaves=1024, lr=0.05 | 0.9959 | 0.9566 | 0.9959 | 0.9562 | 284 |
| v4 | Kaggle | 2 | xgb_best [46f] | leaves=1024, lr=0.05 | 0.9974 | 0.9720 | 0.9974 | 0.9723 | 44 |
| v4 | Kaggle | 2 | xgb_unw [38f] | depth=12, lr=0.1 | 0.9963 | 0.9602 | 0.9964 | 0.9618 | 9 |
| v4 | Kaggle | 2 | xgb_unw [46f] | depth=12, lr=0.1 | 0.9972 | 0.9696 | 0.9972 | 0.9696 | 12 |
| v4 | Kaggle | 8 | rf [38f-noIAT] | trees=200 | 0.9064 | 0.7206 | 0.9061 | 0.7074 | 339 |
| v4 | Kaggle | 8 | rf [38f] | trees=200 | 0.9959 | 0.8806 | 0.9960 | 0.8758 | 228 |
| v4 | Kaggle | 8 | rf [46f-noIAT] | trees=200 | 0.9170 | 0.7306 | 0.9168 | 0.7193 | 389 |
| v4 | Kaggle | 8 | rf [46f] | trees=200 | 0.9962 | 0.8639 | 0.9963 | 0.8681 | 248 |
| v4 | Kaggle | 8 | xgb_best [38f-noIAT] | leaves=1024, lr=0.05, w-power=0.5 | 0.9075 | 0.7419 | 0.9076 | 0.7346 | 934 |
| v4 | Kaggle | 8 | xgb_best [38f] | leaves=1024, lr=0.05, w-power=0.5 | 0.9962 | 0.8823 | 0.9961 | 0.8799 | 207 |
| v4 | Kaggle | 8 | xgb_best [46f-noIAT] | leaves=1024, lr=0.05, w-power=0.5 | 0.9156 | 0.7603 | 0.9154 | 0.7547 | 926 |
| v4 | Kaggle | 8 | xgb_best [46f] | leaves=1024, lr=0.05, w-power=0.5 | 0.9968 | 0.8967 | 0.9967 | 0.8925 | 210 |
| v4 | Kaggle | 8 | xgb_unw [38f] | leaves=1024, lr=0.05 | 0.9953 | 0.8219 | 0.9954 | 0.8138 | 232 |
| v4 | Kaggle | 8 | xgb_unw [46f] | leaves=1024, lr=0.05 | 0.9960 | 0.8381 | 0.9960 | 0.8254 | 235 |
| v4 | Kaggle | 34 | rf [38f-noIAT] | trees=200 | 0.8844 | 0.6795 | 0.8842 | 0.6830 | 359 |
| v4 | Kaggle | 34 | rf [38f] | trees=200 | 0.9947 | 0.8082 | 0.9948 | 0.8011 | 249 |
| v4 | Kaggle | 34 | rf [46f-noIAT] | trees=200 | 0.8951 | 0.6951 | 0.8950 | 0.6959 | 380 |
| v4 | Kaggle | 34 | rf [46f] | trees=200 | 0.9949 | 0.7901 | 0.9950 | 0.7962 | 292 |
| v4 | Kaggle | 34 | xgb_best [38f-noIAT] | depth=12, lr=0.05, w-power=0.5 | 0.8900 | 0.6969 | 0.8900 | 0.6979 | 358 |
| v4 | Kaggle | 34 | xgb_best [38f] | depth=12, lr=0.05, w-power=0.5 | 0.9951 | 0.8291 | 0.9951 | 0.8361 | 293 |
| v4 | Kaggle | 34 | xgb_best [46f-noIAT] | depth=12, lr=0.05, w-power=0.5 | 0.8973 | 0.7153 | 0.8976 | 0.7143 | 409 |
| v4 | Kaggle | 34 | xgb_best [46f] | depth=12, lr=0.05, w-power=0.5 | 0.9957 | 0.8406 | 0.9957 | 0.8484 | 332 |
| v4 | Kaggle | 34 | xgb_unw [38f] | depth=12, lr=0.05 | 0.9940 | 0.7780 | 0.9941 | 0.7819 | 349 |
| v4 | Kaggle | 34 | xgb_unw [46f] | depth=12, lr=0.05 | 0.9946 | 0.7926 | 0.9947 | 0.7907 | 363 |
