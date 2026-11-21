**v4 ablation on the same rows of the Kaggle subset (80/20 + 5% val, seed 42). Cells: test accuracy / test macro-F1 of the model selected on validation; BEAT/NOT against the paper on the judged metric.**

| Task (judged on) | 38 feat. | 46 feat. | 38 feat., no IAT | 46 feat., no IAT | Paper RF |
|---|---|---|---|---|---|
| 2-class (acc) | 0.9972 / 0.9696 (BEAT) | 0.9977 / 0.9754 (BEAT) | 0.9947 / 0.9436 (NOT) | 0.9959 / 0.9562 (NOT) | 0.9968 / 0.9653 |
| 8-class (mF1) | 0.9961 / 0.8799 (BEAT) | 0.9967 / 0.8925 (BEAT) | 0.9076 / 0.7346 (BEAT) | 0.9154 / 0.7547 (BEAT) | 0.9944 / 0.7193 |
| 34-class (mF1) | 0.9951 / 0.8361 (BEAT) | 0.9957 / 0.8484 (BEAT) | 0.8900 / 0.6979 (NOT) | 0.8976 / 0.7143 (BEAT) | 0.9916 / 0.7140 |
