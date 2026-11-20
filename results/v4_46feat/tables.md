# v4 tables (test acc / test macro-F1; model per cell selected on validation)

## All features vs our 38

| Task | Judged on | v3: full release, 39f (75/5/20) | v4 38 features | v4 46 features | Neto et al. 2023 RF | 38f vs paper | 46f vs paper |
|---|---|---|---|---|---|---|---|
| 2-class | accuracy | 0.9937 / 0.9339 | 0.9972 / 0.9696 (rf) | 0.9977 / 0.9754 (rf) | 0.9968 / 0.9653 | **BEAT** | **BEAT** |
| 8-class | macro-F1 | 0.8410 / 0.7324 | 0.9961 / 0.8799 (xgb_best) | 0.9967 / 0.8925 (xgb_best) | 0.9944 / 0.7193 | **BEAT** | **BEAT** |
| 34-class | macro-F1 | 0.7895 / 0.6452 | 0.9951 / 0.8361 (xgb_best) | 0.9957 / 0.8484 (xgb_best) | 0.9916 / 0.7140 | **BEAT** | **BEAT** |

## Without IAT (capture-timestamp proxy removed)

| Task | Judged on | v3: full release, 39f (75/5/20) | v4 38 features, no IAT | v4 46 features, no IAT | Neto et al. 2023 RF | 38nf vs paper | 46nf vs paper |
|---|---|---|---|---|---|---|---|
| 2-class | accuracy | 0.9937 / 0.9339 | 0.9947 / 0.9436 (xgb_best) | 0.9959 / 0.9562 (xgb_best) | 0.9968 / 0.9653 | NOT BEAT | NOT BEAT |
| 8-class | macro-F1 | 0.8410 / 0.7324 | 0.9076 / 0.7346 (xgb_best) | 0.9154 / 0.7547 (xgb_best) | 0.9944 / 0.7193 | **BEAT** | **BEAT** |
| 34-class | macro-F1 | 0.7895 / 0.6452 | 0.8900 / 0.6979 (xgb_best) | 0.8976 / 0.7143 (xgb_best) | 0.9916 / 0.7140 | NOT BEAT | **BEAT** |

## DoS vs DDoS confusions (8-class test set; ✓ = validation-selected model)

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

## All runs

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
