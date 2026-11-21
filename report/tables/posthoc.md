**Post-hoc methods on the full release (threshold tuning A1, calibration A2, ensembles and stacking A3). The method is chosen by validation macro-F1 only.**

| Ver | Task | Method | Val mF1 (selection) | Test acc | Test mF1 |
|---|---|---|---|---|---|
| v2 | 2 | LightGBM cfg2 + threshold 0.49 | n/a | 0.9937 | 0.9324 |
| v2 | 2 | Soft-vote ensemble (3 models) | n/a | 0.9938 | 0.9332 |
| v3 | 2 | baseline | 0.9323 | 0.9938 | 0.9328 |
| v3 | 2 | A1_threshold | 0.9328 | 0.9937 | 0.9333 |
| v3 | 2 | A2_isotonic | 0.9315 | 0.9938 | 0.9329 |
| v3 | 2 | A2_sigmoid | 0.9253 | 0.9934 | 0.9257 |
| v3 | 2 | A3_soft | 0.9316 | 0.9937 | 0.9322 |
| v3 | 2 | A3_weighted | 0.9316 | 0.9937 | 0.9322 |
| v3 | 2 | A3_weighted+A1 | 0.9320 | 0.9936 | 0.9328 |
| v3 | 2 | A3_stack | 0.9325 | 0.9938 | 0.9334 |
| v3 | 2 | A3_stack+A1 (selected by val) | 0.9332 | 0.9937 | 0.9339 |
| v3 | 34 | baseline | 0.6412 | 0.7946 | 0.6429 |
| v3 | 34 | A1_threshold (selected by val) | 0.6431 | 0.7895 | 0.6452 |
| v3 | 34 | A2_isotonic | 0.6361 | 0.7980 | 0.6380 |
| v3 | 34 | A2_sigmoid | 0.6369 | 0.7980 | 0.6386 |
| v3 | 34 | A3_soft | 0.6394 | 0.7955 | 0.6416 |
| v3 | 34 | A3_weighted | 0.6402 | 0.7953 | 0.6426 |
| v3 | 34 | A3_weighted+A1 | 0.6423 | 0.7888 | 0.6461 |
| v3 | 34 | A3_stack | 0.6367 | 0.7981 | 0.6398 |
| v3 | 34 | A3_stack+A1 | 0.6423 | 0.7901 | 0.6480 |
