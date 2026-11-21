**N2: protocol ladder on the full release.**

| Protocol (8-class, XGBoost balanced_deep) | Test acc | Test mF1 |
|---|---|---|
| (a) paper protocol: shuffled rows, no de-dup | 0.8415 | 0.7308 |
| (b) + de-duplication | 0.8572 | 0.7297 |
| (c) file-grouped split | 0.7455 | 0.4454 |
| (d) file-grouped + twin-free | 0.7217 | 0.4557 |
