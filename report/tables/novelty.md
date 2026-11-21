**Summary of the analyses N1-N6 with their evidence files.**

| Analysis | Measure | Value | Evidence (JSON) |
|---|---|---|---|
| N1 twins | test rows with an exact train twin | 59.05% (full) / 5.61% (v4) | results/novelty/N1_twin_analysis/results.json; results/v4_46feat/summary.json |
| N2 protocol ladder | 8-class mF1: paper protocol → file-grouped (twin-free) | 0.7308 → 0.4454 (0.4557) | results/novelty/N2_protocol_ladder/results.json |
| N3 DoS vs DDoS | share of 8-class test errors (full / v4 no-IAT) | 95.18% / 92.82% | results/novelty/v3/dos_ddos.json; results/v4_46feat/summary.json |
| N3 merge | 8-class acc → 7-class (DoS+DDoS merged) acc | 0.8410 → 0.9923 | results/novelty/v3/dos_ddos.json |
| N4 IAT | IAT-only depth-8 tree, 8-class val acc (majority baseline) | 0.9905 (0.7278) | results/v4_46feat/iat_probe.json |
| N5 window | capture-window classifier accuracy | 99.9997% | results/novelty/N5_window_size/results.json |
| N6 hierarchical | 34-class test mF1, hierarchical (3 seeds) vs flat | 0.6352 vs 0.6429 | results/novelty/v3/hierarchical_s*.json |
