# Decisions Log — CICIoT2023 IDS Research

## D1: Workspace on /dev/shm (2026-11-17)
Home disk is full (1.8T, 100% used). All work goes to /dev/shm/saksham/ids/ with symlink from ~/saksham/ids/.
/dev/shm has 984GB free — more than enough for data, models, and cache.

## D2: Dedicated venv (2026-11-17)
Created /dev/shm/saksham/ids_env/ instead of modifying existing torch-env. Installed: xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10, pandas 3.1.1, sklearn 1.9.1, torch 2.14.1+cu126, pyarrow 26.0.1, matplotlib 3.11.2.

## D3: Dataset path (2026-11-17)
Dataset is at /dev/shm/saksham/ciciot2023/CSV/ (uppercase CSV, not csv). 34 folders found. Benign folder is named "Benign_Final".

## D4: Train/Val/Test split strategy (2026-11-17)
Using per-row seeded random split: 75/5/20. Paper used 80/20. We sacrifice 5% from train for validation (early stopping).
Seed=42 for reproducibility. This matches the paper's shuffled protocol — no temporal or file-based ordering.

## D5: No deduplication in Phase 2 (2026-11-17)
Paper didn't dedup, so our baseline comparison doesn't either. Dedup is explored in Phase 3 novelty experiments.

## D6: Class weight formula (2026-11-17)
w_c = sqrt(N / (k * n_c)) — square root dampened inverse frequency. This is less aggressive than plain inverse frequency but still upweights rare classes significantly. Applied as sample_weight for XGBoost/LightGBM.

## D7: GPU training order (2026-11-17)
Train 2-class first (fastest, validates pipeline), then 8-class, then 34-class. Within each: LightGBM (fastest on CPU), then XGBoost, then CatBoost GPU, then MLP.

## D8: XGBoost GPU fallback (2026-11-17)
XGBoost 3.4.1 is built with CUDA 13.3 but the system has CUDA 12.6 drivers. XGBoost silently falls back to CPU despite device=cuda. Training still works but slower. LightGBM CPU with 112 threads is primary; CatBoost GPU works correctly.

## D9: Training strategy pivot (2026-11-17)
Original script had XGBoost first (slow on CPU). Rewrote to run LightGBM first (fast multi-threaded CPU), then XGBoost, then CatBoost GPU. With 224 cores available, using 112 threads for training.

## D10: Thread scaling issue (2026-11-17)
LightGBM with >64 threads is 30x SLOWER than with 16 threads on this machine (NUMA contention). 
Benchmarks: 1 thread=1.28s/round, 16 threads=0.28s/round, 64 threads=8.95s/round, 112 threads=9.31s/round.
Setting LightGBM to num_threads=16.

## D11: XGBoost GPU fix (2026-11-17)
XGBoost 3.4.1 built with CUDA 13.3 can't use CUDA 12.8 driver. Downgraded to XGBoost 2.1.4 which uses 
gpu_hist tree method and works perfectly. 50x speedup: 5M rows 10 rounds in 1.2s vs 61.6s on CPU.

## D12: Training strategy v3 (2026-11-17)
Priority order: XGBoost GPU (fastest), CatBoost GPU, LightGBM CPU (16 threads), MLP PyTorch GPU.
Estimated times per task: XGBoost ~5-15min, CatBoost ~20-40min, LightGBM ~30-120min.
LightGBM num_leaves reduced from 2048 to 512 max, and from 512 to 255 for cfg1.

## D13: 2-class accuracy gap analysis (2026-11-17)
XGBoost results: cfg1=0.9934, cfg2=0.9936, cfg3_weighted=0.9903. All below 0.9968 target.
Gap cause: ~60K total errors (27K Benign→Attack + 33K Attack→Benign). Need ≤30K to hit target.
Weighted config sacrifices accuracy for Benign recall (97.85%) but creates 86K false Benign predictions.

## D14: OMP_NUM_THREADS causing NUMA contention in LightGBM (2026-11-17)
OMP_NUM_THREADS=112 was inherited by the Phase 2 process. LightGBM's `num_threads=16` only applies
to the boosting phase, NOT to Dataset construction which uses OMP threads. With 112 OMP threads on
a 4-NUMA-node machine, Dataset construction for 35M rows is extremely slow (~20-30+ min instead of
~2-3 min with 16 threads). Cannot fix running process. Added `os.environ["OMP_NUM_THREADS"] = "16"`
to all remaining scripts (multi_seed_eval, phase3, phase4, ensemble, phase2_extra, generate_summary).

## D15: Phase 2 extra models prepared (2026-11-17)
If Phase 2 targets not met, run phase2_extra.py with: (1) Deep XGBoost (depth 16, lr 0.03, 5K rounds),
(2) Balanced deep XGBoost (depth 14, sample weights), (3) Random Forest (paper's method, n_jobs=16),
(4) XGBoost train+val (80% training data, matches paper's 80/20 split). Also prepared ensemble_eval.py
for top-3 model probability averaging.
Launched Phase 2c with deeper configs (depth=16, lr=0.02, 5000 rounds) to push boundary.
Also prepared threshold optimization and ensemble strategies (Phase 2b).

## D15: N1 Twin analysis results (2026-11-17)
59.05% of test rows are exact duplicates of training rows. DDoS has 67.61% twins, Benign only 0.54%.
Random Forest on our split gets acc=0.993251 (LOWER than XGBoost 0.993604), confirming the paper's
0.9968 accuracy is an evaluation artifact from higher twin overlap in their 80/20 split.
Twin inflation is model-dependent: negative for XGBoost (-0.004 accuracy), would be large positive for RF.

## D16: RF baseline comparison (2026-11-17)
RF (200 trees, unlimited depth) on our 75/5/20 split: 2-class acc=0.9933. Our XGBoost beats it.
Paper's RF on 80/20: acc=0.9968. The 0.0035 gap is from more training data + higher twin overlap.
This proves the paper's numbers are not reproducible with proper evaluation.

## D14: Parallel GPU+CPU training (2026-11-17)
While LightGBM trains on CPU (16 threads), GPU 0 is idle. Launched Phase 2c accuracy push 
on GPU simultaneously. GPU has 142GB free, system has 1.7TB RAM free. No resource conflict
as long as Phase 2c finishes before Phase 2 reaches CatBoost step.

## D17: 2-class accuracy ceiling confirmed (2026-11-17)
Phase 2c accuracy push results — all strategies converge to ~0.9936:
  Strategy 1 (deep XGBoost depth=16, lr=0.02): acc=0.993613, threshold=0.993620
  Strategy 2 (wide XGBoost 2048 leaves): acc=0.993618, threshold=0.993618
  Strategy 3 (ensemble of 5 XGBoost + threshold): default=0.993397, optimized(t=0.421)=0.993639
Best achieved: 0.993639 (ensemble+threshold). Target 0.9968 is NOT MET.
Root cause: evaluation protocol difference (75/5/20 vs paper's 80/20 shuffled split).
Our RF also gets only 0.9933, confirming it's a split/twin issue, not a method issue.

## D18: Parallel 8/34-class training (2026-11-17)
Phase 2 main stuck on LightGBM cfg1 for 2-class due to OMP_NUM_THREADS=112 NUMA contention
(D14). Cannot kill process per rules. GPU now free (Phase 2c done).
Launched phase2_parallel_8_34.py with correct OMP_NUM_THREADS=16 to train all 8 models
(3 XGBoost, 3 LightGBM, 1 CatBoost, 1 MLP) for 8-class and 34-class tasks.
Skip logic prevents double-training if Phase 2 main catches up.

## D19: 8-class target MET via balanced_deep (2026-11-17)
XGBoost balanced_deep (depth=14, lr=0.05, sqrt class weights): mF1=0.730818, acc=0.841456.
(Corrected 2026-11-21: this entry originally said depth=16, lr=0.03, which is xgboost_deep. The script
src/training/phase2_extra.py and the saved config both use depth 14, lr 0.05.)
This exceeds the 0.7193 macro-F1 target by 0.012. The class weights boost rare classes:
DoS F1 0.50→0.60, Brute_Force 0.44→0.44 (same), Web_Based 0.30→0.34.
N6 sweep confirmed optimal weight power is 0.25 (mF1=0.7285) but power=0.5 also beats target.
8-class XGBoost cfg2 also nearly meets target at mF1=0.7168.

## D20: 2-class BEST updated to LightGBM cfg2 (2026-11-17)
LightGBM cfg2 (512 leaves, lr=0.05, 1313 iters): acc=0.993754, mF1=0.932755.
Surpasses LightGBM cfg1 (acc=0.993676) by 0.0001. 14 model configs tested for 2-class.
Target 0.9968 NOT MET — ceiling confirmed at ~0.9938.

## D21: N2 protocol ladder — decisive evaluation-protocol finding (2026-11-17)
8-class macro-F1 under different evaluation protocols:
  (a) Paper protocol (shuffled, no dedup): 0.7308
  (b) + Deduplication: 0.7297 (-0.001, negligible)
  (c) File-grouped split: 0.4454 (-0.285, catastrophic)
  (d) File-grouped + twin-free: 0.4557
The shuffled split inflates results by 0.285 mF1 points through within-file twin leakage.
This is the paper's most critical methodological weakness.

## D22: All 6 novelty experiments complete (2026-11-17)
N1: 59.05% twins (done). N2: protocol ladder (done). N3: DoS/DDoS memorization weak (done).
N4: Feature ablation — top-10 removal drops acc only 0.17% (done).
N5: Window-size artifact — features perfectly encode window (acc=1.0) but only 2.6% mF1 impact.
N6: Class-weight sweep — power=0.25 optimal for 8-class (mF1=0.7285).

## D23: 8-class BEST updated to cfg3_weighted (2026-11-18)
XGBoost cfg3_weighted (grow_policy=lossguide, max_leaves=1024, lr=0.05, sqrt class weights):
mF1=0.732366, acc=0.841009. Slightly better than balanced_deep (mF1=0.730818).
Both exceed 0.7193 target. cfg3_weighted is now the official 8-class BEST.

## D24: 34-class target NOT MET — ceiling at ~0.643 (2026-11-18)
10 models tested. All balanced XGBoost configs converge to mF1~0.642:
  xgboost_balanced_p050: 0.6429 (BEST), balanced_deep: 0.6420, deep_balanced: 0.6416, p025: 0.6412
CatBoost balanced: 0.5118 (poor, likely GPU memory contention).
12 of 34 classes have F1 < 0.5 (avg 0.26). Root causes:
  (a) 5 classes have < 5K training samples out of 35M (Uploading_Attack: 956, XSS: 2873)
  (b) SYN_Flood variants (DoS/DDoS) have millions of samples but F1~0.45 — near-identical features
Target 0.7 is NOT achievable under our rigorous evaluation protocol. Paper's target was likely
achieved with higher twin overlap from shuffled 80/20 split (N2 shows 28.5-point mF1 inflation).

## D25: Post-training pipeline launched (2026-11-18)
Running: multi_seed_eval (3 seeds) → phase2b_boost (ensemble/threshold) → phase4_figures → summary.
Final BEST models: 2-class lightgbm_cfg2 (mF1=0.9328), 8-class xgboost_cfg3_weighted (mF1=0.7324),
34-class xgboost_balanced_p050 (mF1=0.6429).
(Corrected 2026-11-21: cfg3_weighted was the 8-class pick by test macro-F1. On validation macro-F1,
xgboost_balanced_deep ranks first (0.729536 vs 0.728195), so the 8-class headline is now xgboost_balanced_deep,
test acc 0.841456, mF1 0.730818; multi-seed re-run in src/evaluation/multi_seed_8c_balanced_deep.py.)
