<div class="titlepage" markdown="1">

# Intrusion Detection and Prevention for IoT Networks on CICIoT2023

## Beating the benchmark honestly: leakage, timestamp proxies and a deployable prevention layer

**Capstone Project Report**

**Saksham Kala**

Enrolment No. E23CSEU1027

B.Tech Computer Science and Engineering (2023–2027)

Bennett University, Greater Noida

November 2026

</div>

<div class="pagebreak"></div>

## Abstract

IoT devices are easy targets for attacks such as floods, botnets and scanning. A machine-learning intrusion
detection system (IDS) learns to recognise these attacks from network-flow features. CICIoT2023 is a large public
benchmark for this task, and its authors (Neto et al., 2023) report Random Forest accuracies close to 99%.

In this project I trained {{nModels}} models (Random Forest, LightGBM, XGBoost, CatBoost, an MLP, focal-loss
XGBoost and a hierarchical classifier) and tried {{nPosthoc}} post-hoc variants (threshold tuning, calibration,
ensembles), always choosing models on a validation split and never on the test split.

- On the **full 39-feature release** ({{fullRows}} rows) I beat the paper's 8-class macro-F1
  ({{fullBestMfEight}} vs {{paperMfEight}}) but not its 2-class accuracy ({{fullBestAccTwo}} vs {{paperAccTwo}})
  or 34-class macro-F1 ({{fullBestMfThirtyFour}} vs {{paperMfThirtyFour}}).
- On a **46-feature subset** that matches the paper's feature table I beat all three targets
  ({{vfBAccTwo}}, {{vfBMfEight}}, {{vfBMfThirtyFour}}). However, one feature, **IAT**, behaves like a capture
  timestamp: a small tree on IAT alone gets {{iatAcc}} 8-class accuracy. Without IAT the 2-class target is not met,
  8-class macro-F1 is {{vfDMfEight}} and 34-class macro-F1 ties the paper ({{vfDMfThirtyFour}} vs {{paperMfThirtyFour}}).
- The paper-style shuffled split has **{{nOneTwin}} duplicate rows** between train and test, and a
  file-grouped split lowers 8-class macro-F1 by {{nTwoDrop}}. **DoS vs DDoS** confusion causes {{nThreeShare}}
  of 8-class errors.
- Finally, I built an **intrusion prevention layer (IDPS)** on the no-IAT model. It chooses the right action
  for {{idsActAcc}} of {{idsRows}} unseen flows, blocks {{idsFalseBlock}} of benign flows, misses {{idsMiss}} of
  attacks and needs {{idsLatSingle}} ms per flow on a CPU.

<div class="pagebreak"></div>

## 1. Introduction

The Internet of Things (IoT) connects cameras, plugs, sensors and speakers to home and office networks. These
devices often have weak passwords and old software, so attackers use them to launch floods (DoS and DDoS), build
botnets (Mirai), scan networks (reconnaissance), fake identities (spoofing) and attack web services.

An **intrusion detection system (IDS)** watches the traffic and raises an alarm when it sees an attack. A
machine-learning IDS learns this from labelled examples. Each example is a *flow*: a short window of packets
summarised by numbers such as packet sizes, flag counts and rates.

**CICIoT2023** (Neto et al., 2023) is one of the largest public datasets for this. It was captured from 105 real
IoT devices and covers 33 attacks plus benign traffic. Labels come at three levels of detail:

- **2-class:** benign or attack
- **8-class:** Benign, DDoS, DoS, Mirai, Recon, Spoofing, Web-based, Brute-force
- **34-class:** each individual attack

The authors' Random Forest results are the targets for this project:

| Task | Paper accuracy | Paper F1 |
|---|---|---|
| 2-class | {{paperAccTwo}} | {{paperMfTwo}} |
| 8-class | {{paperAccEight}} | {{paperMfEight}} |
| 34-class | {{paperAccThirtyFour}} | {{paperMfThirtyFour}} |

(Source: Neto et al. 2023, Table 6. The paper does not say how its F1 is averaged; I compare against my macro-F1.)

A **prevention system (IDPS)** goes one step further than an IDS: it also acts, for example by blocking or
rate-limiting the attacker.

## 2. Objectives

1. Reproduce and try to beat the published results for 2, 8 and 34 classes, **honestly**: the test set is never
   used to choose a model, a threshold or a feature.
2. Find out *why* the published numbers are hard to reproduce, and prove each finding with numbers and figures.
3. Check whether the 8 features missing from our dataset version explain the gap.
4. Build a working prevention layer (IDPS) on top of the best *deployable* model, with a live demo.

## 3. Dataset

### 3.1 Two versions of CICIoT2023

**Full release (39 features).** The public CSV release I used has 309 files (one or more per attack) in 34
folders, with **{{fullRows}} rows** and **{{fullFeat}} numeric features**. It is missing 8 of the 46 features
listed in the paper (flow_duration, Srate, Drate, urg_count, Magnitue, Radius, Covariance, Weight) and has one
extra (IGMP).

**46-feature subset (Kaggle).** I could not find a 46-feature copy of the full release. A `MERGED_CSV` copy I
checked also had only 39 features. So for the 46-feature experiments I used a public Kaggle subset
(`himadri07/ciciot2023`) with **{{vfRows}} rows** after merging its train, validation and test files. It is a
**subset**, not the full release, so numbers on the two versions are not directly comparable.

### 3.2 Class counts

{{table:class_counts}}

The data is very imbalanced: DDoS dominates, while Brute_Force and Web_Based have only a few thousand rows.
This is why I use **macro-F1** (the average F1 over classes, so every class counts equally) as the main metric
for 8 and 34 classes. Accuracy alone would mostly measure DDoS.

The full 34-class counts are in Appendix D.

## 4. Methodology

### 4.1 Splitting the data

- **Full release (v1–v3):** shuffled rows, split 75% train / 5% validation / 20% test with seed 42
  (train {{vtTrain}}, validation {{vtVal}}, test {{vtTest}} rows). No duplicates were removed, to match the
  paper.
- **Kaggle subset (v4):** the paper's protocol. Shuffle, split 80% train / 20% test with seed 42, then take 5% of
  the training part as validation (train {{vfTrain}}, validation {{vfVal}}, test {{vfTest}} rows).

The **validation split** is used for early stopping, model selection and tuning thresholds. The **test split**
is scored once per reported model and never used for any decision.

### 4.2 Preprocessing

- No missing or infinite values were found.
- In v4, a StandardScaler is fitted on the training part only and then applied to validation and test.
- Labels are mapped to 2, 8 and 34 classes with one fixed mapping file
  (`src/preprocessing/class_maps.json`).

### 4.3 Handling class imbalance

Rare classes get bigger sample weights: `w_c = (N / (k · n_c))^p`, where `n_c` is the class size and `k` the
number of classes. I use `p = 0.5` ("square-root weights") by default and test `p = 0`, `0.25` and `0.5` in N6.
CatBoost uses its built-in `Balanced` option, and one Random Forest uses `balanced_subsample`.

### 4.4 Models

| Model | Why it was tried |
|---|---|
| Random Forest (200 trees) | The paper's own method, for a direct comparison |
| LightGBM (CPU, 16 threads) | Fast, strong gradient boosting for tabular data |
| XGBoost 2.1.4 on GPU | Strong boosting; the GPU makes training on tens of millions of rows practical |
| CatBoost on GPU | A third boosting library with built-in class balancing |
| MLP (1024-512-256) | To check whether a neural network helps on tabular data |
| Focal-loss XGBoost | Focuses training on hard, rare examples |
| Hierarchical 8→34 | First predicts the family, then the exact attack |

**Post-hoc methods** (applied to already-trained models, tuned on validation):

- **A1 threshold tuning:** multiply each class's probability by a factor before taking the maximum.
- **A2 calibration:** isotonic and sigmoid (Platt) calibration.
- **A3 ensembles:** soft voting, weighted voting and a stacked meta-model.

### 4.5 Choosing the best model

- 2-class: highest **validation accuracy** (the paper's 2-class headline metric).
- 8 and 34 classes: highest **validation macro-F1**.
- The best full-release models were re-trained with seeds 42, 123 and 456 to check stability.

### 4.6 Tools and hardware

Python, scikit-learn 1.9.1, XGBoost 2.1.4, LightGBM 4.7.0, CatBoost 1.2.10 and PyTorch, on a DGX server with
NVIDIA H200 GPUs and 224 CPU cores. Large files were kept in `/dev/shm` (RAM disk) because the home disk was small.

## 5. Experiments in time order

### 5.1 v1: first benchmark (full release)

**What:** XGBoost and LightGBM with three configs each (cfg1: depth-wise; cfg2: wide and loss-guided; cfg3: cfg2
plus class weights), CatBoost and an MLP, for all three tasks. The training script went through three
engineering versions (`phase2_train.py`, `phase2_train_v2.py`, `phase2_train_v3.py`) that changed only the
hardware set-up and run order: XGBoost 3.x could not use the GPU with the installed driver, so I moved to 2.1.4,
and LightGBM became much slower with more than 64 threads because of NUMA effects, so I limited it to 16 threads.

**Result:** the 8-class target was met by XGBoost cfg3_weighted (macro-F1 {{fullBestMfEight}} vs
{{paperMfEight}}). 2-class accuracy and 34-class macro-F1 stayed below the targets (all scores are in
Appendix B).

### 5.2 v2: closing the gap and asking why (full release)

**What changed and why:** since 2-class and 34-class were short, I tried deeper and balanced XGBoost, the paper's
Random Forest, training on train+validation (to use 80% of the data like the paper), an "accuracy push" with
very deep or wide trees, threshold optimisation and a soft-vote ensemble. When none of these closed the gap, I
started asking whether the *evaluation* was the problem, which led to the analyses N1–N6 (Section 7) and a
3-seed check of the best models.

**Result:** every 2-class strategy converged to about the same accuracy (best single model
{{msAccTwo}} averaged over 3 seeds). The analyses showed heavy duplication and file-level leakage (N1, N2).

### 5.3 v3: new methods on top (full release)

**What:** post-hoc threshold tuning, calibration and stacking; CatBoost on GPU with several settings;
focal-loss XGBoost; and a hierarchical 8→34 classifier.

**Result:** post-hoc methods changed test macro-F1 only slightly (Table 6). The final 2-class result is
{{fullBestAccTwo}} accuracy and the final 34-class result is {{fullBestMfThirtyFour}} macro-F1. CatBoost, focal
and hierarchical models were all worse than flat XGBoost (N6).

### 5.4 v4: the paper's 46 features (Kaggle subset)

**What changed and why:** our release lacked 8 of the paper's features, so I tested whether they explain the
gap. On the same rows of the 46-feature subset, with the paper's 80/20 protocol, I trained each model with
38 features (the 46 minus the 8 paper-only ones) and with all 46: RF-200, XGBoost with the best earlier
configs, and one unweighted XGBoost per task. A single-feature check then showed that IAT alone almost solves
the task, so I added runs **without IAT** ({{nModelsVfour}} runs in total).

**Result:** with 46 features every target is beaten, but 38 features do almost as well, so the 8 features are
*not* the reason. The real reason is IAT (N4).

### 5.5 IDPS: from detection to prevention

**What:** a policy maps each 8-class prediction to an action, BLOCK actions generate firewall rules, and a
simulator and live demo run on unseen test flows (Section 8).

## 6. Results

### 6.1 Best model per task vs the paper

{{table:best_vs_paper}}

**How to read this:** on the full release I beat the paper only on 8-class. On the 46-feature subset all three
are beaten, but once IAT is removed, 2-class falls below the paper and 34-class only ties it.

### 6.2 v4 ablation: 38 vs 46 features, with and without IAT

{{table:v4_ablation}}

![v4 ablation](figures/fig_v4_ablation.png)

*Figure 1. v4 results on the same rows. Solid bars use IAT and hatched bars do not; the dashed line is the paper.
Source: `results/v4_46feat/summary.json`.*

### 6.3 Stability over seeds

{{table:multiseed}}

The standard deviations are tiny (at most {{msMfSdThirtyFour}}), so the differences between models are real
and not luck.

### 6.4 Post-hoc methods

{{table:posthoc}}

### 6.5 Confusion matrices and per-class F1

![8-class confusion](figures/fig_cm8_full.png)

*Figure 2. 8-class confusion matrix of the best full-release model (XGBoost cfg3_weighted). The big off-diagonal
cells are DDoS↔DoS. Source: `results/8_class/xgboost_cfg3_weighted/results.json`.*

![per-class F1](figures/fig_perclass_f1_8.png)

*Figure 3. Per-class F1, 8-class, full release. Brute_Force, DoS and Web_Based are the weak classes.*

![34-class confusion](figures/fig_cm34_full.png)

*Figure 4. 34-class confusion matrix of XGBoost balanced_p050, the single model with the highest test macro-F1
(decision D24; its validation score was not recorded, so this ranking is on test).
Source: `results/34_class/xgboost_balanced_p050/results.json`.*

![v4 8-class confusion](figures/fig_v4_cm8.png)

*Figure 5. v4 8-class confusion matrices with and without IAT. Without IAT, DoS↔DDoS confusion comes back.*

![training times](figures/fig_training_times.png)

*Figure 6. Training time per model (full release, v1/v2).*

The table of **every** trained model, with config, validation and test scores and training time, is in
Appendix B.

## 7. Novelties, each with its proof

{{table:novelty}}

### N1. Twin leakage: the test set contains copies of the training set

**Finding:** under the shuffled split, **{{nOneTwin}}** of full-release test rows have an exact duplicate
("twin") in the training set: {{nOneDDoS}} for DDoS but only {{nOneBenign}} for benign. In the v4 subset only
{{vfTwin}} of test rows have a twin.

**Proof:** `results/novelty/N1_twin_analysis/results.json`, `results/v4_46feat/summary.json`, Figure 7.

{{table:n1_twins}}

![twins](figures/fig_twins.png)

*Figure 7. Twin share per class and F1 on twin vs non-twin rows.*

### N2. Protocol ladder: the split decides the score

**Finding:** keeping the model fixed, 8-class macro-F1 is {{nTwoAMf}} with the paper protocol and
{{nTwoBMf}} after de-duplication, but drops to **{{nTwoCMf}}** when whole capture files are kept on one side
({{nTwoDMf}} if also twin-free). That is a fall of {{nTwoDrop}}.

**Proof:** `results/novelty/N2_protocol_ladder/results.json`, Figure 8.

{{table:n2_ladder}}

![ladder](figures/fig_protocol_ladder.png)

*Figure 8. 8-class macro-F1 and accuracy under four evaluation protocols.*

### N3. DoS vs DDoS is the bottleneck

**Finding:** **{{nThreeShare}}** of all 8-class test errors on the full release are DoS↔DDoS confusions.
Re-scoring the *same* predictions with the two merged raises accuracy from {{nThreeEightAcc}} to
**{{nThreeFloodAcc}}**. In v4 the confusion is {{vfDdWithShare}} of errors with IAT but **{{vfDdNoShare}}**
without it, and the 8 extra features do not fix it ({{vfDdNoA}} confusions with 38 features and no IAT).

**Why:** a flood from one source and a flood from many sources look the same flow by flow. The difference only
shows up in the overall traffic.

**Proof:** `results/novelty/v3/dos_ddos.json`, `results/v4_46feat/summary.json`, Figure 9.

![dos ddos](figures/fig_dosddos_merge.png)

*Figure 9. (a) Merging DoS and DDoS on the same predictions; (b) DoS↔DDoS share of errors.*

### N4. IAT acts as a capture timestamp

**Finding:** in the 46-feature subset, the median raw IAT is {{iatMedDoSTCP}} for DoS-TCP_Flood,
{{iatMedDDoSICMP}} for DDoS-ICMP_Flood and {{iatMedBenign}} for benign traffic, and each attack sits in its own
narrow band. These are not plausible "time since the previous packet" values; they behave like a clock reading
from when the attack was captured. A single depth-8 tree using **only IAT** reaches **{{iatAcc}}** 8-class
validation accuracy (majority class: {{iatMaj}}; next best feature, {{iatSecondName}}: {{iatSecondAcc}}). IAT
carries {{iatImpEight}} of the importance in the 46-feature 8-class model. Without IAT, 8-class accuracy falls
from {{vfBAccEight}} to {{vfDAccEight}}.

**Contrast:** on the full release, removing TTL and IAT changed 2-class accuracy by only {{nFourDrop}}, so the
proxy belongs to this data version.

**Proof:** `results/v4_46feat/iat_probe.json`, `results/v4_46feat/summary.json`, Figures 10–11.

{{table:iat_bands}}

![iat](figures/fig_iat_probe.png)

*Figure 10. (a) Accuracy of a one-feature tree for each feature; (b) IAT range per class.*

![importance](figures/fig_v4_importance.png)

*Figure 11. Top-15 feature importances of the 46-feature models (orange = IAT, blue = paper-only features).*

### N5. Window-size artifact

**Finding:** the dataset averages packets over windows of 10 or 100 packets. A classifier can tell which window
a row came from with **{{nFiveAcc}}** accuracy. Removing the window-related features changes 8-class
macro-F1 by {{nFiveMfChange}}.

**Proof:** `results/novelty/N5_window_size/results.json`.

### N6. Negative results: hierarchical, CatBoost, focal

**Finding:** the hierarchical 8→34 model reaches {{hierMf}} ± {{hierMfSd}} 34-class macro-F1 over 3 seeds,
below flat XGBoost ({{flatMfThirtyFour}}). Telling subtypes apart *within* a family is easy; the errors come from
choosing the wrong family. The best CatBoost GPU run gives {{cbBestMf}} and focal reweighting gives
{{focalMf}}. In the class-weight sweep, power {{nSixBestP}} is best for 8-class ({{nSixBestMf}}).

**Proof:** `results/novelty/v3/hierarchical_s{1,2,42}.json`, `results/34_class/v3/catboost/`,
`results/34_class/v3/focal/`, `results/novelty/N6_class_weight_sweep/results.json`, Figure 12.

![hier](figures/fig_hier_f1.png)

*Figure 12. 34-class per-class F1, flat vs hierarchical.*

### N7. The benchmark paper used IAT

**Finding:** Table 4 of Neto et al. (2023), "Features extracted from the network traffic", lists IAT and
describes it as *"The time difference with the previous packet"*. The paper's cleaning step says: *"In our
experiments, we only remove the timestamp from the list since it does not illustrate the network behavior"*.
So only the `ts` (timestamp) column was removed, and **IAT stayed in**. Given N4, the paper's ~99% accuracies
probably depend partly on this capture-time proxy.

Feature list quoted from the paper (Table 4, exactly as rendered in the PubMed Central copy, where three names
appear truncated):

> ts, flow duration, Header Length, Protocol Type, Duration, Rate, Srate, Drate, fin flag number, syn flag
> number, rst flag number, psh flag numbe [sic], ack flag number, ece flag numbe [sic], cwr flag number, ack
> count, syn count, fin count, urg coun [sic], rst count, HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP,
> DHCP, ARP, ICMP, IPv, LLC, Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Magnitude, Radius,
> Covariance, Variance, Weight
> — Neto et al. 2023, Table 4

**Note:** I could not run the authors' code, so this is an inference from their stated pipeline plus my
measurements.

## 8. Intrusion Prevention (IDPS)

### 8.1 Policy

{{table:idps_policy}}

DoS and DDoS share one action on purpose, since they are the pair the model confuses most. BLOCK writes an
`iptables` rate-limit rule (hashlimit, drop above 100 packets/second per source). BLOCK+ALERT writes a full drop
rule and raises an alert. ALERT only raises an alert.

### 8.2 Results on {{idsRows}} unseen test flows

| Measure | Value |
|---|---|
| Plain 8-class accuracy | {{idsEightAcc}} |
| **Action accuracy** (right action for the true class) | **{{idsActAcc}}** |
| Class errors that still got the right action | {{idsAbsorbed}} of {{idsErr}} ({{idsAbsorbedDd}} are DoS↔DDoS) |
| False block rate (benign flows blocked) | {{idsFalseBlock}} ({{idsFalseBlockN}} flows) |
| Miss rate (attack flows allowed) | {{idsMiss}} ({{idsMissN}} flows) |
| Latency, one flow at a time (CPU) | {{idsLatSingle}} ms average, {{idsLatSingleP}} ms p99 |
| Latency, batch of 1000 (per flow) | {{idsLatBatch}} ms average, {{idsLatBatchP}} ms p99 |

{{table:idps_actions}}

![idps](figures/fig_idps_actions.png)

*Figure 13. Correct action vs action taken. Source: `results/prevention/summary.json`.*

![idps taken](figures/fig_idps_taken.png)

*Figure 14. How often each action was taken, compared with how often it should have been.*

**Important notes:**

- **Blocking is simulated.** CICIoT2023 has no IP addresses, so rules use placeholder IPs from the
  documentation range 203.0.113.0/24. They are generated as text and never executed.
- **The deployed model excludes IAT** because of the timestamp leak (N4).

### 8.3 How to run the demo

```bash
cd src/prevention
python train_model.py 0     # only if /dev/shm/saksham/v4/prevention/ has no model (argument = GPU id)
python idps_simulator.py    # 200k-flow simulation -> results/prevention/, figures/prevention/
python demo.py              # live demo: one coloured line per flow + running counters
python demo.py --attacks-only --delay 0.3 --rows 200   # fewer DDoS rows, so rarer attacks show up
```

The demo runs on a CPU with only the packages in `requirements.txt`. It reads 2,000 test flows from
`results/prevention/demo_flows.csv` and the model from `/dev/shm/saksham/v4/prevention/` (change this with
`--model-dir`).

## 9. Discussion

A model that wins by reading IAT is really reading **when** the attack was recorded. On a real network, attacks
are not neatly separated into capture sessions, so that signal would not exist and the model would fail. The same
goes for duplicate rows (N1) and file-level patterns (N2, N5): they reward memorising the recording, not
recognising the attack.

That is why the **deployable model is the no-IAT model**. Its 8-class macro-F1 ({{vfDMfEight}}) looks lower than
the headline numbers, but it does not depend on capture timing. At the action level, where DoS and DDoS need the
same response anyway, it is accurate ({{idsActAcc}}).

## 10. Limitations and future work

**Limitations**

- The 46-feature results use a Kaggle **subset** ({{vfRows}} rows), not the full release, and how it was
  sampled is not documented.
- There are **no IP addresses** in CICIoT2023, so blocking is simulated.
- Everything uses **one dataset** from one testbed.
- A few early runs did not save validation metrics (Appendix C).
- The claim that the paper used IAT is **inferred** from its text, not from running its code.

**Future work**

- Re-extract features from the original packet captures with a correct IAT.
- Evaluate with file-grouped splits and on other datasets.
- Add multi-flow (aggregate) features to separate DoS from DDoS.
- Test the IDPS on live traffic with real addresses.

## 11. Conclusion

With honest, validation-only selection, I beat the paper's 8-class result on the full release and all three
results on the 46-feature subset. The bigger lesson is *why* the numbers move: shuffled splits are full of
duplicates, file-level grouping removes much of the score, and in the 46-feature data IAT works like a capture
timestamp. The model I would actually deploy (no IAT) ties or beats the paper only on the multi-class
macro-F1 targets, but in a prevention layer it picks the right action for {{idsActAcc}} of unseen flows in about
{{idsLatSingle}} ms per flow.

## References

1. E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani, "CICIoT2023: A Real-Time Dataset
   and Benchmark for Large-Scale Attacks in IoT Environment," *Sensors*, 23(13):5941, 2023. doi:10.3390/s23135941
2. A. Gueriani, H. Kheddar, A. C. Mazari, "Enhancing IoT Security with CNN and LSTM-Based Intrusion Detection
   Systems," arXiv:2405.18624, 2024.
3. A. Bensaoud, J. Kalita, "Optimized detection of cyber-attacks on IoT networks via hybrid deep learning
   models," *Ad Hoc Networks*, 170:103770, 2025.
4. I. Andrade, S. S. Mahadik, M. Mukherjee, P. M. Pawar, R. Muthalagu, "Deep learning based intelligent IDS for
   Large-scale IoT networks," arXiv:2603.16342, 2026.
5. G. Engelen, V. Rimmer, W. Joosen, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study,"
   IEEE Security and Privacy Workshops (SPW), pp. 7–12, 2021.
6. D. Arp et al., "Dos and Don'ts of Machine Learning in Computer Security," 31st USENIX Security Symposium,
   pp. 3971–3988, 2022.
7. Kaggle user himadri07, "CICIoT2023" (46-feature subset), https://www.kaggle.com/datasets/himadri07/ciciot2023
8. T. Chen, C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD, pp. 785–794, 2016.
9. G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," NeurIPS 30, pp. 3146–3154, 2017.
10. L. Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features," NeurIPS 31, 2018.
11. L. Breiman, "Random Forests," *Machine Learning*, 45(1):5–32, 2001.
12. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, 12:2825–2830, 2011.
13. T.-Y. Lin et al., "Focal Loss for Dense Object Detection," ICCV, pp. 2980–2988, 2017.

<div class="pagebreak"></div>

## Appendix A. Every result file

All paths are relative to the repository root (`capstone-cloud/`).

{{jsonlist}}

## Appendix B. Every trained model

{{table:all_models}}

## Appendix C. Values marked n/r

n/r = not recorded during the original run. These values were searched for in every result JSON, the training
logs (`logs/`, `ids/logs`, `ids/v3/logs`, `logs/v4`) and the decision log. They were never written: the scripts
for these runs scored only the test split, and the 2-class accuracy-push ensemble (which only averages already
trained models) was not timed.

{{table:not_recorded}}

## Appendix D. 34-class counts

{{table:class34_counts}}

## Appendix E. Notes on the records

- **34-class best model name.** `xgboost_balanced_p050` is the 34-class best. Phase 2e
  (`src/training/phase2e_34class_boost.py`, log `ids/logs/phase2e_34class.log`, 2026-11-18) compared all ten
  34-class models and wrote `xgboost_balanced_p050` to `results/34_class/BEST/`, and the multi-seed run used it.
  Later, the Phase 2 v3 run (`ids/logs/phase2_v3.log`, 2026-11-19 09:06) compared only its own eight models and
  overwrote `BEST/results.json` with `xgboost_cfg3_weighted`. `BEST/results.json` now holds
  `xgboost_balanced_p050` again. Note that both of these old scripts ranked models on **test** macro-F1, and the
  validation score of `xgboost_balanced_p050` was not recorded. So the 34-class single-model "best" is a test-set
  ranking. The 34-class headline result (the v3 final ensemble) was selected on validation.
- **8-class `xgboost_balanced_deep` config.** The script (`src/training/phase2_extra.py`) and the saved config
  both use depth 14, learning rate 0.05 and sqrt (power 0.5) class weights. Decision D19 wrongly said depth 16 /
  0.03, which is the config of `xgboost_deep`. D19 in `reports/DECISIONS.md` has been corrected.
- The v4 split sizes in an earlier version of `reports/REPORT.md` were wrong (they were read from `.npy` file sizes,
  which include a header). They were corrected to the values in `results/v4_46feat/summary.json`.
- Numbers are formatted by one script (`paper/tools/build_facts.py`): metrics to 4 decimals, standard deviations to
  6 decimals, percentages to 2 decimals. The IDPS action and 8-class accuracies are shown to 5 decimals, as in
  the IDPS summary.
