"""Builds v3/results/REPORT.md, SUMMARY.txt and 2 figures from saved JSONs/probs. Re-runnable; missing items marked PENDING."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import *
import glob, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
FIG = V3 / "figures"; FIG.mkdir(exist_ok=True)
J = lambda p: json.load(open(p)) if Path(p).exists() else None
rel = lambda p: str(Path(p)).replace("/dev/shm/saksham/ids/", "ids/")
T = {"2_class": (0.9968, 0.9653), "8_class": (0.9944, 0.7193), "34_class": (0.9916, 0.7140)}
L = []; w = L.append
def f(x, n=4): return "—" if x is None else f"{x:.{n}f}"
def ms(v, n=4): return f"{np.mean(v):.{n}f} ± {np.std(v):.{n}f} (n={len(v)})"

w("# CICIoT2023 IDS — v3 report (new methods on top of the previous run)\n")
w(f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Split: prior run's shuffled per-row 75/5/20 (seed 42), val 2,339,084 rows, test 9,350,212 rows. "
  "All selection (thresholds, calibration, ensemble choice, early stopping) uses **validation only**; val-fitted methods were scored by 2-fold CV on val for selection. "
  "Test numbers are reported for every candidate, but the pick is by val. Targets: Neto et al. 2023 (RF).\n")
w("All paths are relative to `/dev/shm/saksham/` (`~/saksham/ids` → `ids/`).\n")

# ---------- headline ----------
w("## 1. Headline\n")
w("| Task | Prior best single model (test; best on val) | v3 best (test, selected by val) | Target | Status |\n|---|---|---|---|---|")
rows = {}
for task, prior, pm in [("2_class", "LightGBM cfg2", (0.993754, 0.932755)), ("8_class", "XGB balanced_deep", (0.841456, 0.730818)),
                        ("34_class", "XGB balanced_p050", (0.794564, 0.642876))]:
    best = None
    for name in ["final_ensemble", "phase_a"]:
        r = J(RES / task / name / "results.json")
        if r: best = (name, r["selected_by_val"], r["selected_test"]); break
    ta, tf = T[task]
    if best:
        acc, mf = best[2]["accuracy"], best[2]["macro_f1"]; src = f"{best[0]}/{best[1]}"
    else:
        acc, mf = pm; src = "unchanged (prior model)" if task == "8_class" else "PENDING"
    met = (mf >= tf) if task != "2_class" else (acc >= ta)
    rows[task] = (acc, mf, met)
    w(f"| {task} | {prior}: acc {pm[0]:.4f}, mF1 {pm[1]:.4f} | {src}: acc {acc:.4f}, mF1 {mf:.4f} | acc {ta}, mF1 {tf} | **{'MET' if met else 'NOT MET'}** |")
w("\n2-class status is judged on accuracy (+0.30 pts needed); 8/34-class on macro-F1. 8-class was already MET and not re-tuned.\n")

# ---------- Phase A ----------
w("## 2. Phase A — free wins on existing models (CPU)\n")
for task in ["2_class", "34_class"]:
    for name, title in [("phase_a", "best single model only + A3 over prior models"), ("final_ensemble", "all models incl. v3 CatBoost / hierarchical / focal")]:
        p = RES / task / name / "results.json"; r = J(p)
        w(f"**{task} — {title}** (`{rel(p)}`)\n")
        if not r: w("PENDING (still running when this report was generated)\n"); continue
        w(f"Models: {', '.join(r['models'])}\n")
        w("| Candidate | val mF1 (selection) | test acc | test mF1 |\n|---|---|---|---|")
        for c, m in r["candidates"].items():
            star = " ←selected" if c == r["selected_by_val"] else ""
            w(f"| {c}{star} | {f(m['val_mF1_selection'])} | {f(m['accuracy'])} | {f(m['macro_f1'])} |")
        w("")
w("A1 = per-class probability multipliers (0.3–3.0, step 0.1, coordinate ascent on val mF1). A2 = per-class isotonic / Platt calibration "
  "(one-vs-rest, equivalent to CalibratedClassifierCV on frozen probs). A3 = soft vote, val-mF1-weighted vote, multinomial-LogReg stacking on log-probs "
  "(final_ensemble: stacker fit on ≤20k rows/class with prior correction, for speed).\n")

# ---------- CatBoost ----------
w("## 3. Phase B1 — CatBoost GPU\n")
w("1M rows/class cap; probabilities prior-corrected back to the full-train distribution; early stopping on val macro-F1 "
  "(val reweighted to the training prior). Two bugs found and fixed mid-run (logloss early-stop fooled by the cap's prior shift; double prior correction "
  "with `Balanced`); runs below are post-fix unless noted.\n")
w("| Run | best iter | val acc | val mF1 | test acc | test mF1 | train s |\n|---|---|---|---|---|---|---|")
for p in sorted(glob.glob(str(RES / "*/catboost/*/results.json"))):
    r = J(p); fixed = "early_stop_metric" in r["config"]
    note = "" if fixed else " (pre-fix, logloss stop)"
    w(f"| {Path(p).parent.name}{note} | {r['best_iteration']} | {f(r['val']['accuracy'])} | {f(r['val']['macro_f1'])} | {f(r['test']['accuracy'])} | {f(r['test']['macro_f1'])} | {r['train_time_s']} |")
for task in ["2_class", "34_class"]:
    v = [J(p)["test"]["macro_f1"] for p in glob.glob(str(RES / task / "catboost" / f"cb_{task}_c1_none_s*" / "results.json")) if "_b" not in Path(p).parent.name.split("_s")[-1]]
    a = [J(p)["test"]["accuracy"] for p in glob.glob(str(RES / task / "catboost" / f"cb_{task}_c1_none_s*" / "results.json")) if "_b" not in Path(p).parent.name.split("_s")[-1]]
    if v: w(f"\n{task} c1_none over seeds: test mF1 {ms(v)}, test acc {ms(a)}")
w("\nFinding: CatBoost plateaus far below XGBoost/LightGBM on 34-class (≈0.54 vs 0.64 mF1); 1024 borders gives no gain (`cb_34_class_c2_none_s42_b1024`), so binning is not the cause. Not closing either gap.\n")

# ---------- focal ----------
w("## 4. Phase B2 — focal reweighting (34-class)\n")
p = RES / "34_class/focal/xgb_34_class_focal_s42/results.json"; r = J(p)
if r:
    w(f"`{rel(p)}` — w = sqrt-balanced class weight × (1−p_correct)^2 (floor 1e-3), focal renormalised within class; p_correct from balanced_p050 on train; "
      f"XGB depth 12, lr 0.05, max_bin 128, 1M/class cap, {r['best_iteration']} rounds, {r['train_time_s']:.0f}s, VRAM {r['vram_used_MiB_gpu0_after_fit']} MiB.\n")
    w(f"val mF1 {f(r['val']['macro_f1'])} / acc {f(r['val']['accuracy'])}; **test mF1 {f(r['test']['macro_f1'])} / acc {f(r['test']['accuracy'])}** — much worse than baseline 0.6429. "
      "The 'hard' rows are mostly near-identical DoS/DDoS SYN-flood twins, so focal weighting amplifies label ambiguity. Single seed only (≈50 min/run; clearly negative).\n")

# ---------- hierarchical ----------
w("## 5. Novelty C1 — hierarchical 8→34 classification\n")
H = {s: J(V3 / "novelty" / f"hierarchical_s{s}.json") for s in [42, 1, 2]}; H = {s: h for s, h in H.items() if h}
if H:
    h0 = H.get(42) or next(iter(H.values()))
    w("Stage 1 = prior 8-class XGB cfg3_weighted (cached probs). Stage 2 = XGB expert per multi-subtype family (depth 8, 300 rounds, max_bin 128, ≤300k rows/subtype). "
      "Soft: P(sub)=P(family)·P(sub|family); hard: route to argmax family.\n")
    w("| Expert | subtypes | train rows | within-family val acc (true family) |\n|---|---|---|---|")
    for fam, e in h0["experts"].items(): w(f"| {fam} | {len(e['subtypes'])} | {e['train_rows']:,} | {e['expert_val_acc_on_true_family']:.4f} |")
    for k_ in ["hard_test", "soft_test", "hard_val"]:
        w(f"\n{k_}: mF1 {ms([h[k_]['macro_f1'] for h in H.values()])}, acc {ms([h[k_]['accuracy'] for h in H.values()])}")
    w(f"\nFlat baseline (balanced_p050) test mF1 0.6429, acc 0.7946. Files: `ids/v3/novelty/hierarchical_s{{42,1,2}}.json`.\n")
    w("Finding: within-family experts are near-perfect for DoS (0.999) and Mirai (0.999) but the hierarchy is ≈0.007 mF1 *below* flat: stage-1 DoS↔DDoS routing errors "
      "propagate and cannot be undone by stage 2. Web_Based (0.51) and Recon (0.73) experts are data-starved. Negative-but-informative result.\n")
    # per-class figure flat vs hier
    yt = np.load(PROBS / "34_class/y_test.npy").astype(np.int64); nm = names("34_class")
    ff = f1_score(yt, np.load(PROBS / "34_class/xgb_34_class_balanced_p050_test.npy").argmax(1), average=None, labels=range(34))
    hf = [h0["hard_test"]["per_class_f1"][n] for n in nm]
    fig, ax = plt.subplots(figsize=(13, 4.5)); x = np.arange(34)
    ax.bar(x - 0.2, ff, 0.4, label="flat XGB balanced_p050", color="#4a7bb7"); ax.bar(x + 0.2, hf, 0.4, label="hierarchical (hard, s42)", color="#e08a3c")
    ax.set_xticks(x); ax.set_xticklabels(nm, rotation=90, fontsize=7); ax.set_ylabel("test F1"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    ax.set_title("34-class per-class test F1: flat vs hierarchical"); fig.tight_layout(); fig.savefig(FIG / "34class_flat_vs_hier_f1.png", dpi=130); plt.close(fig)
    w("Figure: `ids/v3/figures/34class_flat_vs_hier_f1.png`\n")

# ---------- DoS/DDoS ----------
w("## 6. Novelty C3 — DoS vs DDoS\n")
d = J(V3 / "novelty/dos_ddos.json")
if d:
    c = d["test_8class_dos_ddos_cells"]; b = d["binary_dos_vs_ddos_prior_run"]
    w(f"`ids/v3/novelty/dos_ddos.json`. 8-class test confusion: DDoS→DoS {c['DDoS->DoS']:,}, DoS→DDoS {c['DoS->DDoS']:,} — "
      f"**{100*c['share_of_all_8class_errors']:.1f}% of all 8-class errors**.\n")
    w(f"Merging DoS+DDoS into 'Flood' (7 classes, same predictions): test acc {d['test_8class']['accuracy']:.4f} → **{d['test_7class_flood']['accuracy']:.4f}**, "
      f"mF1 {d['test_8class']['macro_f1']:.4f} → **{d['test_7class_flood']['macro_f1']:.4f}**.\n")
    w(f"Dedicated binary DoS-vs-DDoS classifier (prior run N3, `ids/novelty/N3_dos_vs_ddos/results.json`): AUC {b['paper_protocol']['auc']:.3f}, acc {b['paper_protocol']['accuracy']:.3f}; "
      f"top features: {', '.join(list(b['paper_protocol']['top_features'])[:10])}.\n")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(["8-class", "7-class (Flood)"], [d["test_8class"]["accuracy"], d["test_7class_flood"]["accuracy"]], color=["#4a7bb7", "#3c9e6e"], width=0.5, label="accuracy")
    ax.plot(["8-class", "7-class (Flood)"], [d["test_8class"]["macro_f1"], d["test_7class_flood"]["macro_f1"]], "ko-", label="macro-F1")
    ax.set_ylim(0.6, 1.0); ax.legend(fontsize=8); ax.set_title("Merging DoS+DDoS (test, same model)"); fig.tight_layout()
    fig.savefig(FIG / "dos_ddos_merge.png", dpi=130); plt.close(fig); w("Figure: `ids/v3/figures/dos_ddos_merge.png`\n")

# ---------- C2/C4 ----------
w("## 7. Novelty C2 / C4 — reused from prior run (not re-run)\n")
pl = J(V3 / "novelty/protocol_ladder.json"); wa = J(V3 / "novelty/window_artifact.json")
if pl:
    rr = pl["results"]; keys = list(rr)
    w("Protocol ladder, 8-class (`ids/novelty/N2_protocol_ladder/results.json`): " + "; ".join(
        f"{k}: mF1 {rr[k]['macro_f1']:.4f}" for k in keys if isinstance(rr[k], dict) and "macro_f1" in rr[k]) +
      ". File-grouped split drops mF1 by ≈0.285 → shuffled-split inflation from within-file twins (59% of test rows have an exact train twin, N1).\n")
if wa:
    r = wa["results"]; w(f"Window-size artifact (`ids/novelty/N5_window_size/results.json`): features identify the 10- vs 100-packet window group with acc {r['window_classifier_accuracy']:.4f}; "
                         f"top features {', '.join(r['top_window_features'][:5])}.\n")

# ---------- conclusions ----------
w("## 8. Conclusions\n")
w("- **2-class: NOT MET.** Every post-hoc method moves accuracy by <0.0002; the 0.9968 target sits above the ≈0.994 ceiling seen across 20+ models. "
  "Prior-run evidence (N1 twins, D15/D16) attributes the paper's number to evaluation protocol, not model quality.")
w("- **34-class: NOT MET.** Best gain is threshold tuning (≈+0.002 mF1). CatBoost (≈0.54), focal (0.515) and hierarchical (≈0.636) do not beat flat XGBoost (0.643).")
w("- **Main novelty:** the residual error is structural — 95% of 8-class errors are DoS↔DDoS, and merging them gives 99.2% accuracy. The hierarchy confirms subtype "
  "separation inside a family is easy; family-level DoS/DDoS separation is the bottleneck.")
w("- **Honesty notes:** test never used for selection; 3 seeds for hierarchical and CatBoost c1_none; focal and ensembles single-seed (deterministic given inputs / too slow and clearly negative). "
  "Phase A 2-class was run earlier in this session by a previous v3 script (`ids/v3/results/2_class/{threshold_tuned,calibrated,ensemble}`); numbers agree with `phase_a`.")
w("\n## 9. Files\n")
w("- Results JSON: `ids/v3/results/<task>/{phase_a,final_ensemble,catboost/<run>,focal/<run>}/results.json`\n- Novelty: `ids/v3/novelty/{hierarchical_s*,dos_ddos,protocol_ladder,window_artifact}.json`\n"
  "- Cached probabilities: `ids/v3/probs/<task>/*.npy`\n- New models: `ids_cache/v3_models/`\n- Scripts: `ids/v3/src/`, logs: `ids/v3/logs/`\n- Figures: `ids/v3/figures/`")
open(RES / "REPORT.md", "w").write("\n".join(L) + "\n")

S = ["CICIoT2023 IDS v3 — SUMMARY (details: ids/v3/results/REPORT.md)", ""]
for task in ["2_class", "8_class", "34_class"]:
    a, m_, met = rows[task]; S.append(f"{task:9s} acc={a:.4f} mF1={m_:.4f}  target acc {T[task][0]} / mF1 {T[task][1]}  -> {'MET' if met else 'NOT MET'}")
S += ["", "New in v3:", "- A1/A2/A3 post-hoc tuning: <=+0.002 mF1 (34c), <0.0002 acc (2c)", "- CatBoost GPU (fixed prior-shift early-stop bug): 34c mF1 ~0.54, 2c acc ~0.9927",
      "- Focal reweighting 34c: mF1 0.515 (worse)", "- Hierarchical 8->34: mF1 ~0.636 vs flat 0.643 (DoS/DDoS routing is the bottleneck)",
      "- DoS<->DDoS = 95% of 8-class errors; merged 'Flood' 7-class acc 0.992", "", "Test never used for selection. Every number cites a JSON in REPORT.md."]
open(RES / "SUMMARY.txt", "w").write("\n".join(S) + "\n")
print("wrote", RES / "REPORT.md", len(L), "lines")
