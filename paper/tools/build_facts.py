"""Collect every number used in paper/ and report/ from the repo's result JSON files.

Writes:
  paper/data/facts.json        flat key -> formatted value (+ raw value and source JSON path)
  paper/data/missing.json      numbers we looked for and did not find
  paper/tables/*.tex           LaTeX tables (IEEEtran)
  report/tables/*.md           the same tables in Markdown
  paper/macros.tex             \\newcommand per fact, so prose in main.tex never retypes a number

Formatting rule (applied everywhere, never by hand): metrics in [0,1] to 4 decimals, standard deviations
to 6 decimals, percentages to 2 decimals, seconds as integers, counts with thousands separators.
"""
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
R = REPO / "results"
FACTS, MISSING = {}, []
PAPER = {2: (0.99680798, 0.965279544), 8: (0.994368173, 0.71928904), 34: (0.99164365, 0.714021981)}
PAPER_SRC = "Neto et al. 2023, Table 6 (Random Forest; F1 averaging not stated in the paper)"
M = "[MISSING]"


def load(p):
    return json.loads((R / p).read_text())


def f4(x):
    return M if x is None else f"{x:.4f}"


def pct(x):
    return M if x is None else f"{x:.2f}\\%"


def fact(key, raw, fmt, src):
    FACTS[key] = {"value": fmt(raw) if raw is not None else M, "raw": raw, "source": src}
    if raw is None:
        MISSING.append({"key": key, "source": src})
    return FACTS[key]["value"]


def num(x):
    return f"{int(x):,}"


# ---------------------------------------------------------------- dataset facts
dm = json.loads((REPO / "src/preprocessing/data_meta.json").read_text())
fact("fullRows", dm["total_rows"], num, "src/preprocessing/data_meta.json")
fact("fullFeat", dm["n_features"], str, "src/preprocessing/data_meta.json")
fact("fullBenign", dm["class8_counts"]["Benign"], num, "src/preprocessing/data_meta.json")
fact("fullAttack", dm["total_rows"] - dm["class8_counts"]["Benign"], num, "src/preprocessing/data_meta.json (total - Benign)")
v4cc = load("v4_46feat/class_counts.json")
v4s = load("v4_46feat/summary.json")
fact("vfRows", v4cc["n_rows"], num, "results/v4_46feat/class_counts.json")
for k in ("n_train", "n_val", "n_test"):
    fact("vf" + k.split("_")[1].capitalize(), v4s["split"][k], num, "results/v4_46feat/summary.json")
fact("vfBenign", v4cc["class2"]["Benign"], num, "results/v4_46feat/class_counts.json")
fact("vfAttack", v4cc["class2"]["Attack"], num, "results/v4_46feat/class_counts.json")
lg2 = load("2_class/lightgbm_cfg2/results.json")
nval = int(np.sum(lg2["val_metrics"]["confusion_matrix"]))
ntest = int(np.sum(lg2["test_metrics"]["confusion_matrix"]))
fact("vtVal", nval, num, "results/2_class/lightgbm_cfg2/results.json (sum of val confusion matrix)")
fact("vtTest", ntest, num, "results/2_class/lightgbm_cfg2/results.json (sum of test confusion matrix)")
fact("vtTrain", dm["total_rows"] - nval - ntest, num, "data_meta total - val - test")

# ---------------------------------------------------------------- every trained model
V1 = {"xgboost_cfg1", "xgboost_cfg2", "xgboost_cfg3_weighted", "lightgbm_cfg1", "lightgbm_cfg2",
      "lightgbm_cfg3_weighted", "catboost_balanced", "mlp_v1"}


def cfg_str(c):
    if not c:
        return "-"
    keys = [("n_estimators", "trees"), ("max_depth", "depth"), ("max_leaves", "leaves"), ("num_leaves", "leaves"),
            ("depth", "depth"), ("learning_rate", "lr"), ("layers", "layers"), ("iterations", "iters"),
            ("gamma", "focal-gamma"), ("weights", "w"), ("class_weight", "w"), ("auto_class_weights", "w"),
            ("weight_power", "w-power")]
    out = []
    for k, lab in keys:
        if k in c and c[k] not in (None, "", "0", 0) or (k == "max_depth" and c.get(k) in (0, "0") and "max_leaves" in c):
            v = c[k]
            if k == "max_depth" and v in (0, "0"):
                continue
            if k == "class_weight" and "sqrt" in str(v):
                v = "sqrt-bal"
            out.append(f"{lab}={v}" if not isinstance(v, list) else f"{lab}={'-'.join(map(str, v))}")
    return ", ".join(out) or "-"


rows = []  # dict(version, dataset, task, model, config, val_acc, val_mf1, test_acc, test_mf1, time_s, src)


def add(version, data, task, model, cfg, val, test, t, src, note=""):
    rows.append(dict(version=version, data=data, task=task, model=model, config=cfg,
                     val_acc=None if val is None else val.get("accuracy"),
                     val_mf1=None if val is None else val.get("macro_f1"),
                     test_acc=test.get("accuracy"), test_mf1=test.get("macro_f1"), time_s=t, src=src, note=note))


for task in (2, 8, 34):
    for d in sorted((R / f"{task}_class").iterdir()):
        p = d / "results.json"
        if d.name in ("BEST", "v3", "threshold_optimized", "ensemble_soft_vote", "accuracy_push") or not p.exists():
            continue
        j = json.loads(p.read_text())
        ver = "v1" if d.name in V1 else "v2"
        note = "val merged into train" if d.name == "xgboost_trainval" else ""
        val = j.get("val_metrics")
        cs = cfg_str(j.get("config"))
        c = j.get("config") or {}
        if any(k in c for k in ("auto_class_weights", "class_weight", "weight_power")):
            pass  # weighting already shown from the config
        elif "p025" in d.name:
            cs += ", class weights power 0.25"
        elif "weighted" in d.name or "balanced" in d.name:
            cs += ", sqrt class weights"
        add(ver, "full", task, d.name, cs, val, j["test_metrics"],
            j.get("training_time_seconds"), str(p.relative_to(REPO)), note)
for j in load("2_class/accuracy_push/results.json"):
    add("v2", "full", 2, "accpush_" + j["method"], cfg_str(j.get("config")), None, j["test_metrics"],
        j.get("training_time_seconds"), "results/2_class/accuracy_push/results.json", "test-only push run")
for task in (2, 34):
    for p in sorted((R / f"{task}_class/v3/catboost").glob("*/results.json")):
        j = json.loads(p.read_text())
        add("v3", "full", task, "catboost_" + p.parent.name.replace(f"cb_{task}_class_", ""), cfg_str(j["config"]),
            j["val"], j["test"], j.get("train_time_s"), str(p.relative_to(REPO)))
j = load("34_class/v3/focal/xgb_34_class_focal_s42/results.json")
add("v3", "full", 34, "xgboost_focal_s42", cfg_str(j["config"]), j["val"], j["test"], j.get("train_time_s"),
    "results/34_class/v3/focal/xgb_34_class_focal_s42/results.json")
for s in (1, 2, 42):
    j = load(f"novelty/v3/hierarchical_s{s}.json")
    t = sum(e["train_s"] for e in j["experts"].values())
    add("v3", "full", 34, f"hierarchical_8to34_hard_s{s}", "stage1 xgb cfg3_weighted + 6 family experts",
        j["hard_val"], j["hard_test"], t, f"results/novelty/v3/hierarchical_s{s}.json", "time = experts only")
for p in sorted((R / "v4_46feat").glob("*_*c_*f.json")):
    j = json.loads(p.read_text())
    fs = {"38": "38f", "46": "46f", "38n": "38f-noIAT", "46n": "46f-noIAT"}[str(j["fset"])]
    add("v4", "kaggle", j["task"], f"{j['model']} [{fs}]", cfg_str(j.get("config")), j["val_metrics"],
        j["test_metrics"], j.get("train_seconds"), str(p.relative_to(REPO)))

for r in rows:
    if r["val_acc"] is None and not r["note"]:
        r["note"] = "val not recorded"
        MISSING.append({"key": f"val metrics {r['model']} ({r['task']}-class)", "source": r["src"]})
    if r["time_s"] is None:
        MISSING.append({"key": f"training time {r['model']} ({r['task']}-class)", "source": r["src"]})

# ---------------------------------------------------------------- post-hoc (v2/v3) table
posthoc = []
j = load("2_class/threshold_optimized/results.json")
posthoc.append(("v2", 2, "LightGBM cfg2 + threshold 0.49", None, j["test_metrics_optimized"], "results/2_class/threshold_optimized/results.json"))
j = load("2_class/ensemble_soft_vote/results.json")
posthoc.append(("v2", 2, "Soft-vote ensemble (3 models)", None, j["test_metrics"], "results/2_class/ensemble_soft_vote/results.json"))
for task in (2, 34):
    j = load(f"{task}_class/v3/final_ensemble/results.json")
    for name, c in j["candidates"].items():
        sel = " (selected by val)" if name == j["selected_by_val"] else ""
        posthoc.append(("v3", task, f"{name}{sel}", c.get("val_mF1_selection"), c,
                        f"results/{task}_class/v3/final_ensemble/results.json"))

# ---------------------------------------------------------------- headline facts
ms = {t: load(f"{t}_class/BEST/multi_seed_results.json") for t in (2, 8, 34)}
fe2, fe34 = load("2_class/v3/final_ensemble/results.json"), load("34_class/v3/final_ensemble/results.json")
x8 = load("8_class/xgboost_cfg3_weighted/results.json")
full_best = {2: ("Stacked ensemble + threshold (v3)", fe2["selected_test"], "results/2_class/v3/final_ensemble/results.json"),
             8: ("XGBoost cfg3_weighted (v1)", x8["test_metrics"], "results/8_class/xgboost_cfg3_weighted/results.json"),
             34: ("XGBoost balanced_p050 + per-class thresholds (v3)", fe34["selected_test"], "results/34_class/v3/final_ensemble/results.json")}
for t, (n, m, s) in full_best.items():
    fact(f"fullBestAcc{t}", m["accuracy"], f4, s)
    fact(f"fullBestMf{t}", m["macro_f1"], f4, s)
for t in (2, 8, 34):
    fact(f"paperAcc{t}", PAPER[t][0], f4, PAPER_SRC)
    fact(f"paperMf{t}", PAPER[t][1], f4, PAPER_SRC)
    fact(f"msAcc{t}", ms[t]["mean_accuracy"], f4, f"results/{t}_class/BEST/multi_seed_results.json")
    fact(f"msAccSd{t}", ms[t]["std_accuracy"], lambda x: f"{x:.6f}", f"results/{t}_class/BEST/multi_seed_results.json")
    fact(f"msMf{t}", ms[t]["mean_macro_f1"], f4, f"results/{t}_class/BEST/multi_seed_results.json")
    fact(f"msMfSd{t}", ms[t]["std_macro_f1"], lambda x: f"{x:.6f}", f"results/{t}_class/BEST/multi_seed_results.json")
for key, sel in v4s["selected"].items():
    t, fs = key.split("_")
    tag = {"38f": "A", "46f": "B", "38nf": "C", "46nf": "D"}[fs]
    n = t.replace("c", "")
    fact(f"vf{tag}Acc{n}", sel["test"]["accuracy"], f4, "results/v4_46feat/summary.json")
    fact(f"vf{tag}Mf{n}", sel["test"]["macro_f1"], f4, "results/v4_46feat/summary.json")

# ---------------------------------------------------------------- novelty facts
n1 = load("novelty/N1_twin_analysis/results.json")
fact("nOneTwin", n1["overall_twin_pct"], lambda x: f"{x:.2f}\\%", "results/novelty/N1_twin_analysis/results.json")
fact("nOneDDoS", n1["per_class_twin_pct"]["DDoS"], lambda x: f"{x:.2f}\\%", "results/novelty/N1_twin_analysis/results.json")
fact("nOneBenign", n1["per_class_twin_pct"]["Benign"], lambda x: f"{x:.2f}\\%", "results/novelty/N1_twin_analysis/results.json")
fact("nOneInfl", n1["twin_inflation_macro_f1"], f4, "results/novelty/N1_twin_analysis/results.json")
fact("nOneTwinMf", n1["twin_subset_metrics"]["macro_f1"], f4, "results/novelty/N1_twin_analysis/results.json")
fact("nOneNonTwinMf", n1["nontwin_subset_metrics"]["macro_f1"], f4, "results/novelty/N1_twin_analysis/results.json")
fact("vfTwin", v4s["twins"]["46f"]["test_twin_pct"], lambda x: f"{x:.2f}\\%", "results/v4_46feat/summary.json")
fact("vfDup", v4s["twins"]["46f"]["dup_rows_in_full_data_pct"], lambda x: f"{x:.2f}\\%", "results/v4_46feat/summary.json")
n2 = load("novelty/N2_protocol_ladder/results.json")
for k, tag in [("protocol_a_paper", "A"), ("protocol_b_dedup", "B"), ("protocol_c_file_grouped", "C"),
               ("protocol_d_file_grouped_twin_free", "D")]:
    fact(f"nTwo{tag}Mf", n2[k]["macro_f1"], f4, "results/novelty/N2_protocol_ladder/results.json")
    fact(f"nTwo{tag}Acc", n2[k]["accuracy"], f4, "results/novelty/N2_protocol_ladder/results.json")
fact("nTwoDrop", n2["protocol_a_paper"]["macro_f1"] - n2["protocol_d_file_grouped_twin_free"]["macro_f1"], f4,
     "results/novelty/N2_protocol_ladder/results.json (a - d)")
fact("nTwoDedupPct", n2["dedup_removed_pct"], lambda x: f"{x:.2f}\\%", "results/novelty/N2_protocol_ladder/results.json")
dd = load("novelty/v3/dos_ddos.json")
fact("nThreeShare", dd["test_8class_dos_ddos_cells"]["share_of_all_8class_errors"] * 100, lambda x: f"{x:.2f}\\%", "results/novelty/v3/dos_ddos.json")
fact("nThreeFloodAcc", dd["test_7class_flood"]["accuracy"], f4, "results/novelty/v3/dos_ddos.json")
fact("nThreeFloodMf", dd["test_7class_flood"]["macro_f1"], f4, "results/novelty/v3/dos_ddos.json")
fact("nThreeEightAcc", dd["test_8class"]["accuracy"], f4, "results/novelty/v3/dos_ddos.json")
fact("nThreeEightMf", dd["test_8class"]["macro_f1"], f4, "results/novelty/v3/dos_ddos.json")
fact("nThreeBinPaper", dd["binary_dos_vs_ddos_prior_run"]["paper_protocol"]["accuracy"], f4, "results/novelty/v3/dos_ddos.json")
fact("nThreeBinGroup", dd["binary_dos_vs_ddos_prior_run"]["file_grouped"]["accuracy"], f4, "results/novelty/v3/dos_ddos.json")
vdd = v4s["dos_ddos"]
fact("vfDdWith", vdd["xgb_best_8c_46f"]["dos_ddos_total"], num, "results/v4_46feat/summary.json")
fact("vfDdWithShare", vdd["xgb_best_8c_46f"]["share_of_errors"] * 100, lambda x: f"{x:.2f}\\%", "results/v4_46feat/summary.json")
fact("vfDdNo", vdd["xgb_best_8c_46nf"]["dos_ddos_total"], num, "results/v4_46feat/summary.json")
fact("vfDdNoShare", vdd["xgb_best_8c_46nf"]["share_of_errors"] * 100, lambda x: f"{x:.2f}\\%", "results/v4_46feat/summary.json")
fact("vfDdNoA", vdd["xgb_best_8c_38nf"]["dos_ddos_total"], num, "results/v4_46feat/summary.json")
ip = load("v4_46feat/iat_probe.json")
iat = next(r for r in ip["single_feature_probe"] if r["feature"] == "IAT")
second = [r for r in ip["single_feature_probe"] if r["feature"] != "IAT"][0]
fact("iatAcc", iat["val_accuracy"], f4, "results/v4_46feat/iat_probe.json")
fact("iatMf", iat["val_macro_f1"], f4, "results/v4_46feat/iat_probe.json")
fact("iatMaj", ip["majority_class_val_accuracy"], f4, "results/v4_46feat/iat_probe.json")
fact("iatSecondName", second["feature"], str, "results/v4_46feat/iat_probe.json")
fact("iatSecondAcc", second["val_accuracy"], f4, "results/v4_46feat/iat_probe.json")
fact("iatSecondMf", second["val_macro_f1"], f4, "results/v4_46feat/iat_probe.json")
fact("iatImpEight", v4s["importance"]["8c_46f"]["IAT_share"] * 100, lambda x: f"{x:.2f}\\%", "results/v4_46feat/summary.json")
n4 = load("novelty/N4_fingerprint_ablation/results.json")
fact("nFourDrop", n4["without_ttl_iat"]["accuracy_drop"], lambda x: f"{x:.6f}", "results/novelty/N4_fingerprint_ablation/results.json")
fact("nFourBase", n4["baseline"]["accuracy"], f4, "results/novelty/N4_fingerprint_ablation/results.json")
n5 = load("novelty/N5_window_size/results.json")
fact("nFiveAcc", n5["window_classifier_accuracy"] * 100, lambda x: f"{x:.4f}\\%", "results/novelty/N5_window_size/results.json")
fact("nFiveMfChange", n5["macro_f1_change"], f4, "results/novelty/N5_window_size/results.json")
hier = [load(f"novelty/v3/hierarchical_s{s}.json")["hard_test"]["macro_f1"] for s in (1, 2, 42)]
fact("hierMf", float(np.mean(hier)), f4, "results/novelty/v3/hierarchical_s{1,2,42}.json (mean of hard_test)")
fact("hierMfSd", float(np.std(hier)), lambda x: f"{x:.6f}", "results/novelty/v3/hierarchical_s{1,2,42}.json (population std)")
flat34 = load("34_class/xgboost_balanced_p050/results.json")["test_metrics"]["macro_f1"]
fact("flatMfThirtyFour", flat34, f4, "results/34_class/xgboost_balanced_p050/results.json")
cb34 = [r for r in rows if r["task"] == 34 and r["model"].startswith("catboost_") and r["version"] == "v3"]
best_cb = max(cb34, key=lambda r: r["val_mf1"])
fact("cbBestMf", best_cb["test_mf1"], f4, best_cb["src"])
fact("cbBestName", best_cb["model"].replace("_", "\\_"), str, best_cb["src"])
foc = load("34_class/v3/focal/xgb_34_class_focal_s42/results.json")["test"]["macro_f1"]
fact("focalMf", foc, f4, "results/34_class/v3/focal/xgb_34_class_focal_s42/results.json")
n6 = load("novelty/N6_class_weight_sweep/results.json")
bestp = max(n6["sweep"], key=lambda s: s["macro_f1"])
fact("nSixBestP", bestp["power"], lambda x: f"{x}", "results/novelty/N6_class_weight_sweep/results.json")
fact("nSixBestMf", bestp["macro_f1"], f4, "results/novelty/N6_class_weight_sweep/results.json")

# ---------------------------------------------------------------- IDPS facts
ids = load("prevention/summary.json")
al = ids["action_level"]
src = "results/prevention/summary.json"
fact("idsRows", ids["stream"]["rows"], num, src)
fact("idsActAcc", al["accuracy"], lambda x: f"{x:.5f}", src)
fact("idsEightAcc", ids["plain_8class"]["accuracy"], lambda x: f"{x:.5f}", src)
fact("idsEightMf", ids["plain_8class"]["macro_f1"], f4, src)
fact("idsFalseBlock", al["false_block_rate"] * 100, lambda x: f"{x:.2f}\\%", src)
fact("idsFalseBlockN", al["false_block_count"], num, src)
fact("idsMiss", al["miss_rate"] * 100, lambda x: f"{x:.2f}\\%", src)
fact("idsMissN", al["miss_count"], num, src)
fact("idsErr", ids["plain_8class"]["errors"], num, src)
fact("idsAbsorbed", al["class_errors_with_correct_action"], num, src)
fact("idsAbsorbedDd", al["of_which_dos_ddos_confusions"], num, src)
fact("idsLatSingle", ids["latency_ms_per_row"]["single_row"]["avg"], lambda x: f"{x:.2f}", src)
fact("idsLatSingleP", ids["latency_ms_per_row"]["single_row"]["p99"], lambda x: f"{x:.2f}", src)
fact("idsLatBatch", ids["latency_ms_per_row"]["batch_1000"]["avg"], lambda x: f"{x:.4f}", src)
fact("idsLatBatchP", ids["latency_ms_per_row"]["batch_1000"]["p99"], lambda x: f"{x:.4f}", src)

fact("nModels", len(rows), str, "paper/data/all_models.json (rows collected above)")
fact("nPosthoc", len(posthoc), str, "post-hoc rows collected above")
fact("nModelsVfour", sum(r["version"] == "v4" for r in rows), str, "results/v4_46feat/*_*c_*f.json")
for lab, key in [("DoS-TCP_Flood", "iatMedDoSTCP"), ("DDoS-ICMP_Flood", "iatMedDDoSICMP"), ("BenignTraffic", "iatMedBenign")]:
    fact(key, ip["iat_per_class_quantiles_raw"][lab]["p50"], lambda x: f"{x:.4g}", "results/v4_46feat/iat_probe.json")

# ---------------------------------------------------------------- table writers
TEX, MD = REPO / "paper/tables", REPO / "report/tables"
TEX.mkdir(parents=True, exist_ok=True)
MD.mkdir(parents=True, exist_ok=True)
(REPO / "paper/data").mkdir(parents=True, exist_ok=True)


def esc(s):
    parts = str(s).split("$")
    if len(parts) > 1:  # keep $...$ math segments untouched
        return "$".join(p if i % 2 else _esc(p) for i, p in enumerate(parts))
    return _esc(str(s))


def _esc(s):
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                 ("→", "$\\rightarrow$"), ("↔", "$\\leftrightarrow$"), ("±", "$\\pm$"), ("Δ", "$\\Delta$")]:
        s = s.replace(a, b)
    return s.replace("\\textbackslash{}%", "\\%")


def tomd(c):
    c = str(c)
    c = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", c)
    for a, b in [("\\%", "%"), ("$\\pm$", "±"), ("$\\rightarrow$", "→"), ("\\_", "_")]:
        c = c.replace(a, b)
    return c


def write_table(name, header, body, caption, label, colspec, wide=True, size="\\scriptsize", longtable=False):
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(tomd(c) for c in r) + " |" for r in body]
    header = [tomd(h) for h in header]
    md[0] = "| " + " | ".join(header) + " |"
    (MD / f"{name}.md").write_text(f"**{caption}**\n\n" + "\n".join(md) + "\n")
    texrows = [" & ".join(c if str(c).startswith("$") or "\\%" in str(c) or "\\textbf" in str(c) else esc(c)
                          for c in r) + " \\\\" for r in body]
    head = " & ".join(f"\\textbf{{{esc(h)}}}" for h in header) + " \\\\"
    if longtable:
        tex = (f"{{{size}\n\\begin{{longtable}}{{{colspec}}}\n\\caption{{{esc(caption)}}}\\label{{{label}}}\\\\\n\\toprule\n{head}\n\\midrule\n\\endfirsthead\n"
               f"\\toprule\n{head}\n\\midrule\n\\endhead\n" + "\n".join(texrows) + "\n\\bottomrule\n\\end{longtable}\n}\n")
    else:
        env = "table*" if wide else "table"
        tex = (f"\\begin{{{env}}}[!t]\n\\centering\n\\caption{{{esc(caption)}}}\\label{{{label}}}\n{size}\n"
               f"\\setlength{{\\tabcolsep}}{{3pt}}\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{head}\n\\midrule\n"
               + "\n".join(texrows) + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")
    (TEX / f"{name}.tex").write_text(tex)


def ts(x):
    return M if x is None else f"{x:,.0f}"


# T_all: every trained model
body = []
for r in sorted(rows, key=lambda r: (["v1", "v2", "v3", "v4"].index(r["version"]), r["task"], r["model"])):
    na = "n/a" if r["note"] in ("val merged into train", "test-only push run") else M
    body.append([r["version"], "full 39f" if r["data"] == "full" else "Kaggle", r["task"], r["model"], r["config"],
                 f4(r["val_acc"]) if r["val_acc"] is not None else na, f4(r["val_mf1"]) if r["val_mf1"] is not None else na,
                 f4(r["test_acc"]), f4(r["test_mf1"]), ts(r["time_s"])])
write_table("all_models", ["Ver", "Data", "Task", "Model", "Config", "Val acc", "Val mF1", "Test acc", "Test mF1", "Train s"],
            body, "Every model trained in this project (v1-v4). Val = validation split, mF1 = macro-F1. n/a: no separate validation (train+val run, or test-only push run). [MISSING]: not recorded in the result JSON.",
            "tab:allmodels", "llrp{3.2cm}p{4.4cm}rrrrr", size="\\tiny", longtable=True)

# T_posthoc
body = [[v, t, n, f4(vm) if vm is not None else "n/a", f4(m["accuracy"]), f4(m["macro_f1"])] for v, t, n, vm, m, s in posthoc]
write_table("posthoc", ["Ver", "Task", "Method", "Val mF1 (selection)", "Test acc", "Test mF1"], body,
            "Post-hoc methods on the full release (threshold tuning A1, calibration A2, ensembles and stacking A3). The method is chosen by validation macro-F1 only.",
            "tab:posthoc", "llp{5.2cm}rrr", wide=True)

# T_best vs paper
body = []
regimes = [("Full release, 39 features (v1-v3)", {t: (full_best[t][0], full_best[t][1]) for t in (2, 8, 34)})]
names = {"38f": "Kaggle subset, 38 features (v4)", "46f": "Kaggle subset, 46 features (v4)",
         "38nf": "Kaggle subset, 38 features, no IAT (v4)", "46nf": "Kaggle subset, 46 features, no IAT (v4)"}
for fs in ("46f", "46nf"):
    pretty = {"rf": "Random Forest (200 trees)", "xgb_best": "XGBoost (prior best config)", "xgb_unw": "XGBoost (unweighted)"}
    regimes.append((names[fs], {t: (pretty[v4s["selected"][f"{t}c_{fs}"]["model"]], v4s["selected"][f"{t}c_{fs}"]["test"]) for t in (2, 8, 34)}))
for reg, d in regimes:
    for t in (2, 8, 34):
        mname, m = d[t]
        key = "accuracy" if t == 2 else "macro_f1"
        pv = PAPER[t][0] if t == 2 else PAPER[t][1]
        delta = m[key] - pv
        body.append([reg, f"{t}-class", mname, f4(m["accuracy"]), f4(m["macro_f1"]), f"{f4(PAPER[t][0])} / {f4(PAPER[t][1])}",
                     ("acc " if t == 2 else "mF1 ") + f"{delta:+.4f}", "\\textbf{BEAT}" if delta > 0 else "NOT BEAT"])
write_table("best_vs_paper", ["Setting", "Task", "Model (selected on val)", "Test acc", "Test mF1", "Paper acc / F1",
                              "Δ (judged metric)", "Verdict"], body,
            "Best model per task against Neto et al. 2023 (RF, Table 6). 2-class is judged on accuracy, 8- and 34-class on macro-F1. Δ = ours minus paper.",
            "tab:best", "p{3.4cm}lp{3.6cm}rrrrl")

# T_v4 ablation
body = []
for t in (2, 8, 34):
    k = "accuracy" if t == 2 else "macro_f1"
    cells = []
    for fs in ("38f", "46f", "38nf", "46nf"):
        s = v4s["selected"][f"{t}c_{fs}"]
        v = v4s["verdicts"][f"{t}c_{fs}"]
        cells.append(f"{f4(s['test']['accuracy'])} / {f4(s['test']['macro_f1'])} ({'BEAT' if v == 'BEAT' else 'NOT'})")
    body.append([f"{t}-class ({'acc' if t == 2 else 'mF1'})"] + cells + [f"{f4(PAPER[t][0])} / {f4(PAPER[t][1])}"])
write_table("v4_ablation", ["Task (judged on)", "38 feat.", "46 feat.", "38 feat., no IAT", "46 feat., no IAT", "Paper RF"], body,
            "v4 ablation on the same rows of the Kaggle subset (80/20 + 5% val, seed 42). Cells: test accuracy / test macro-F1 of the model selected on validation; BEAT/NOT against the paper on the judged metric.",
            "tab:v4", "lccccc")

# T_multiseed
body = []
for t in (2, 8, 34):
    body.append([f"{t}-class", ms[t]["method"], ", ".join(map(str, ms[t]["seeds"])),
                 f"{f4(ms[t]['mean_accuracy'])} $\\pm$ {ms[t]['std_accuracy']:.6f}", f"{f4(ms[t]['mean_macro_f1'])} $\\pm$ {ms[t]['std_macro_f1']:.6f}"])
hacc = [load(f"novelty/v3/hierarchical_s{s}.json")["hard_test"]["accuracy"] for s in (1, 2, 42)]
body.append(["34-class", "hierarchical 8$\\rightarrow$34 (hard)", "1, 2, 42",
             f"{f4(float(np.mean(hacc)))} $\\pm$ {float(np.std(hacc)):.6f}", f"{f4(float(np.mean(hier)))} $\\pm$ {float(np.std(hier)):.6f}"])
for t in (2, 34):
    cbs = [r for r in rows if r["task"] == t and r["model"].startswith("catboost_c1_none_s")]
    body.append([f"{t}-class", "CatBoost c1, no weights", ", ".join(r["model"].split("_s")[-1] for r in cbs),
                 f"{f4(float(np.mean([r['test_acc'] for r in cbs])))} $\\pm$ {float(np.std([r['test_acc'] for r in cbs])):.6f}",
                 f"{f4(float(np.mean([r['test_mf1'] for r in cbs])))} $\\pm$ {float(np.std([r['test_mf1'] for r in cbs])):.6f}"])
write_table("multiseed", ["Task", "Model", "Seeds", "Test acc (mean $\\pm$ std)", "Test mF1 (mean $\\pm$ std)"], body,
            "Multi-seed results on the full release. Std is the population std over seeds; BEST rows from results/{task}_class/BEST/multi_seed_results.json, others computed from the per-seed JSONs.",
            "tab:multiseed", "llp{1.6cm}cc", wide=True)

# T_novelty summary
body = [
    ["N1 twins", "test rows with an exact train twin", f"{FACTS['nOneTwin']['value']} (full) / {FACTS['vfTwin']['value']} (v4)", "results/novelty/N1_twin_analysis/results.json; results/v4_46feat/summary.json"],
    ["N2 protocol ladder", "8-class mF1: paper protocol → file-grouped (twin-free)", f"{FACTS['nTwoAMf']['value']} → {FACTS['nTwoCMf']['value']} ({FACTS['nTwoDMf']['value']})", "results/novelty/N2_protocol_ladder/results.json"],
    ["N3 DoS vs DDoS", "share of 8-class test errors (full / v4 no-IAT)", f"{FACTS['nThreeShare']['value']} / {FACTS['vfDdNoShare']['value']}", "results/novelty/v3/dos_ddos.json; results/v4_46feat/summary.json"],
    ["N3 merge", "8-class acc → 7-class (DoS+DDoS merged) acc", f"{FACTS['nThreeEightAcc']['value']} → {FACTS['nThreeFloodAcc']['value']}", "results/novelty/v3/dos_ddos.json"],
    ["N4 IAT", "IAT-only depth-8 tree, 8-class val acc (majority baseline)", f"{FACTS['iatAcc']['value']} ({FACTS['iatMaj']['value']})", "results/v4_46feat/iat_probe.json"],
    ["N5 window", "capture-window classifier accuracy", FACTS["nFiveAcc"]["value"], "results/novelty/N5_window_size/results.json"],
    ["N6 hierarchical", "34-class test mF1, hierarchical (3 seeds) vs flat", f"{FACTS['hierMf']['value']} vs {FACTS['flatMfThirtyFour']['value']}", "results/novelty/v3/hierarchical_s*.json"],
]
write_table("novelty", ["Analysis", "Measure", "Value", "Evidence (JSON)"], body,
            "Summary of the analyses N1-N6 with their evidence files.", "tab:novelty", "lp{4.2cm}p{3.2cm}p{5.4cm}")

# T_n1 per class, T_n2 ladder, T_iat bands
body = [[c, f"{v:.2f}\\%"] for c, v in sorted(n1["per_class_twin_pct"].items(), key=lambda kv: -kv[1])]
write_table("n1_twins", ["8-class label", "Test rows with exact train twin"], body,
            "N1: twin rate per class on the full release (75/5/20 split).", "tab:n1", "lr", wide=False)
body = [[lab, f4(n2[k]["accuracy"]), f4(n2[k]["macro_f1"])] for lab, k in [
    ("(a) paper protocol: shuffled rows, no de-dup", "protocol_a_paper"), ("(b) + de-duplication", "protocol_b_dedup"),
    ("(c) file-grouped split", "protocol_c_file_grouped"), ("(d) file-grouped + twin-free", "protocol_d_file_grouped_twin_free")]]
write_table("n2_ladder", ["Protocol (8-class, XGBoost balanced_deep)", "Test acc", "Test mF1"], body,
            "N2: protocol ladder on the full release.", "tab:n2", "p{4.6cm}rr", wide=False)
pick = ["DoS-TCP_Flood", "DoS-SYN_Flood", "DoS-UDP_Flood", "DDoS-TCP_Flood", "DDoS-SYN_Flood", "DDoS-ICMP_Flood",
        "Mirai-greeth_flood", "BenignTraffic", "DNS_Spoofing", "XSS"]
bands = ip["iat_per_class_quantiles_raw"]
body = [[c, f"{bands[c]['p01']:.4g}", f"{bands[c]['p50']:.4g}", f"{bands[c]['p99']:.4g}"] for c in pick]
write_table("iat_bands", ["Raw label (Kaggle)", "IAT 1st pct", "IAT median", "IAT 99th pct"], body,
            "N4: per-class IAT quantiles in the Kaggle subset (selected classes; all 34 in results/v4_46feat/iat_probe.json).",
            "tab:iat", "lrrr", wide=False)

# T_class counts
c8full, c8v4 = dm["class8_counts"], v4cc["class8"]
body = [[c, num(c8full[c]), num(c8v4[c])] for c in dm["class8_counts"]]
body.append(["Total (2-class: Benign / Attack)", f"{num(dm['total_rows'])} ({FACTS['fullBenign']['value']} / {FACTS['fullAttack']['value']})",
             f"{num(v4cc['n_rows'])} ({FACTS['vfBenign']['value']} / {FACTS['vfAttack']['value']})"])
write_table("class_counts", ["8-class label", "Full release (39 feat.)", "Kaggle subset (46 feat.)"], body,
            "Rows per 8-class label in the two dataset versions used. 34-class counts are listed in the appendix of the report.",
            "tab:counts", "lrr", wide=False)
body = [[c, num(dm["class34_counts"][c]), num(v4cc["class34"].get(
    {"Benign_Final": "Benign_Final"}.get(c, c), 0))] for c in sorted(dm["class34_counts"])]
write_table("class34_counts", ["34-class label", "Full release", "Kaggle subset"], body,
            "Rows per 34-class label (Kaggle labels mapped to our names).", "tab:counts34", "lrr", wide=False)

# T_idps policy + metrics
pol = json.loads((REPO / "src/prevention/policy.json").read_text())["actions"]
body = [[c, a] for c, a in pol.items()]
write_table("idps_policy", ["8-class label", "Action"], body, "IDPS policy (src/prevention/policy.json).", "tab:policy", "ll", wide=False)
body = [[a, f4(v["precision"]), f4(v["recall"]), num(v["support"]), num(v["taken"])] for a, v in al["per_action"].items()]
write_table("idps_actions", ["Action", "Precision", "Recall", "Support", "Taken"], body,
            f"IDPS per-action precision and recall on {FACTS['idsRows']['value']} test flows.", "tab:idps", "lrrrr", wide=False)

# ---------------------------------------------------------------- outputs
(REPO / "paper/data/facts.json").write_text(json.dumps(FACTS, indent=1))
(REPO / "paper/data/all_models.json").write_text(json.dumps(rows, indent=1))
dedup = []
for m in MISSING:
    if m not in dedup:
        dedup.append(m)
(REPO / "paper/data/missing.json").write_text(json.dumps(dedup, indent=1))
items = "\n".join(f"\\item {esc(m['key'])} (\\texttt{{{esc(m['source'])}}})" for m in dedup)
(TEX / "missing.tex").write_text("\\begin{itemize}\\footnotesize\n" + items + "\n\\end{itemize}\n")
(MD / "missing.md").write_text("\n".join(f"- {m['key']} (`{m['source']}`)" for m in dedup) + "\n")


def macro_name(k):
    for a, b in [("34", "ThirtyFour"), ("2", "Two"), ("8", "Eight")]:
        k = k.replace(a, b)
    return re.sub(r"[^A-Za-z]", "", k)


mac = ["% Auto-generated by paper/tools/build_facts.py from the result JSONs. Do not edit by hand."]
for k, v in FACTS.items():
    mac.append(f"\\newcommand{{\\{macro_name(k)}}}{{{v['value']}}}")
(REPO / "paper/macros.tex").write_text("\n".join(mac) + "\n")
print(len(rows), "model rows;", len(posthoc), "post-hoc rows;", len(FACTS), "facts;", len(dedup), "missing")
for m in dedup:
    print("  MISSING:", m["key"], "<-", m["source"])
