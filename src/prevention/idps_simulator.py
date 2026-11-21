"""IDPS simulation: replay 200k v4 TEST rows in random order as live traffic.

For each flow: classify with the deployed no-IAT 8-class model, map the class to an action with
policy.json, and for BLOCK-type actions generate (never run) an iptables rule with a placeholder IP.

Classification runs in micro-batches of 1000 rows in stream order (the "batch" latency mode);
single-row latency is measured separately by classifying the first 2,000 streamed rows one at a
time. Writes results/prevention/{summary.json, actions_log.csv, demo_flows.csv} and
figures/prevention/*.png.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from idps import ACTIONS, DEFAULT_MODEL_DIR, Detector, Policy, placeholder_ip

REPO = Path(__file__).resolve().parents[2]
RES, FIG = REPO / "results" / "prevention", REPO / "figures" / "prevention"
RAW = Path("/dev/shm/saksham/v4/merged_raw.parquet")
RENAME = {"BenignTraffic": "Benign_Final", "DDoS-PSHACK_Flood": "DDoS-PSHACK_FLOOD",
          "DDoS-RSTFINFlood": "DDoS-RSTFINFLOOD"}
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def load_stream(model_dir, n, seed):
    cm = json.loads((REPO / "src/preprocessing/class_maps.json").read_text())
    idx_te = np.load(Path(model_dir) / "idx_test.npy")
    rng = np.random.default_rng(seed)
    order = rng.permutation(idx_te)[:n]  # random order, like live traffic
    df = pd.read_parquet(RAW).iloc[order]
    y8 = df["label"].replace(RENAME).map(cm["folder_to_8class"]).to_numpy()
    return order, df, y8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nthread", type=int, default=8)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    det, pol = Detector(a.model_dir, nthread=a.nthread), Policy()
    flow_ids, df, y_true = load_stream(a.model_dir, a.rows, a.seed)
    X = df[det.features].to_numpy(np.float64)
    n = len(X)

    # Warm-up (first call builds caches), then single-row latency on the first 2,000 flows.
    det.predict(X[:10])
    single = []
    for i in range(2000):
        t0 = time.perf_counter()
        det.predict(X[i])
        single.append((time.perf_counter() - t0) * 1e3)
    single = np.array(single)

    # Stream in micro-batches of 1000 (latency per batch, divided by 1000 for per-row).
    preds, confs, batch_ms = np.empty(n, np.int64), np.empty(n), []
    for s in range(0, n, 1000):
        t0 = time.perf_counter()
        p, c = det.predict(X[s:s + 1000])
        batch_ms.append((time.perf_counter() - t0) * 1e3)
        preds[s:s + 1000], confs[s:s + 1000] = p, c
    batch_ms = np.array(batch_ms)
    y_pred = np.array(det.classes)[preds]

    act_true = np.array([pol.action(c) for c in y_true])
    act_pred = np.array([pol.action(c) for c in y_pred])

    # Action log: every 20th flow (keeps the file small) with its generated rule.
    with open(RES / "actions_log.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stream_pos", "flow_id", "true_class", "pred_class", "confidence", "action",
                    "correct_action", "placeholder_src_ip", "generated_rule"])
        for i in range(0, n, 20):
            ip = placeholder_ip(flow_ids[i])
            w.writerow([i, int(flow_ids[i]), y_true[i], y_pred[i], f"{confs[i]:.3f}", act_pred[i],
                        act_pred[i] == act_true[i], ip, pol.rule(y_pred[i], ip)])
    n_rules = int(sum(1 for c in y_pred if pol.rule(c, "x")))

    # Small committed sample of raw test flows for demo.py (no labels needed by the model).
    demo = df[det.features].iloc[:2000].copy()
    demo.insert(0, "flow_id", flow_ids[:2000])
    demo["true_class"] = y_true[:2000]
    demo.to_csv(RES / "demo_flows.csv", index=False, float_format="%.6g")

    benign, attack = y_true == "Benign", y_true != "Benign"
    blocked = np.char.startswith(act_pred.astype(str), "BLOCK")
    prec, rec, f1, sup = precision_recall_fscore_support(act_true, act_pred, labels=ACTIONS, zero_division=0)
    acc8 = float(accuracy_score(y_true, y_pred))
    acc_act = float((act_true == act_pred).mean())
    wrong8 = y_true != y_pred
    dosddos = wrong8 & np.isin(y_true, ["DoS", "DDoS"]) & np.isin(y_pred, ["DoS", "DDoS"])
    absorbed = wrong8 & (act_true == act_pred)
    # Same predictions scored with DoS and DDoS merged into one class.
    merge = lambda a: np.where(np.isin(a, ["DoS", "DDoS"]), "DoS+DDoS", a)
    acc_merged = float((merge(y_true) == merge(y_pred)).mean())

    summary = {
        "model": det.meta["model"], "features": len(det.features), "IAT_excluded": "IAT" not in det.features,
        "model_val_metrics": det.meta["val_metrics"],
        "stream": {"source": "v4 TEST split (never used for training or selection)", "rows": n,
                   "order": f"random permutation, seed {a.seed}", "class_counts": pd.Series(y_true).value_counts().to_dict()},
        "plain_8class": {"accuracy": acc8, "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                         "errors": int(wrong8.sum())},
        "dos_ddos_merged_8class_accuracy": acc_merged,
        "action_level": {
            "accuracy": acc_act,
            "errors": int((act_true != act_pred).sum()),
            "class_errors_with_correct_action": int(absorbed.sum()),
            "of_which_dos_ddos_confusions": int(dosddos.sum()),
            "gain_vs_plain_8class_accuracy": acc_act - acc8,
            "false_block_rate": float(blocked[benign].mean()),
            "false_block_count": int(blocked[benign].sum()),
            "miss_rate": float((act_pred[attack] == "ALLOW").mean()),
            "miss_count": int((act_pred[attack] == "ALLOW").sum()),
            "per_action": {act: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s),
                                 "taken": int((act_pred == act).sum())}
                           for act, p, r, f, s in zip(ACTIONS, prec, rec, f1, sup)},
        },
        "rules_generated": n_rules,
        "latency_ms_per_row": {
            "cpu_threads": a.nthread,
            "single_row": {"avg": float(single.mean()), "p50": float(np.percentile(single, 50)),
                           "p99": float(np.percentile(single, 99)), "n": len(single)},
            "batch_1000": {"avg": float(batch_ms.mean() / 1000), "p99": float(np.percentile(batch_ms, 99) / 1000),
                           "avg_batch_ms": float(batch_ms.mean()), "n_batches": len(batch_ms)},
            "includes": "scaling + XGBoost inference on CPU; excludes policy lookup and logging",
        },
        "notes": ["Blocking is simulated: CICIoT2023 has no IP addresses, so rules use placeholder IPs from "
                  "203.0.113.0/24 and are generated, never executed.",
                  "The deployed model excludes IAT, which acts as a capture timestamp in CICIoT2023."],
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))

    # Figures
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.spines.top": False,
                         "axes.spines.right": False})
    cmx = confusion_matrix(act_true, act_pred, labels=ACTIONS)
    cmn = cmx / np.maximum(cmx.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i, j]:.3f}\n({cmx[i, j]:,})", ha="center", va="center", fontsize=8,
                    color="#ffffff" if cmn[i, j] > 0.5 else INK)
    ax.set_xticks(range(4), ACTIONS)
    ax.set_yticks(range(4), ACTIONS)
    ax.set_xlabel("action taken (from predicted class)")
    ax.set_ylabel("correct action (from true class)")
    ax.set_title(f"IDPS actions on {n:,} test flows: action accuracy {acc_act:.4f}", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "action_confusion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(4)
    tk = [(act_pred == a_).sum() for a_ in ACTIONS]
    tr = [(act_true == a_).sum() for a_ in ACTIONS]
    b1 = ax.bar(x - 0.2, tk, 0.38, color=BLUE, label="taken by IDPS", edgecolor=SURF, linewidth=2)
    b2 = ax.bar(x + 0.2, tr, 0.38, color=ORANGE, label="correct (true class)", edgecolor=SURF, linewidth=2)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height()):,}", ha="center",
                    va="bottom", fontsize=7.5, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x, ACTIONS)
    ax.set_ylabel("flows (log scale)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.set_title(f"Actions taken vs correct actions ({n:,} test flows)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "actions_taken.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
