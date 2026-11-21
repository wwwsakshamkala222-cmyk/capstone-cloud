"""v4 report: 38- vs 46-feature ablation on the same rows (with and without IAT), vs Neto et al. 2023.

Selection per task and feature set is by VALIDATION only (2-class: accuracy, matching the paper's
headline; 8/34-class: macro-F1). Test numbers are reported for every run and the selected run is
marked. Writes results/v4_46feat/ and figures/v4/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("/dev/shm/saksham/v4/runs")
META = json.loads(Path("/dev/shm/saksham/v4/meta.json").read_text())
REPO = Path(__file__).resolve().parents[2]
RES, FIG = REPO / "results" / "v4_46feat", REPO / "figures" / "v4"
PAPER = {2: (0.9968, 0.9653), 8: (0.9944, 0.7193), 34: (0.9916, 0.7140)}  # (acc, macro-F1)
V3 = {2: (0.9937, 0.9339), 8: (0.8415, 0.7308), 34: (0.7895, 0.6452)}    # full release, 39f, 75/5/20 (8-class: xgboost_balanced_deep, val-selected)
EXTRA = set(META["extra_8"])
FSETS = ("38", "46", "38n", "46n")  # n = IAT dropped
FLABEL = {"38": "38 features", "46": "46 features", "38n": "38 features, no IAT", "46n": "46 features, no IAT"}
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})


def sel_key(task):
    return "accuracy" if task == 2 else "macro_f1"


def target(task):
    return PAPER[task][0] if task == 2 else PAPER[task][1]


def main():
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]
    for r in runs:
        r["fset"] = str(r["fset"])  # early runs stored ints
        (RES / f"{r['name']}.json").write_text(json.dumps(r, indent=1))

    selected, rows = {}, []
    for task in (2, 8, 34):
        for fset in FSETS:
            cand = [r for r in runs if r["task"] == task and r["fset"] == fset]
            if not cand:
                continue
            best = max(cand, key=lambda r: r["val_metrics"][sel_key(task)])
            selected[(task, fset)] = best
            for r in cand:
                rows.append((task, fset, r["model"], r["val_metrics"], r["test_metrics"], r is best))

    summary = {"selected": {}, "verdicts": {}, "paper": PAPER, "v3_full_release": V3,
               "selection_rule": "validation accuracy (2-class) / validation macro-F1 (8, 34-class)"}
    for (task, fset), b in selected.items():
        v = "BEAT" if b["test_metrics"][sel_key(task)] > target(task) else "NOT BEAT"
        summary["selected"][f"{task}c_{fset}f"] = {"model": b["model"], "val": b["val_metrics"],
                                                   "test": b["test_metrics"]}
        summary["verdicts"][f"{task}c_{fset}f"] = v

    def headline(fa, fb):
        md = [f"| Task | Judged on | v3: full release, 39f (75/5/20) | v4 {FLABEL[fa]} | v4 {FLABEL[fb]} | Neto et al. 2023 RF | {fa}f vs paper | {fb}f vs paper |",
              "|---|---|---|---|---|---|---|---|"]
        for task in (2, 8, 34):
            cells, verd = [], []
            for fset in (fa, fb):
                b = selected.get((task, fset))
                if b is None:
                    cells.append("—")
                    verd.append("—")
                    continue
                t = b["test_metrics"]
                cells.append(f"{t['accuracy']:.4f} / {t['macro_f1']:.4f} ({b['model']})")
                v = summary["verdicts"][f"{task}c_{fset}f"]
                verd.append(f"**{v}**" if v == "BEAT" else v)
            pa, pf = PAPER[task]
            md.append(f"| {task}-class | {'accuracy' if task == 2 else 'macro-F1'} | {V3[task][0]:.4f} / {V3[task][1]:.4f} "
                      f"| {cells[0]} | {cells[1]} | {pa:.4f} / {pf:.4f} | {verd[0]} | {verd[1]} |")
        return md

    md, md_n = headline("38", "46"), headline("38n", "46n")

    allruns = ["| Task | Features | Model | Val acc | Val mF1 | Test acc | Test mF1 | Test wF1 | Selected (val) |",
               "|---|---|---|---|---|---|---|---|---|"]
    for task, fset, model, v, t, s in sorted(rows, key=lambda r: (r[0], FSETS.index(r[1]), r[2])):
        allruns.append(f"| {task} | {FLABEL[fset]} | {model} | {v['accuracy']:.4f} | {v['macro_f1']:.4f} | "
                       f"{t['accuracy']:.4f} | {t['macro_f1']:.4f} | {t['weighted_f1']:.4f} | {'✓' if s else ''} |")

    # --- DoS vs DDoS (8-class test confusion) ---
    c8 = META["class8_order"]
    iD, iS = c8.index("DDoS"), c8.index("DoS")
    dd = {}
    for r in runs:
        if r["task"] != 8:
            continue
        cm = np.array(r["test_confusion"])
        errors = int(cm.sum() - np.trace(cm))
        conf = int(cm[iS, iD] + cm[iD, iS])
        dd[r["name"]] = {"fset": r["fset"], "model": r["model"], "DoS_to_DDoS": int(cm[iS, iD]),
                         "DDoS_to_DoS": int(cm[iD, iS]), "dos_ddos_total": conf, "all_errors": errors,
                         "share_of_errors": conf / errors if errors else 0.0,
                         "selected": r is selected.get((8, r["fset"]))}
    summary["dos_ddos"] = dd
    ddmd = ["| Features | Model | DoS→DDoS | DDoS→DoS | Total | Share of all 8-class errors |", "|---|---|---|---|---|---|"]
    for k, d in sorted(dd.items(), key=lambda kv: (FSETS.index(kv[1]["fset"]), kv[1]["model"])):
        ddmd.append(f"| {FLABEL[d['fset']]} | {d['model']}{' ✓' if d['selected'] else ''} | {d['DoS_to_DDoS']:,} | "
                    f"{d['DDoS_to_DoS']:,} | {d['dos_ddos_total']:,} | {d['share_of_errors']:.1%} |")

    # --- importances (selected 46f models) ---
    imps = {}
    for task in (2, 8, 34):
        for fset in ("46", "46n"):
            b = selected.get((task, fset))
            if b:
                imps[f"{task}c_{fset}f"] = {
                    "model": b["model"], "top15": list(b["feature_importance"].items())[:15],
                    "extra8_share_total": float(sum(v for f, v in b["feature_importance"].items() if f in EXTRA)),
                    "IAT_share": float(b["feature_importance"].get("IAT", 0.0))}
    summary["importance"] = imps
    summary["twins"] = {"46f": META["twins_46"], "38f": META["twins_38"]}
    summary["split"] = {k: META[k] for k in ("n_rows", "n_train", "n_val", "n_test")}
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))

    figs = []
    # 1) headline metric per task, 4 feature sets, paper as dashed reference
    fig, ax = plt.subplots(figsize=(9, 5.0))
    x = np.arange(3)
    series = [("38", ORANGE, None), ("46", BLUE, None), ("38n", ORANGE, "///"), ("46n", BLUE, "///")]
    for j, (fset, col, hatch) in enumerate(series):
        vals = [selected[(t, fset)]["test_metrics"][sel_key(t)] if (t, fset) in selected else np.nan for t in (2, 8, 34)]
        bars = ax.bar(x + (j - 1.5) * 0.19, vals, 0.17, color=col if hatch is None else SURF,
                      edgecolor=col, hatch=hatch, linewidth=1.2, label=FLABEL[fset])
        for bx, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bx.get_x() + bx.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=6.5, color=INK, rotation=90)
    for i, t in enumerate((2, 8, 34)):
        ax.plot([i - 0.42, i + 0.42], [target(t)] * 2, color=INK, lw=1.3, ls="--")
        ax.text(i + 0.44, target(t), f"paper\n{target(t):.3f}", va="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(x, ["2-class\n(test accuracy)", "8-class\n(test macro-F1)", "34-class\n(test macro-F1)"])
    ax.set_ylim(0, 1.18)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8, ncol=4)
    ax.set_title("v4, same rows: 38 vs 46 features, with and without IAT, vs Neto et al. 2023 RF",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "v4_ablation_headline.png", dpi=150)
    plt.close(fig)
    figs.append("v4_ablation_headline.png")

    # 2) top-15 importances of the selected 46-feature models (extra-8 in blue, IAT in orange)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, task in zip(axes, (2, 8, 34)):
        key = f"{task}c_46f"
        if key not in imps:
            ax.axis("off")
            continue
        top = imps[key]["top15"][::-1]
        names = [f for f, _ in top]
        cols = [BLUE if f in EXTRA else ORANGE if f == "IAT" else "#c3c2b7" for f in names]
        ax.barh(names, [v for _, v in top], color=cols, height=0.7)
        ax.set_title(f"{task}-class ({imps[key]['model']})", fontsize=9.5, color=INK, loc="left")
        ax.tick_params(axis="y", labelsize=8, colors=INK)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("normalised importance", fontsize=8)
    fig.suptitle("Top-15 feature importances, selected 46-feature models  (blue = paper-only feature, orange = IAT)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "v4_top15_importance_46f.png", dpi=150)
    plt.close(fig)
    figs.append("v4_top15_importance_46f.png")

    # 3) 8-class confusion (row-normalised), selected model per feature set
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    for ax, fset in zip(axes.ravel(), FSETS):
        b = selected.get((8, fset))
        if b is None:
            ax.axis("off")
            continue
        cm = np.array(b["test_confusion"], float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(8):
            for j in range(8):
                if cmn[i, j] >= 0.01:
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="#ffffff" if cmn[i, j] > 0.5 else INK)
        ax.set_xticks(range(8), c8, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(8), c8, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        d = dd[b["name"]]
        ax.set_title(f"{FLABEL[fset]} ({b['model']}): DoS↔DDoS = {d['dos_ddos_total']:,} "
                     f"({d['share_of_errors']:.0%} of errors)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "v4_8class_confusion.png", dpi=150)
    plt.close(fig)
    figs.append("v4_8class_confusion.png")

    (RES / "tables.md").write_text(
        "# v4 tables (test acc / test macro-F1; model per cell selected on validation)\n\n"
        "## All features vs our 38\n\n" + "\n".join(md)
        + "\n\n## Without IAT (capture-timestamp proxy removed)\n\n" + "\n".join(md_n)
        + "\n\n## DoS vs DDoS confusions (8-class test set; ✓ = validation-selected model)\n\n" + "\n".join(ddmd)
        + "\n\n## All runs\n\n" + "\n".join(allruns) + "\n")
    print((RES / "tables.md").read_text())
    print(json.dumps({k: (v["model"], [f for f, _ in v["top15"]], round(v["extra8_share_total"], 4),
                          round(v["IAT_share"], 4)) for k, v in imps.items()}, indent=1))
    print("figures:", figs)


if __name__ == "__main__":
    main()
