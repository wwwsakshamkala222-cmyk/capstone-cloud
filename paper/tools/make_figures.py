"""Make the paper/report figures that do not exist yet (300 dpi, from the result JSONs) and copy the rest.

New: fig_iat_probe.png (results/v4_46feat/iat_probe.json), fig_dosddos_merge.png (results/novelty/v3/dos_ddos_balanced_deep.json
and results/v4_46feat/summary.json). Copied: existing figures from figures/ into paper/figures and report/figures.
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUTS = [REPO / "paper/figures", REPO / "report/figures"]
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9", "#ffffff"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    for o in OUTS:
        fig.savefig(o / name)
    plt.close(fig)


# IAT probe: single-feature accuracy (top 10) + IAT median per class
ip = json.loads((REPO / "results/v4_46feat/iat_probe.json").read_text())
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
top = ip["single_feature_probe"][:10][::-1]
a.barh([r["feature"] for r in top], [r["val_accuracy"] for r in top],
       color=[ORANGE if r["feature"] == "IAT" else "#c3c2b7" for r in top], height=0.7)
a.axvline(ip["majority_class_val_accuracy"], color=INK, ls="--", lw=1)
a.text(ip["majority_class_val_accuracy"], -0.9, f" majority class {ip['majority_class_val_accuracy']:.4f}", fontsize=7, color=MUTED)
for i, r in enumerate(top):
    a.text(r["val_accuracy"] + 0.003, i, f"{r['val_accuracy']:.4f}", va="center", fontsize=7, color=INK)
a.set_xlim(0.6, 1.04)
a.set_xlabel("8-class validation accuracy, one feature, depth-8 tree")
a.set_title("(a) Single-feature probe (v4 subset)", loc="left", fontsize=9, color=INK)
bands = ip["iat_per_class_quantiles_raw"]
labs = list(bands)
med = [bands[c]["p50"] for c in labs]
fam = [ORANGE if c.startswith("DoS") else BLUE if c.startswith("DDoS") else "#c3c2b7" for c in labs]
b.scatter(med, range(len(labs)), c=fam, s=14, zorder=3)
for i, c in enumerate(labs):
    b.plot([bands[c]["p01"], bands[c]["p99"]], [i, i], color=fam[i], lw=1, alpha=0.6)
b.set_yticks(range(len(labs)), labs, fontsize=5.5)
b.set_xscale("log")
b.set_xlim(1e-7, 1e9)
b.set_xlabel("raw IAT value (median, line = 1st-99th percentile)")
b.set_title("(b) IAT per class: orange DoS, blue DDoS", loc="left", fontsize=9, color=INK)
b.xaxis.grid(True, color=GRID, lw=0.5)
save(fig, "fig_iat_probe.png")

# DoS/DDoS merge: full release (v3) and v4 no-IAT
dd = json.loads((REPO / "results/novelty/v3/dos_ddos_balanced_deep.json").read_text())
v4 = json.loads((REPO / "results/v4_46feat/summary.json").read_text())
fig, (a, b) = plt.subplots(1, 2, figsize=(8.5, 3.2))
x = np.arange(2)
e8, e7 = dd["test_8class"], dd["test_7class_flood"]
for k, (lab, off, col) in enumerate([("8-class", -0.18, "#c3c2b7"), ("DoS+DDoS merged (7-class)", 0.18, BLUE)]):
    m = e8 if k == 0 else e7
    vals = [m["accuracy"], m["macro_f1"]]
    bars = a.bar(x + off, vals, 0.34, color=col, label=lab)
    for bx, v in zip(bars, vals):
        a.text(bx.get_x() + bx.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=7, color=INK)
a.set_xticks(x, ["test accuracy", "test macro-F1"])
a.set_ylim(0, 1.1)
a.legend(frameon=False, fontsize=7, loc="lower right")
a.set_title("(a) Re-scoring the same predictions (full release)", loc="left", fontsize=9, color=INK)
shares = [dd["test_8class_dos_ddos_cells"]["share_of_all_8class_errors"],
          v4["dos_ddos"]["xgb_best_8c_46f"]["share_of_errors"], v4["dos_ddos"]["xgb_best_8c_46nf"]["share_of_errors"]]
labs = ["full release\n39 feat.", "v4 subset\n46 feat. (with IAT)", "v4 subset\n46 feat., no IAT"]
bars = b.bar(range(3), [s * 100 for s in shares], 0.55, color=[BLUE, ORANGE, BLUE])
for bx, s in zip(bars, shares):
    b.text(bx.get_x() + bx.get_width() / 2, s * 100 + 1.5, f"{s * 100:.2f}%", ha="center", fontsize=7, color=INK)
b.set_xticks(range(3), labs, fontsize=7.5)
b.set_ylim(0, 110)
b.set_ylabel("DoS<->DDoS share of 8-class test errors (%)")
b.set_title("(b) DoS<->DDoS share of all errors", loc="left", fontsize=9, color=INK)
save(fig, "fig_dosddos_merge.png")

COPY = {"confusion_matrix_8class.png": "fig_cm8_full.png", "confusion_matrix_34class.png": "fig_cm34_full.png",
        "per_class_f1_8class.png": "fig_perclass_f1_8.png", "protocol_ladder.png": "fig_protocol_ladder.png",
        "twin_analysis.png": "fig_twins.png", "v4/v4_top15_importance_46f.png": "fig_v4_importance.png",
        "v4/v4_ablation_headline.png": "fig_v4_ablation.png", "v4/v4_8class_confusion.png": "fig_v4_cm8.png",
        "prevention/action_confusion.png": "fig_idps_actions.png", "prevention/actions_taken.png": "fig_idps_taken.png",
        "v3/34class_flat_vs_hier_f1.png": "fig_hier_f1.png",
        "training_times.png": "fig_training_times.png"}
for src, dst in COPY.items():
    for o in OUTS:
        shutil.copy(REPO / "figures" / src, o / dst)
print(sorted(p.name for p in OUTS[0].iterdir()))
