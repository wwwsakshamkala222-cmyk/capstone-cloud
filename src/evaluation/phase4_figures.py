#!/usr/bin/env python3
"""Phase 4: Generate research paper figures and tables."""
import os
os.environ["OMP_NUM_THREADS"] = "16"
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

IDS_ROOT = Path("/dev/shm/saksham/ids")
RESULTS_ROOT = IDS_ROOT / "results"
NOVELTY_ROOT = IDS_ROOT / "novelty"
FIG_ROOT = IDS_ROOT / "figures"
os.makedirs(FIG_ROOT, exist_ok=True)

with open(IDS_ROOT / "src" / "class_maps.json") as f:
    CLASS_MAPS = json.load(f)

CLASS8_ORDER = CLASS_MAPS["class8_order"]
CLASS34_ORDER = CLASS_MAPS["class34_order"]

# Paper's reported per-class F1 (8-class RF)
PAPER_8CLASS_F1 = {
    "Benign": 0.93, "Brute_Force": 0.11, "DDoS": 1.00, "DoS": 1.00,
    "Mirai": 1.00, "Recon": 0.80, "Spoofing": 0.84, "Web_Based": 0.08,
}
PAPER_8CLASS_MACRO_F1 = 0.7193
PAPER_8CLASS_ACC = 0.9944
PAPER_2CLASS_ACC = 0.9968
PAPER_2CLASS_F1 = 0.9653

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


def load_best(task):
    path = RESULTS_ROOT / task / "BEST" / "results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def fig1_comparison_table():
    """Generate comparison table as text and save."""
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 1: Performance Comparison — Our Best vs. Paper's RF")
    lines.append("=" * 80)

    for task, paper_metrics in [
        ("2_class", {"accuracy": 0.9968, "macro_f1": 0.9653}),
        ("8_class", {"accuracy": 0.9944, "macro_f1": 0.7193}),
        ("34_class", {"accuracy": None, "macro_f1": 0.70}),
    ]:
        best = load_best(task)
        lines.append(f"\n--- {task} ---")
        if best:
            m = best["test_metrics"]
            lines.append(f"  Our best:  {best['method']}")
            lines.append(f"  Accuracy:  {m['accuracy']:.6f}" + (f"  (paper: {paper_metrics['accuracy']:.4f})" if paper_metrics['accuracy'] else ""))
            lines.append(f"  Macro-F1:  {m['macro_f1']:.6f}  (paper: {paper_metrics['macro_f1']:.4f})")

            if "per_class" in m:
                lines.append(f"  Per-class F1:")
                for cls, data in m["per_class"].items():
                    paper_f1 = PAPER_8CLASS_F1.get(cls, None)
                    suffix = f"  (paper: {paper_f1:.2f})" if paper_f1 is not None else ""
                    lines.append(f"    {cls:20s}: {data['f1']:.4f}{suffix}")
        else:
            lines.append("  [Results not available]")

    text = "\n".join(lines)
    with open(FIG_ROOT / "comparison_table.txt", "w") as f:
        f.write(text)
    print(text)


def fig2_confusion_matrix_heatmap(task, n_classes, class_names, figname):
    """Plot confusion matrix heatmap."""
    best = load_best(task)
    if not best:
        print(f"No results for {task}")
        return

    cm = np.array(best["test_metrics"]["confusion_matrix"])

    # Normalize by row (true label)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.4), max(7, n_classes * 0.35)))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))

    if n_classes <= 10:
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)
        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm_norm[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       color=color, fontsize=8)
    else:
        short_names = [n[:12] for n in class_names]
        ax.set_xticklabels(short_names, rotation=90, ha="right", fontsize=6)
        ax.set_yticklabels(short_names, fontsize=6)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {task} ({best['method']})")

    plt.tight_layout()
    plt.savefig(FIG_ROOT / figname)
    plt.close()
    print(f"  Saved {figname}")


def fig3_per_class_f1_bar():
    """Per-class F1 bar chart: ours vs paper (8-class)."""
    best = load_best("8_class")
    if not best:
        print("No 8-class results")
        return

    our_f1 = [best["test_metrics"]["per_class"][c]["f1"] for c in CLASS8_ORDER]
    paper_f1 = [PAPER_8CLASS_F1[c] for c in CLASS8_ORDER]

    x = np.arange(len(CLASS8_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, paper_f1, width, label="Paper (RF)", color="#7fafd4", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, our_f1, width, label=f"Ours ({best['method']})", color="#f4a261", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Comparison — 8-Class Task")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS8_ORDER, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
               ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
               ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(FIG_ROOT / "per_class_f1_8class.png")
    plt.close()
    print("  Saved per_class_f1_8class.png")


def fig4_feature_importance():
    """Top 20 feature importance plot."""
    for task in ["2_class", "8_class", "34_class"]:
        best = load_best(task)
        if not best or "feature_importance_top20" not in best:
            continue

        imp = best["feature_importance_top20"]
        if not imp:
            continue

        names = list(imp.keys())[:20]
        values = [imp[n] for n in names]

        fig, ax = plt.subplots(figsize=(8, 6))
        y_pos = range(len(names) - 1, -1, -1)
        ax.barh(y_pos, values, color="#2a9d8f", edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Gain")
        ax.set_title(f"Top-20 Feature Importance — {task} ({best['method']})")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIG_ROOT / f"feature_importance_{task}.png")
        plt.close()
        print(f"  Saved feature_importance_{task}.png")


def fig5_twin_analysis():
    """Twin analysis visualization."""
    path = NOVELTY_ROOT / "N1_twin_analysis" / "results.json"
    if not path.exists():
        print("N1 results not found")
        return

    with open(path) as f:
        res = json.load(f)

    # Per-class twin percentage
    twin_pct = res["per_class_twin_pct"]
    classes = list(twin_pct.keys())
    pcts = [twin_pct[c] for c in classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of twin percentages
    ax1.bar(range(len(classes)), pcts, color="#e76f51", edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=30, ha="right")
    ax1.set_ylabel("Twin Percentage (%)")
    ax1.set_title("Test-Set Twins per Class")
    ax1.grid(axis="y", alpha=0.3)

    # Twin vs non-twin F1 comparison
    if res.get("twin_subset_metrics") and res.get("nontwin_subset_metrics"):
        twin_f1 = res["twin_subset_metrics"]["per_class_f1"]
        nontwin_f1 = res["nontwin_subset_metrics"]["per_class_f1"]

        common_classes = sorted(set(twin_f1.keys()) & set(nontwin_f1.keys()))
        x = np.arange(len(common_classes))
        width = 0.35

        ax2.bar(x - width/2, [twin_f1[c] for c in common_classes], width,
               label="Twin rows", color="#f4a261", edgecolor="black", linewidth=0.5)
        ax2.bar(x + width/2, [nontwin_f1[c] for c in common_classes], width,
               label="Non-twin rows", color="#264653", edgecolor="black", linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(common_classes, rotation=30, ha="right")
        ax2.set_ylabel("F1 Score")
        ax2.set_title("F1: Twin vs Non-Twin Test Rows")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_ROOT / "twin_analysis.png")
    plt.close()
    print("  Saved twin_analysis.png")


def fig6_protocol_ladder():
    """Protocol ladder comparison table and chart."""
    path = NOVELTY_ROOT / "N2_protocol_ladder" / "results.json"
    if not path.exists():
        print("N2 results not found")
        return

    with open(path) as f:
        res = json.load(f)

    protocols = []
    labels = []
    for key, label in [
        ("protocol_a_paper", "(a) Shuffled"),
        ("protocol_b_dedup", "(b) Deduplicated"),
        ("protocol_c_file_grouped", "(c) File-grouped"),
        ("protocol_d_file_grouped_twin_free", "(d) File-grouped\n+twin-free"),
    ]:
        if res.get(key):
            protocols.append(res[key])
            labels.append(label)

    if not protocols:
        print("No protocol data")
        return

    # Overall metrics
    macro_f1s = [p["macro_f1"] for p in protocols]
    accs = [p["accuracy"] for p in protocols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"][:len(labels)]

    ax1.bar(range(len(labels)), macro_f1s, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Macro F1")
    ax1.set_title("8-Class Macro F1 by Protocol")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(macro_f1s):
        ax1.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)

    ax2.bar(range(len(labels)), accs, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("8-Class Accuracy by Protocol")
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(accs):
        ax2.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_ROOT / "protocol_ladder.png")
    plt.close()
    print("  Saved protocol_ladder.png")

    # Per-class F1 table
    lines = ["Protocol Ladder — Per-Class F1 (8-class)", "=" * 80]
    header = f"{'Class':15s}" + "".join(f"  {l:>14s}" for l in labels)
    lines.append(header)
    for cls in CLASS8_ORDER:
        row = f"{cls:15s}"
        for p in protocols:
            f1 = p.get("per_class_f1", {}).get(cls, 0)
            row += f"  {f1:>14.4f}"
        lines.append(row)
    lines.append(f"{'MACRO F1':15s}" + "".join(f"  {p['macro_f1']:>14.4f}" for p in protocols))

    text = "\n".join(lines)
    with open(FIG_ROOT / "protocol_ladder_table.txt", "w") as f:
        f.write(text)
    print(text)


def fig7_class_weight_sweep():
    """Class weight sensitivity plot."""
    path = NOVELTY_ROOT / "N6_class_weight_sweep" / "results.json"
    if not path.exists():
        print("N6 results not found")
        return

    with open(path) as f:
        res = json.load(f)

    sweep = res["sweep"]
    powers = [s["power"] for s in sweep]
    macro_f1s = [s["macro_f1"] for s in sweep]
    accs = [s["accuracy"] for s in sweep]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#2a9d8f"
    color2 = "#e76f51"

    ax1.set_xlabel("Weight Power (p)")
    ax1.set_ylabel("Macro F1", color=color1)
    ax1.plot(powers, macro_f1s, "o-", color=color1, linewidth=2, markersize=8, label="Macro F1")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color=color2)
    ax2.plot(powers, accs, "s--", color=color2, linewidth=2, markersize=8, label="Accuracy")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    ax1.set_title("Accuracy vs. Fairness Tradeoff (8-class)")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_ROOT / "class_weight_sweep.png")
    plt.close()
    print("  Saved class_weight_sweep.png")

    # Per-class F1 heatmap
    fig2, ax = plt.subplots(figsize=(10, 5))
    data = []
    for s in sweep:
        row = [s["per_class_f1"].get(c, 0) for c in CLASS8_ORDER]
        data.append(row)
    data = np.array(data)

    im = ax.imshow(data.T, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(powers)))
    ax.set_xticklabels([f"p={p}" for p in powers])
    ax.set_yticks(range(8))
    ax.set_yticklabels(CLASS8_ORDER)
    ax.set_xlabel("Weight Power")
    ax.set_title("Per-Class F1 vs Weight Power")
    plt.colorbar(im, ax=ax, label="F1")

    for i in range(len(powers)):
        for j in range(8):
            ax.text(i, j, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(FIG_ROOT / "class_weight_heatmap.png")
    plt.close()
    print("  Saved class_weight_heatmap.png")


def fig8_all_methods_comparison():
    """Bar chart comparing all methods across all tasks."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (task, target_metric, target_val, title) in enumerate([
        ("2_class", "accuracy", 0.9968, "2-Class (Binary)"),
        ("8_class", "macro_f1", 0.7193, "8-Class"),
        ("34_class", "macro_f1", 0.70, "34-Class"),
    ]):
        ax = axes[idx]
        task_dir = RESULTS_ROOT / task
        if not task_dir.exists():
            continue

        methods = []
        values = []
        for sub in sorted(task_dir.iterdir()):
            if sub.name in ("BEST", "threshold_optimized", "ensemble_soft_vote",
                            "ensemble_stacking", "accuracy_push"):
                continue
            result_file = sub / "results.json"
            if result_file.exists():
                with open(result_file) as f:
                    r = json.load(f)
                m = r.get("test_metrics", {})
                val = m.get(target_metric, 0)
                methods.append(sub.name.replace("_", "\n", 1))
                values.append(val)

        if not methods:
            continue

        colors = ["#2a9d8f" if v >= target_val else "#e76f51" for v in values]
        bars = ax.bar(range(len(methods)), values, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(y=target_val, color="red", linestyle="--", linewidth=1.5, label=f"Paper ({target_val:.4f})")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(target_metric.replace("_", " ").title())
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.001, f"{val:.4f}",
                   ha="center", va="bottom", fontsize=6, rotation=90)

    plt.tight_layout()
    plt.savefig(FIG_ROOT / "all_methods_comparison.png")
    plt.close()
    print("  Saved all_methods_comparison.png")


def fig9_training_time():
    """Training time comparison across methods."""
    fig, ax = plt.subplots(figsize=(12, 5))
    all_methods = []
    all_times = []
    all_tasks = []

    for task in ["2_class", "8_class", "34_class"]:
        task_dir = RESULTS_ROOT / task
        if not task_dir.exists():
            continue
        for sub in sorted(task_dir.iterdir()):
            if sub.name in ("BEST", "threshold_optimized", "ensemble_soft_vote",
                            "ensemble_stacking", "accuracy_push"):
                continue
            result_file = sub / "results.json"
            if result_file.exists():
                with open(result_file) as f:
                    r = json.load(f)
                t = r.get("training_time_seconds", 0)
                all_methods.append(f"{task}\n{sub.name}")
                all_times.append(t)
                all_tasks.append(task)

    if all_methods:
        colors = {"2_class": "#2a9d8f", "8_class": "#f4a261", "34_class": "#e76f51"}
        bar_colors = [colors.get(t, "#264653") for t in all_tasks]
        ax.bar(range(len(all_methods)), [t/60 for t in all_times], color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(all_methods)))
        ax.set_xticklabels(all_methods, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Training Time (minutes)")
        ax.set_title("Training Time Comparison")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_ROOT / "training_times.png")
    plt.close()
    print("  Saved training_times.png")


def main():
    print("PHASE 4: GENERATING FIGURES")
    print("=" * 60)

    fig1_comparison_table()
    fig2_confusion_matrix_heatmap("8_class", 8, CLASS8_ORDER, "confusion_matrix_8class.png")
    fig2_confusion_matrix_heatmap("34_class", 34, CLASS34_ORDER, "confusion_matrix_34class.png")
    fig3_per_class_f1_bar()
    fig4_feature_importance()
    fig5_twin_analysis()
    fig6_protocol_ladder()
    fig7_class_weight_sweep()
    fig8_all_methods_comparison()
    fig9_training_time()

    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print(f"Figures saved to {FIG_ROOT}")
    print("Files:", sorted(os.listdir(FIG_ROOT)))


if __name__ == "__main__":
    main()
