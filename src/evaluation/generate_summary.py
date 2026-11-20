#!/usr/bin/env python3
"""Generate final SUMMARY.txt with all results."""
import os
os.environ["OMP_NUM_THREADS"] = "16"
import json, glob
from pathlib import Path

IDS_ROOT = Path("/dev/shm/saksham/ids")
RESULTS_ROOT = IDS_ROOT / "results"
NOVELTY_ROOT = IDS_ROOT / "novelty"

TARGETS = {
    "2_class": {"accuracy": 0.9968, "macro_f1": 0.9653},
    "8_class": {"macro_f1": 0.7193, "accuracy": 0.9944},
    "34_class": {"macro_f1": 0.7},
}


def load_best(task):
    path = RESULTS_ROOT / task / "BEST" / "results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_novelty(name):
    path = NOVELTY_ROOT / name / "results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("FINAL SUMMARY — CICIoT2023 IDS Benchmark")
    lines.append("=" * 70)

    # Task results — use multi-seed mean if available, else single-run
    for task in ["2_class", "8_class", "34_class"]:
        best = load_best(task)
        targets = TARGETS[task]
        lines.append(f"\n{task.upper()}:")
        if not best:
            lines.append("  [No results available]")
            continue

        ms_path = RESULTS_ROOT / task / "BEST" / "multi_seed_results.json"
        ms = None
        if ms_path.exists():
            with open(ms_path) as f:
                ms = json.load(f)

        lines.append(f"  Best model: {best['method']}")
        for tgt_name, tgt_val in targets.items():
            if ms:
                mean_key = f"mean_{tgt_name}"
                std_key = f"std_{tgt_name}"
                actual = ms.get(mean_key, 0)
                std = ms.get(std_key, 0)
                met = actual >= tgt_val
                lines.append(f"  {tgt_name}: {actual:.6f} +/- {std:.6f} vs paper {tgt_val:.4f} -> {'MET' if met else 'NOT MET'} (3-seed mean)")
            else:
                actual = best["test_metrics"].get(tgt_name, 0)
                met = actual >= tgt_val
                lines.append(f"  {tgt_name}: {actual:.6f} vs paper {tgt_val:.4f} -> {'MET' if met else 'NOT MET'} (single seed)")
        lines.append(f"  Training time: {best.get('training_time_seconds', 0):.0f}s")

    # Novelty findings
    lines.append(f"\n{'='*70}")
    lines.append("NOVELTY FINDINGS:")

    n1 = load_novelty("N1_twin_analysis")
    if n1:
        lines.append(f"  N1 Twin Analysis: {n1['overall_twin_pct']:.1f}% test rows are exact twins of train rows")
        if n1.get("twin_inflation_macro_f1"):
            lines.append(f"    Twin inflation on macro-F1: {n1['twin_inflation_macro_f1']:.4f}")

    n2 = load_novelty("N2_protocol_ladder")
    if n2:
        a = n2.get("protocol_a_paper", {})
        d = n2.get("protocol_d_file_grouped_twin_free", {})
        if a and d:
            drop = a.get("macro_f1", 0) - d.get("macro_f1", 0)
            lines.append(f"  N2 Protocol Ladder: macro-F1 drops {drop:.4f} from paper to strictest protocol")

    n3 = load_novelty("N3_dos_vs_ddos")
    if n3:
        lines.append(f"  N3 DoS vs DDoS: paper acc={n3['paper_protocol']['accuracy']:.4f}, file-grouped acc={n3['file_grouped']['accuracy']:.4f}")
        lines.append(f"    Memorization signal: {n3['memorization_signal']}")

    n4 = load_novelty("N4_fingerprint_ablation")
    if n4:
        lines.append(f"  N4 Fingerprint: removing TTL+IAT drops acc by {n4['without_ttl_iat']['accuracy_drop']:.6f}")
        lines.append(f"    Removing top-10 features drops acc by {n4['without_top10']['accuracy_drop']:.6f}")

    n5 = load_novelty("N5_window_size")
    if n5:
        lines.append(f"  N5 Window-Size: classifier acc={n5['window_classifier_accuracy']:.4f} (leak={'YES' if n5['window_leaks_class'] else 'NO'})")
        lines.append(f"    8-class macro-F1 change without window features: {n5['macro_f1_change']:.4f}")

    n6 = load_novelty("N6_class_weight_sweep")
    if n6:
        best_p = max(n6["sweep"], key=lambda x: x["macro_f1"])
        lines.append(f"  N6 Class-Weight: best macro-F1 at p={best_p['power']} ({best_p['macro_f1']:.4f})")

    # Multi-seed results
    lines.append(f"\n{'='*70}")
    lines.append("MULTI-SEED EVALUATION (3 seeds: 42, 123, 456):")
    for task in ["2_class", "8_class", "34_class"]:
        ms_path = RESULTS_ROOT / task / "BEST" / "multi_seed_results.json"
        if ms_path.exists():
            with open(ms_path) as f:
                ms = json.load(f)
            lines.append(f"  {task}: {ms['method']}")
            lines.append(f"    accuracy:  {ms['mean_accuracy']:.6f} +/- {ms['std_accuracy']:.6f}")
            lines.append(f"    macro-F1:  {ms['mean_macro_f1']:.6f} +/- {ms['std_macro_f1']:.6f}")
        else:
            lines.append(f"  {task}: [not available]")

    # Accuracy push results (2-class)
    push_path = RESULTS_ROOT / "2_class" / "accuracy_push" / "results.json"
    if push_path.exists():
        with open(push_path) as f:
            push_results = json.load(f)
        if push_results:
            lines.append(f"\n{'='*70}")
            lines.append("ACCURACY PUSH (2-class additional strategies):")
            for r in push_results:
                m = r.get("test_metrics", {})
                lines.append(f"  {r['method']}: acc={m.get('accuracy', 0):.6f} mF1={m.get('macro_f1', 0):.6f}")
                if "test_metrics_threshold_opt" in r:
                    mo = r["test_metrics_threshold_opt"]
                    lines.append(f"    + threshold({r.get('threshold', 0.5):.3f}): acc={mo.get('accuracy', 0):.6f}")

    # File locations
    lines.append(f"\n{'='*70}")
    lines.append("FILE LOCATIONS:")
    lines.append(f"  Results:  {RESULTS_ROOT}/")
    lines.append(f"  Novelty:  {NOVELTY_ROOT}/")
    lines.append(f"  Figures:  {IDS_ROOT / 'figures'}/")
    lines.append(f"  Scripts:  {IDS_ROOT / 'src'}/")
    lines.append(f"  Logs:     {IDS_ROOT / 'logs'}/")
    lines.append(f"  Models:   /dev/shm/saksham/ids_cache/models/")
    lines.append(f"  Decisions: {IDS_ROOT / 'DECISIONS.md'}")

    # List all result files
    lines.append(f"\nRESULT FILES:")
    for p in sorted(glob.glob(str(RESULTS_ROOT / "**" / "results.json"), recursive=True)):
        lines.append(f"  {p}")
    for p in sorted(glob.glob(str(NOVELTY_ROOT / "**" / "results.json"), recursive=True)):
        lines.append(f"  {p}")

    text = "\n".join(lines)

    with open(IDS_ROOT / "SUMMARY.txt", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
