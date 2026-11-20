#!/usr/bin/env python3
"""Run all post-training steps in sequence.
Must be run AFTER Phase 2 parallel training completes for all tasks.
Sequence: multi-seed eval → novelty experiments → Phase 2b boost → figures → summary
"""
import os, sys, subprocess, time
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IDS_SRC = "/dev/shm/saksham/ids/src"
IDS_ROOT = "/dev/shm/saksham/ids"
ENV = "/dev/shm/saksham/ids_env/bin/python3"
LOG = f"{IDS_ROOT}/logs/run_after_training.log"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run_script(name, script_path):
    log(f"Starting: {name}")
    t0 = time.time()
    result = subprocess.run(
        [ENV, "-u", script_path],
        cwd=IDS_ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "16", "CUDA_VISIBLE_DEVICES": "0"},
        capture_output=False
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        log(f"Completed: {name} ({elapsed:.0f}s)")
    else:
        log(f"FAILED: {name} (exit {result.returncode}, {elapsed:.0f}s)")
    return result.returncode == 0


def main():
    log("=" * 60)
    log("POST-TRAINING PIPELINE")
    log("=" * 60)

    # Verify all 3 tasks have BEST results
    from pathlib import Path
    RESULTS_ROOT = Path(IDS_ROOT) / "results"
    for task in ["2_class", "8_class", "34_class"]:
        best = RESULTS_ROOT / task / "BEST" / "results.json"
        if best.exists():
            import json
            with open(best) as f:
                r = json.load(f)
            m = r["test_metrics"]
            log(f"  {task}: {r['method']} acc={m['accuracy']:.6f} mF1={m['macro_f1']:.6f}")
        else:
            log(f"  {task}: NO BEST RESULT (will skip multi-seed)")

    # Step 1: Multi-seed evaluation
    log("\n--- STEP 1: Multi-Seed Evaluation ---")
    run_script("Multi-seed eval", f"{IDS_SRC}/multi_seed_eval.py")

    # Step 2: Novelty experiments — SKIP (N1-N6 all complete)
    log("\n--- STEP 2: Novelty Experiments --- SKIPPED (all complete)")
    for n in ["N1_twin_analysis", "N2_protocol_ladder", "N3_dos_vs_ddos",
              "N4_fingerprint_ablation", "N5_window_size", "N6_class_weight_sweep"]:
        p = Path(IDS_ROOT) / "novelty" / n / "results.json"
        log(f"  {n}: {'DONE' if p.exists() else 'MISSING'}")

    # Step 3: Phase 2b boost (ensemble/threshold)
    log("\n--- STEP 3: Ensemble & Threshold ---")
    run_script("Phase 2b boost", f"{IDS_SRC}/phase2b_boost.py")

    # Step 4: Figures
    log("\n--- STEP 4: Figures ---")
    run_script("Phase 4 figures", f"{IDS_SRC}/phase4_figures.py")

    # Step 5: Summary
    log("\n--- STEP 5: Summary ---")
    run_script("Generate summary", f"{IDS_SRC}/generate_summary.py")

    log("\n" + "=" * 60)
    log("POST-TRAINING PIPELINE COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
