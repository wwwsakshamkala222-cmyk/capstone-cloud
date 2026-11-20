#!/usr/bin/env python3
"""Continue post-training pipeline from where it died.
Runs: multi-seed eval (8/34-class only), phase2b boost, phase4 figures, summary.
"""
import os, sys, subprocess, time
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IDS_SRC = "/dev/shm/saksham/ids/src"
IDS_ROOT = "/dev/shm/saksham/ids"
ENV = "/dev/shm/saksham/ids_env/bin/python3"
LOG = f"{IDS_ROOT}/logs/continue_pipeline.log"


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
    log("CONTINUATION PIPELINE (picking up from failed run)")
    log("=" * 60)

    # Step 1: Multi-seed eval for 8-class and 34-class only
    # (2-class already completed)
    from pathlib import Path
    ms_8 = Path(IDS_ROOT) / "results" / "8_class" / "BEST" / "multi_seed_results.json"
    ms_34 = Path(IDS_ROOT) / "results" / "34_class" / "BEST" / "multi_seed_results.json"

    if not ms_8.exists() or not ms_34.exists():
        log("\n--- STEP 1: Multi-Seed Evaluation (8/34-class) ---")
        run_script("Multi-seed eval (8+34 class)",
                   f"{IDS_SRC}/multi_seed_eval_remaining.py")
    else:
        log("\n--- STEP 1: Multi-Seed Evaluation --- SKIPPED (already done)")

    # Step 2: Phase 2b boost (ensemble/threshold)
    log("\n--- STEP 2: Ensemble & Threshold ---")
    run_script("Phase 2b boost", f"{IDS_SRC}/phase2b_boost.py")

    # Step 3: Figures
    log("\n--- STEP 3: Figures ---")
    run_script("Phase 4 figures", f"{IDS_SRC}/phase4_figures.py")

    # Step 4: Summary
    log("\n--- STEP 4: Summary ---")
    run_script("Generate summary", f"{IDS_SRC}/generate_summary.py")

    log("\n" + "=" * 60)
    log("CONTINUATION PIPELINE COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
