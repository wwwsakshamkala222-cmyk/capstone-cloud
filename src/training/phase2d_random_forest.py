#!/usr/bin/env python3
"""Phase 2d: Random Forest (paper's method) for comparison.
Uses sklearn RF with 200 trees (paper's config). Runs on CPU.
"""
import os, sys, json, time, glob, gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CACHE_ROOT = Path("/dev/shm/saksham/ids_cache")
IDS_ROOT = Path("/dev/shm/saksham/ids")
RESULTS_ROOT = IDS_ROOT / "results"

SEED = 42

with open(IDS_ROOT / "src" / "class_maps.json") as f:
    CLASS_MAPS = json.load(f)
with open(IDS_ROOT / "src" / "data_meta.json") as f:
    DATA_META = json.load(f)

FEATURES = DATA_META["features"]
CLASS8_ORDER = CLASS_MAPS["class8_order"]
CLASS34_ORDER = CLASS_MAPS["class34_order"]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_split(split_name):
    parts = sorted(glob.glob(str(CACHE_ROOT / split_name / "*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def compute_metrics(y_true, y_pred, n_classes, class_names):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    r = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    f = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    per_class = {}
    for i in range(n_classes):
        name = class_names[i]
        per_class[name] = {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    return {"accuracy": float(acc), "macro_f1": float(macro_f1),
            "per_class": per_class, "confusion_matrix": cm.tolist()}


def run_rf(task, label_col, n_classes, class_names, n_jobs=32):
    log(f"\n{'='*60}")
    log(f"Random Forest — {task}")

    log("Loading data...")
    train_df = load_split("train")
    test_df = load_split("test")

    X_train = train_df[FEATURES].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.int32)
    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.int32)
    del train_df, test_df
    gc.collect()

    log(f"Data: train={len(y_train):,} test={len(y_test):,}")

    # Paper's config: default RF (no depth limit, 100 trees)
    # We use 200 trees for slightly better results
    log("Training RF (n_estimators=200, max_depth=None, n_jobs=32)...")
    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=n_jobs,
        random_state=SEED,
        verbose=1,
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    log(f"Training done in {elapsed:.0f}s")

    log("Predicting...")
    y_pred = rf.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, n_classes, class_names)

    log(f"RF: acc={metrics['accuracy']:.6f} mF1={metrics['macro_f1']:.6f}")
    for cls, data in metrics["per_class"].items():
        log(f"  {cls:20s}: P={data['precision']:.4f} R={data['recall']:.4f} F1={data['f1']:.4f}")

    # Feature importance
    importance = dict(zip(FEATURES, rf.feature_importances_.tolist()))
    top20 = dict(sorted(importance.items(), key=lambda x: -x[1])[:20])

    result = {
        "method": f"random_forest",
        "task": task,
        "config": {"n_estimators": 200, "max_depth": None, "n_jobs": n_jobs},
        "training_time_seconds": elapsed,
        "test_metrics": metrics,
        "feature_importance_top20": top20,
    }

    out_dir = RESULTS_ROOT / task / "random_forest"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"Saved: {out_dir / 'results.json'}")

    del rf, X_train, y_train, X_test, y_test
    gc.collect()
    return result


def main():
    log("PHASE 2d: RANDOM FOREST (paper's method)")

    # 2-class first (fastest)
    run_rf("2_class", "label_2class", 2, ["Benign", "Attack"])

    # 8-class
    run_rf("8_class", "label_8class", 8, CLASS8_ORDER)

    # 34-class
    run_rf("34_class", "label_34class", 34, CLASS34_ORDER)

    log("\nPHASE 2d COMPLETE")


if __name__ == "__main__":
    main()
