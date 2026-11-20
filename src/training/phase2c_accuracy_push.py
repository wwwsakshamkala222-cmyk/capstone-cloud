#!/usr/bin/env python3
"""Phase 2c: Aggressive accuracy push for 2-class task.
Only runs if Phase 2 doesn't hit 0.9968 accuracy.
Strategies:
1. XGBoost deeper with lower LR
2. RandomForest (paper's method) with sklearn
3. Threshold-optimized ensemble
"""
import os, sys, json, time, glob, gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CACHE_ROOT = Path("/dev/shm/saksham/ids_cache")
IDS_ROOT = Path("/dev/shm/saksham/ids")
RESULTS_ROOT = IDS_ROOT / "results"
MODEL_ROOT = CACHE_ROOT / "models"

SEED = 42

with open(IDS_ROOT / "src" / "class_maps.json") as f:
    CLASS_MAPS = json.load(f)
with open(IDS_ROOT / "src" / "data_meta.json") as f:
    DATA_META = json.load(f)

FEATURES = DATA_META["features"]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_split(split_name):
    parts = sorted(glob.glob(str(CACHE_ROOT / split_name / "*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p = precision_score(y_true, y_pred, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    per_class = {}
    for i, name in enumerate(["Benign", "Attack"]):
        per_class[name] = {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
    return {"accuracy": float(acc), "macro_f1": float(macro_f1),
            "per_class": per_class, "confusion_matrix": cm.tolist()}


def optimize_threshold(y_val, probs_val, y_test, probs_test):
    """Find threshold that maximizes accuracy on val, evaluate on test."""
    thresholds = np.linspace(0.01, 0.99, 199)
    best_t, best_acc = 0.5, 0
    for t in thresholds:
        pred = (probs_val >= t).astype(int)
        acc = accuracy_score(y_val, pred)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)

    y_pred = (probs_test >= best_t).astype(int)
    metrics = compute_metrics(y_test, y_pred)
    return best_t, best_acc, metrics


def main():
    log("PHASE 2c: ACCURACY PUSH for 2-class")

    # Check if target already met
    best_path = RESULTS_ROOT / "2_class" / "BEST" / "results.json"
    if best_path.exists():
        with open(best_path) as f:
            best = json.load(f)
        if best["test_metrics"]["accuracy"] >= 0.9968:
            log(f"Target already met: {best['test_metrics']['accuracy']:.6f} >= 0.9968")
            return

    log("Loading data...")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train = train_df[FEATURES].values.astype(np.float32)
    y_train = train_df["label_2class"].values.astype(np.int32)
    X_val = val_df[FEATURES].values.astype(np.float32)
    y_val = val_df["label_2class"].values.astype(np.int32)
    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df["label_2class"].values.astype(np.int32)
    del train_df, val_df, test_df
    gc.collect()

    log(f"Data: train={len(y_train):,} val={len(y_val):,} test={len(y_test):,}")

    all_results = []

    # ── Strategy 1: Deep XGBoost with low LR ──
    log("\n=== Strategy 1: Deep XGBoost (depth=16, lr=0.02, 5000 rounds) ===")
    try:
        import xgboost as xgb

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {"max_depth": 16, "learning_rate": 0.02, "max_bin": 512,
                  "objective": "binary:logistic", "eval_metric": "logloss",
                  "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0,
                  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5}

        t0 = time.time()
        model = xgb.train(params, dtrain, num_boost_round=5000,
                           evals=[(dval, "val")], early_stopping_rounds=100,
                           verbose_eval=200)
        elapsed = time.time() - t0

        # Standard threshold
        y_prob_test = model.predict(dtest)
        y_pred = (y_prob_test > 0.5).astype(int)
        metrics = compute_metrics(y_test, y_pred)
        log(f"  Standard: acc={metrics['accuracy']:.6f} mF1={metrics['macro_f1']:.6f} ({elapsed:.0f}s)")

        # Optimized threshold
        y_prob_val = model.predict(dval)
        best_t, val_acc, metrics_opt = optimize_threshold(y_val, y_prob_val, y_test, y_prob_test)
        log(f"  Threshold-optimized (t={best_t:.3f}): acc={metrics_opt['accuracy']:.6f} mF1={metrics_opt['macro_f1']:.6f}")

        result = {
            "method": "xgboost_deep_low_lr",
            "config": {"max_depth": 16, "learning_rate": 0.02, "max_bin": 512},
            "training_time_seconds": elapsed,
            "test_metrics": metrics,
            "test_metrics_threshold_opt": metrics_opt,
            "threshold": best_t,
            "n_estimators_used": getattr(model, 'best_iteration', 5000) + 1,
        }
        all_results.append(result)

        model.save_model(str(MODEL_ROOT / "xgb_2_class_deep.json"))

        del dtrain, dval, dtest, model
        gc.collect()
    except Exception as e:
        log(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

    # ── Strategy 2: XGBoost lossguide even wider ──
    log("\n=== Strategy 2: XGBoost lossguide (2048 leaves, lr=0.02) ===")
    try:
        import xgboost as xgb

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {"max_leaves": 2048, "max_depth": 0, "grow_policy": "lossguide",
                  "learning_rate": 0.02, "max_bin": 512,
                  "objective": "binary:logistic", "eval_metric": "logloss",
                  "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0,
                  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5}

        t0 = time.time()
        model = xgb.train(params, dtrain, num_boost_round=5000,
                           evals=[(dval, "val")], early_stopping_rounds=100,
                           verbose_eval=200)
        elapsed = time.time() - t0

        y_prob_test = model.predict(dtest)
        y_pred = (y_prob_test > 0.5).astype(int)
        metrics = compute_metrics(y_test, y_pred)
        log(f"  Standard: acc={metrics['accuracy']:.6f} mF1={metrics['macro_f1']:.6f} ({elapsed:.0f}s)")

        y_prob_val = model.predict(dval)
        best_t, val_acc, metrics_opt = optimize_threshold(y_val, y_prob_val, y_test, y_prob_test)
        log(f"  Threshold-optimized (t={best_t:.3f}): acc={metrics_opt['accuracy']:.6f}")

        result = {
            "method": "xgboost_lossguide_wide",
            "config": {"max_leaves": 2048, "learning_rate": 0.02, "max_bin": 512},
            "training_time_seconds": elapsed,
            "test_metrics": metrics,
            "test_metrics_threshold_opt": metrics_opt,
            "threshold": best_t,
            "n_estimators_used": getattr(model, 'best_iteration', 5000) + 1,
        }
        all_results.append(result)
        model.save_model(str(MODEL_ROOT / "xgb_2_class_wide.json"))

        del dtrain, dval, dtest, model
        gc.collect()
    except Exception as e:
        log(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

    # ── Strategy 3: Combined ensemble of ALL models with threshold ──
    log("\n=== Strategy 3: Ensemble of all XGBoost + threshold ===")
    try:
        import xgboost as xgb

        model_files = sorted(glob.glob(str(MODEL_ROOT / "xgb_2_class_*.json")))
        log(f"  Found {len(model_files)} XGBoost models")

        all_val_probs = []
        all_test_probs = []

        for mf in model_files:
            m = xgb.Booster()
            m.load_model(mf)
            dval = xgb.DMatrix(X_val)
            dtest = xgb.DMatrix(X_test)
            all_val_probs.append(m.predict(dval))
            all_test_probs.append(m.predict(dtest))
            del m
            gc.collect()

        if len(all_val_probs) >= 2:
            avg_val = np.mean(all_val_probs, axis=0)
            avg_test = np.mean(all_test_probs, axis=0)

            y_pred_default = (avg_test > 0.5).astype(int)
            metrics_default = compute_metrics(y_test, y_pred_default)
            log(f"  Ensemble default: acc={metrics_default['accuracy']:.6f}")

            best_t, val_acc, metrics_opt = optimize_threshold(y_val, avg_val, y_test, avg_test)
            log(f"  Ensemble optimized (t={best_t:.3f}): acc={metrics_opt['accuracy']:.6f}")

            result = {
                "method": "xgb_ensemble_threshold",
                "models": [os.path.basename(f) for f in model_files],
                "test_metrics": metrics_default,
                "test_metrics_threshold_opt": metrics_opt,
                "threshold": best_t,
            }
            all_results.append(result)

    except Exception as e:
        log(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

    # ── Save all results ──
    out_dir = RESULTS_ROOT / "2_class" / "accuracy_push"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Check if any strategy met the target
    for r in all_results:
        for key in ("test_metrics", "test_metrics_threshold_opt"):
            if key in r and r[key]["accuracy"] >= 0.9968:
                log(f"\n>>> TARGET MET by {r['method']} ({key}): {r[key]['accuracy']:.6f}")

    log("\nPHASE 2c COMPLETE")


if __name__ == "__main__":
    main()
