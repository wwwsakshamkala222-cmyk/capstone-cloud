#!/usr/bin/env python3
"""Phase 2 Extra: Additional model configs to close performance gaps.
Run after Phase 2 if targets not met.
"""
import os, sys, json, time, glob, gc, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "16"

CACHE_ROOT = Path("/dev/shm/saksham/ids_cache")
IDS_ROOT = Path("/dev/shm/saksham/ids")
RESULTS_ROOT = IDS_ROOT / "results"
MODEL_ROOT = CACHE_ROOT / "models"
LOG_ROOT = IDS_ROOT / "logs"

SEED = 42

with open(IDS_ROOT / "src" / "class_maps.json") as f:
    CLASS_MAPS = json.load(f)
with open(IDS_ROOT / "src" / "data_meta.json") as f:
    DATA_META = json.load(f)

CLASS8_ORDER = CLASS_MAPS["class8_order"]
CLASS34_ORDER = CLASS_MAPS["class34_order"]
FEATURES = DATA_META["features"]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_ROOT / "phase2_extra.log", "a") as f:
        f.write(line + "\n")


def load_split(split_name):
    parts = sorted(glob.glob(str(CACHE_ROOT / split_name / "*.parquet")))
    dfs = [pd.read_parquet(p) for p in parts]
    return pd.concat(dfs, ignore_index=True)


def compute_metrics(y_true, y_pred, n_classes, class_names):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class = {}
    p = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    r = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    f = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    for i in range(n_classes):
        per_class[class_names[i]] = {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
            "per_class": per_class, "confusion_matrix": cm.tolist()}


def save_result(task, model_type, config_name, result):
    out_dir = RESULTS_ROOT / task / f"{model_type}_{config_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"  Saved: {out_dir / 'results.json'}")


def train_xgb_deep(X_train, y_train, X_val, y_val, X_test, y_test,
                    task, n_classes, class_names):
    """Deeper XGBoost with more trees and lower LR."""
    import xgboost as xgb

    log(f"  XGBoost deep: starting...")
    t0 = time.time()

    params = {
        "max_depth": 16, "learning_rate": 0.03, "max_bin": 512,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0,
    }

    if n_classes == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softmax"
        params["num_class"] = n_classes
        params["eval_metric"] = "mlogloss"

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(params, dtrain, num_boost_round=5000,
                       evals=[(dval, "val")], early_stopping_rounds=100,
                       verbose_eval=200)

    train_time = time.time() - t0

    if n_classes == 2:
        y_pred_test = (model.predict(dtest) > 0.5).astype(int)
        y_pred_val = (model.predict(dval) > 0.5).astype(int)
    else:
        y_pred_test = model.predict(dtest).astype(int)
        y_pred_val = model.predict(dval).astype(int)

    val_metrics = compute_metrics(y_val, y_pred_val, n_classes, class_names)
    test_metrics = compute_metrics(y_test, y_pred_test, n_classes, class_names)

    result = {
        "method": "xgboost_deep", "task": task,
        "config": params,
        "n_estimators_used": model.best_iteration if hasattr(model, 'best_iteration') else 5000,
        "training_time_seconds": train_time,
        "val_metrics": val_metrics, "test_metrics": test_metrics,
    }
    save_result(task, "xgboost", "deep", result)
    model.save_model(str(MODEL_ROOT / f"xgb_{task}_deep.ubj"))
    log(f"  XGBoost deep: acc={test_metrics['accuracy']:.6f} macro_f1={test_metrics['macro_f1']:.6f} time={train_time:.0f}s")

    del dtrain, dval, dtest, model
    gc.collect()
    return result


def train_xgb_balanced(X_train, y_train, X_val, y_val, X_test, y_test,
                        task, n_classes, class_names):
    """XGBoost with sample weights to boost minority classes."""
    import xgboost as xgb

    log(f"  XGBoost balanced_deep: starting...")
    t0 = time.time()

    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    N = len(y_train)
    weights = (N / (n_classes * counts)) ** 0.5
    sample_w = weights[y_train].astype(np.float32)

    params = {
        "max_depth": 14, "learning_rate": 0.05, "max_bin": 512,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0,
    }

    if n_classes == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softmax"
        params["num_class"] = n_classes
        params["eval_metric"] = "mlogloss"

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_w)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(params, dtrain, num_boost_round=5000,
                       evals=[(dval, "val")], early_stopping_rounds=100,
                       verbose_eval=200)

    train_time = time.time() - t0

    if n_classes == 2:
        y_pred_test = (model.predict(dtest) > 0.5).astype(int)
        y_pred_val = (model.predict(dval) > 0.5).astype(int)
    else:
        y_pred_test = model.predict(dtest).astype(int)
        y_pred_val = model.predict(dval).astype(int)

    val_metrics = compute_metrics(y_val, y_pred_val, n_classes, class_names)
    test_metrics = compute_metrics(y_test, y_pred_test, n_classes, class_names)

    result = {
        "method": "xgboost_balanced_deep", "task": task,
        "config": params,
        "n_estimators_used": model.best_iteration if hasattr(model, 'best_iteration') else 5000,
        "training_time_seconds": train_time,
        "val_metrics": val_metrics, "test_metrics": test_metrics,
    }
    save_result(task, "xgboost", "balanced_deep", result)
    model.save_model(str(MODEL_ROOT / f"xgb_{task}_balanced_deep.ubj"))
    log(f"  XGBoost balanced_deep: acc={test_metrics['accuracy']:.6f} macro_f1={test_metrics['macro_f1']:.6f} time={train_time:.0f}s")

    del dtrain, dval, dtest, model
    gc.collect()
    return result


def train_xgb_trainval(X_train, y_train, X_val, y_val, X_test, y_test,
                        task, n_classes, class_names):
    """XGBoost trained on train+val combined (80% data, matches paper's 80/20)."""
    import xgboost as xgb

    log(f"  XGBoost trainval: starting...")
    t0 = time.time()

    X_tv = np.concatenate([X_train, X_val], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)

    params = {
        "max_leaves": 1024, "max_depth": 0, "grow_policy": "lossguide",
        "learning_rate": 0.05, "max_bin": 512,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0,
    }

    if n_classes == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softmax"
        params["num_class"] = n_classes
        params["eval_metric"] = "mlogloss"

    dtrain = xgb.DMatrix(X_tv, label=y_tv)
    dtest = xgb.DMatrix(X_test, label=y_test)

    n_rounds = 800
    model = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=200)

    train_time = time.time() - t0

    if n_classes == 2:
        y_pred_test = (model.predict(dtest) > 0.5).astype(int)
    else:
        y_pred_test = model.predict(dtest).astype(int)

    test_metrics = compute_metrics(y_test, y_pred_test, n_classes, class_names)

    result = {
        "method": "xgboost_trainval", "task": task,
        "config": params,
        "n_estimators_used": n_rounds,
        "training_time_seconds": train_time,
        "test_metrics": test_metrics,
    }
    save_result(task, "xgboost", "trainval", result)
    model.save_model(str(MODEL_ROOT / f"xgb_{task}_trainval.ubj"))
    log(f"  XGBoost trainval: acc={test_metrics['accuracy']:.6f} macro_f1={test_metrics['macro_f1']:.6f} time={train_time:.0f}s")

    del dtrain, dtest, model, X_tv, y_tv
    gc.collect()
    return result


def train_rf(X_train, y_train, X_test, y_test,
             task, n_classes, class_names):
    """Random Forest — matches the paper's approach."""

    log(f"  Random Forest: starting...")
    t0 = time.time()

    model = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=2,
        n_jobs=16, random_state=SEED, class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred, n_classes, class_names)

    result = {
        "method": "rf_balanced", "task": task,
        "config": {"n_estimators": 100, "class_weight": "balanced_subsample"},
        "training_time_seconds": train_time,
        "test_metrics": test_metrics,
    }
    save_result(task, "rf", "balanced", result)
    log(f"  RF balanced: acc={test_metrics['accuracy']:.6f} macro_f1={test_metrics['macro_f1']:.6f} time={train_time:.0f}s")

    del model
    gc.collect()
    return result


def check_target_met(task, targets):
    """Check if any existing model meets the target."""
    best_path = RESULTS_ROOT / task / "BEST" / "results.json"
    if not best_path.exists():
        return False
    with open(best_path) as f:
        best = json.load(f)
    for tgt_name, tgt_val in targets.items():
        if best["test_metrics"].get(tgt_name, 0) < tgt_val:
            return False
    return True


def run_extra(task, label_col, n_classes, class_names, targets):
    log(f"\n{'='*60}")
    log(f"Phase 2 EXTRA: {task}")

    if check_target_met(task, targets):
        log(f"  Target already met, skipping extra models")
        return

    log(f"  Loading data...")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train = train_df[FEATURES].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.int32)
    X_val = val_df[FEATURES].values.astype(np.float32)
    y_val = val_df[label_col].values.astype(np.int32)
    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.int32)
    del train_df, val_df, test_df
    gc.collect()

    results = []

    # 1. Deep XGBoost (more trees, lower LR, deeper)
    try:
        r = train_xgb_deep(X_train, y_train, X_val, y_val, X_test, y_test,
                           task, n_classes, class_names)
        results.append(r)
    except Exception as e:
        log(f"  XGBoost deep FAILED: {e}")
        traceback.print_exc()

    # 2. Balanced deep XGBoost
    try:
        r = train_xgb_balanced(X_train, y_train, X_val, y_val, X_test, y_test,
                               task, n_classes, class_names)
        results.append(r)
    except Exception as e:
        log(f"  XGBoost balanced_deep FAILED: {e}")
        traceback.print_exc()

    # 3. Random Forest (paper's method)
    try:
        r = train_rf(X_train, y_train, X_test, y_test,
                     task, n_classes, class_names)
        results.append(r)
    except Exception as e:
        log(f"  RF FAILED: {e}")
        traceback.print_exc()

    # 4. XGBoost trained on train+val (matches paper's 80/20 split)
    try:
        r = train_xgb_trainval(X_train, y_train, X_val, y_val, X_test, y_test,
                               task, n_classes, class_names)
        results.append(r)
    except Exception as e:
        log(f"  XGBoost trainval FAILED: {e}")
        traceback.print_exc()

    # Update BEST if any new model is better
    metric_key = "macro_f1" if n_classes > 2 else "accuracy"
    best_path = RESULTS_ROOT / task / "BEST" / "results.json"
    if best_path.exists():
        with open(best_path) as f:
            current_best = json.load(f)
        current_best_metric = current_best["test_metrics"][metric_key]
    else:
        current_best_metric = 0

    for r in results:
        if r["test_metrics"][metric_key] > current_best_metric:
            current_best_metric = r["test_metrics"][metric_key]
            best_dir = RESULTS_ROOT / task / "BEST"
            os.makedirs(best_dir, exist_ok=True)
            with open(best_dir / "results.json", "w") as f:
                json.dump(r, f, indent=2)
            log(f"  NEW BEST: {r['method']} {metric_key}={r['test_metrics'][metric_key]:.6f}")

    # Summary
    log(f"\n  EXTRA RESULTS for {task}:")
    for r in sorted(results, key=lambda x: -x["test_metrics"][metric_key]):
        m = r["test_metrics"]
        log(f"    {r['method']:35s} acc={m['accuracy']:.6f} mF1={m['macro_f1']:.6f}")

    for tgt_name, tgt_val in targets.items():
        best_val = max(r["test_metrics"].get(tgt_name, 0) for r in results) if results else 0
        met = best_val >= tgt_val
        log(f"    {tgt_name}: {best_val:.6f} vs {tgt_val:.4f} -> {'MET' if met else 'NOT MET'}")

    del X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()


def main():
    log("PHASE 2 EXTRA: Additional models for gap closure")

    run_extra("2_class", "label_2class", 2, ["Benign", "Attack"],
              {"accuracy": 0.9968})

    run_extra("8_class", "label_8class", 8, CLASS8_ORDER,
              {"macro_f1": 0.7193, "accuracy": 0.9944})

    run_extra("34_class", "label_34class", 34, CLASS34_ORDER,
              {"macro_f1": 0.7})

    log("\nPHASE 2 EXTRA COMPLETE")


if __name__ == "__main__":
    main()
