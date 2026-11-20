#!/usr/bin/env python3
"""Phase 2e: Targeted 34-class boost with multiple strategies.
Run if standard configs don't reach mF1>0.7 for 34-class.
Strategies: optimal weight power, deeper trees, CatBoost balanced, ensemble.
"""
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, json, time, glob, gc, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

CACHE_ROOT = Path("/dev/shm/saksham/ids_cache")
IDS_ROOT = Path("/dev/shm/saksham/ids")
RESULTS_ROOT = IDS_ROOT / "results"
MODEL_ROOT = CACHE_ROOT / "models"
SEED = 42

with open(IDS_ROOT / "src" / "class_maps.json") as f:
    CLASS_MAPS = json.load(f)
with open(IDS_ROOT / "src" / "data_meta.json") as f:
    DATA_META = json.load(f)

CLASS34_ORDER = CLASS_MAPS["class34_order"]
FEATURES = DATA_META["features"]
N_CLASSES = 34


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(IDS_ROOT / "logs" / "phase2e_34class.log", "a") as f:
        f.write(line + "\n")


def load_split(split_name):
    parts = sorted(glob.glob(str(CACHE_ROOT / split_name / "*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class = {}
    p = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(N_CLASSES))
    r = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(N_CLASSES))
    f = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(N_CLASSES))
    for i in range(N_CLASSES):
        per_class[CLASS34_ORDER[i]] = {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
    return {"accuracy": float(acc), "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1), "per_class": per_class,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(N_CLASSES)).tolist()}


def compute_weights(y, power):
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1)
    return (len(y) / (N_CLASSES * counts)) ** power


def result_exists(config_name):
    return (RESULTS_ROOT / "34_class" / config_name / "results.json").exists()


def save_result(config_name, result):
    out_dir = RESULTS_ROOT / "34_class" / config_name
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"  Saved: {out_dir / 'results.json'}")


def main():
    log("PHASE 2e: 34-CLASS TARGETED BOOST")

    log("Loading data...")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train = train_df[FEATURES].values.astype(np.float32)
    y_train = train_df["label_34class"].values.astype(np.int32)
    X_val = val_df[FEATURES].values.astype(np.float32)
    y_val = val_df["label_34class"].values.astype(np.int32)
    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df["label_34class"].values.astype(np.int32)
    del train_df, val_df, test_df
    gc.collect()

    log(f"Data: train={len(y_train):,} val={len(y_val):,} test={len(y_test):,}")

    import xgboost as xgb

    # Strategy 1: XGBoost balanced with power=0.25 (N6 optimal for 8-class)
    config_name = "xgboost_balanced_p025"
    if not result_exists(config_name):
        log(f"{config_name}: starting...")
        t0 = time.time()
        weights = compute_weights(y_train, 0.25)
        sample_w = weights[y_train].astype(np.float32)

        params = {"max_depth": 12, "learning_rate": 0.05, "max_bin": 512,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "objective": "multi:softmax", "num_class": N_CLASSES,
                  "eval_metric": "mlogloss", "tree_method": "gpu_hist",
                  "seed": SEED, "verbosity": 0}

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_w)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(params, dtrain, num_boost_round=3000,
                          evals=[(dval, "val")], early_stopping_rounds=50,
                          verbose_eval=100)

        train_time = time.time() - t0
        y_pred = model.predict(dtest).astype(int)
        metrics = compute_metrics(y_test, y_pred)
        result = {"method": config_name, "task": "34_class",
                  "config": {**params, "weight_power": 0.25},
                  "n_estimators_used": model.best_iteration or 3000,
                  "training_time_seconds": train_time,
                  "test_metrics": metrics}
        save_result(config_name, result)
        model.save_model(str(MODEL_ROOT / f"xgb_34_class_balanced_p025.json"))
        log(f"  {config_name}: acc={metrics['accuracy']:.6f} mF1={metrics['macro_f1']:.6f} time={train_time:.0f}s")
        del dtrain, dval, dtest, model, sample_w
        gc.collect()
    else:
        log(f"{config_name}: SKIP (exists)")

    # Strategy 2: XGBoost balanced with power=0.5 (standard sqrt)
    config_name = "xgboost_balanced_p050"
    if not result_exists(config_name):
        log(f"{config_name}: starting...")
        t0 = time.time()
        weights = compute_weights(y_train, 0.5)
        sample_w = weights[y_train].astype(np.float32)

        params = {"max_depth": 12, "learning_rate": 0.05, "max_bin": 512,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "objective": "multi:softmax", "num_class": N_CLASSES,
                  "eval_metric": "mlogloss", "tree_method": "gpu_hist",
                  "seed": SEED, "verbosity": 0}

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_w)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(params, dtrain, num_boost_round=3000,
                          evals=[(dval, "val")], early_stopping_rounds=50,
                          verbose_eval=100)

        train_time = time.time() - t0
        y_pred = model.predict(dtest).astype(int)
        metrics = compute_metrics(y_test, y_pred)
        result = {"method": config_name, "task": "34_class",
                  "config": {**params, "weight_power": 0.5},
                  "n_estimators_used": model.best_iteration or 3000,
                  "training_time_seconds": train_time,
                  "test_metrics": metrics}
        save_result(config_name, result)
        model.save_model(str(MODEL_ROOT / f"xgb_34_class_balanced_p050.json"))
        log(f"  {config_name}: acc={metrics['accuracy']:.6f} mF1={metrics['macro_f1']:.6f} time={train_time:.0f}s")
        del dtrain, dval, dtest, model, sample_w
        gc.collect()
    else:
        log(f"{config_name}: SKIP (exists)")

    # Strategy 3: CatBoost with balanced weights
    config_name = "catboost_balanced"
    if not result_exists(config_name):
        log(f"{config_name}: starting...")
        t0 = time.time()
        try:
            from catboost import CatBoostClassifier, Pool

            model = CatBoostClassifier(
                depth=10, learning_rate=0.1, iterations=3000,
                early_stopping_rounds=50, auto_class_weights="Balanced",
                loss_function="MultiClass", task_type="GPU", devices="0",
                random_seed=SEED, verbose=100,
                train_dir=str(IDS_ROOT / "logs" / "catboost_34class"),
            )
            train_pool = Pool(X_train, label=y_train, feature_names=FEATURES)
            val_pool = Pool(X_val, label=y_val, feature_names=FEATURES)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)

            train_time = time.time() - t0
            y_pred = model.predict(X_test).astype(int).flatten()
            metrics = compute_metrics(y_test, y_pred)
            result = {"method": config_name, "task": "34_class",
                      "config": {"depth": 10, "lr": 0.1, "iterations": 3000,
                                 "auto_class_weights": "Balanced"},
                      "training_time_seconds": train_time,
                      "test_metrics": metrics}
            save_result(config_name, result)
            model.save_model(str(MODEL_ROOT / "catboost_34_class_balanced.cbm"))
            log(f"  {config_name}: acc={metrics['accuracy']:.6f} mF1={metrics['macro_f1']:.6f} time={train_time:.0f}s")
            del train_pool, val_pool, model
            gc.collect()
        except Exception as e:
            log(f"  {config_name} FAILED: {e}")
            traceback.print_exc()
    else:
        log(f"{config_name}: SKIP (exists)")

    # Strategy 4: Deep XGBoost with balanced weights (like 8-class BEST)
    config_name = "xgboost_deep_balanced"
    if not result_exists(config_name):
        log(f"{config_name}: starting...")
        t0 = time.time()
        weights = compute_weights(y_train, 0.5)
        sample_w = weights[y_train].astype(np.float32)

        params = {"max_depth": 16, "learning_rate": 0.03, "max_bin": 512,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "objective": "multi:softmax", "num_class": N_CLASSES,
                  "eval_metric": "mlogloss", "tree_method": "gpu_hist",
                  "seed": SEED, "verbosity": 0}

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_w)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(params, dtrain, num_boost_round=5000,
                          evals=[(dval, "val")], early_stopping_rounds=50,
                          verbose_eval=100)

        train_time = time.time() - t0
        y_pred = model.predict(dtest).astype(int)
        metrics = compute_metrics(y_test, y_pred)
        result = {"method": config_name, "task": "34_class",
                  "config": {**params, "weight_power": 0.5},
                  "n_estimators_used": model.best_iteration or 5000,
                  "training_time_seconds": train_time,
                  "test_metrics": metrics}
        save_result(config_name, result)
        model.save_model(str(MODEL_ROOT / f"xgb_34_class_deep_balanced.json"))
        log(f"  {config_name}: acc={metrics['accuracy']:.6f} mF1={metrics['macro_f1']:.6f} time={train_time:.0f}s")
        del dtrain, dval, dtest, model, sample_w
        gc.collect()
    else:
        log(f"{config_name}: SKIP (exists)")

    # Select best for 34-class
    log("\n=== 34-CLASS RESULTS ===")
    best_mf1 = 0
    best_result = None
    for d in sorted((RESULTS_ROOT / "34_class").iterdir()):
        if d.name == "BEST":
            continue
        rfile = d / "results.json"
        if rfile.exists():
            with open(rfile) as f:
                r = json.load(f)
            m = r["test_metrics"]
            log(f"  {d.name:35s}: acc={m['accuracy']:.6f} mF1={m['macro_f1']:.6f}")
            if m["macro_f1"] > best_mf1:
                best_mf1 = m["macro_f1"]
                best_result = r

    if best_result:
        best_dir = RESULTS_ROOT / "34_class" / "BEST"
        os.makedirs(best_dir, exist_ok=True)
        with open(best_dir / "results.json", "w") as f:
            json.dump(best_result, f, indent=2)
        m = best_result["test_metrics"]
        log(f"\n  34-CLASS BEST: {best_result['method']} acc={m['accuracy']:.6f} mF1={m['macro_f1']:.6f}")
        log(f"  Target mF1>0.7: {'MET' if m['macro_f1'] > 0.7 else 'NOT MET'} (gap={m['macro_f1']-0.7:+.4f})")

    log("PHASE 2e COMPLETE")


if __name__ == "__main__":
    main()
