#!/usr/bin/env python3
"""Train remaining 2-class models (LightGBM, CatBoost, MLP) that Phase 2 main couldn't finish."""
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
LOG_FILE = IDS_ROOT / "logs" / "phase2_2class_remaining.log"
SEED = 42
N_THREADS = 16

with open(IDS_ROOT / "src" / "class_maps.json") as f:
    CLASS_MAPS = json.load(f)
with open(IDS_ROOT / "src" / "data_meta.json") as f:
    DATA_META = json.load(f)

FEATURES = DATA_META["features"]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_split(split_name):
    parts = sorted(glob.glob(str(CACHE_ROOT / split_name / "*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def compute_metrics(y_true, y_pred, n_classes, class_names=None):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class = {}
    p = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    r = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    f = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    for i in range(n_classes):
        name = class_names[i] if class_names else str(i)
        per_class[name] = {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    return {"accuracy": float(acc), "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1), "per_class": per_class,
            "confusion_matrix": cm.tolist()}


def result_exists(method, config_name):
    return (RESULTS_ROOT / "2_class" / f"{method}_{config_name}" / "results.json").exists()


def save_result(method, config_name, result):
    out_dir = RESULTS_ROOT / "2_class" / f"{method}_{config_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"  Saved: {out_dir / 'results.json'}")


def main():
    log("2-CLASS REMAINING MODELS (LightGBM only — CPU)")
    log(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

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

    import lightgbm as lgb

    # LightGBM cfg1
    if not result_exists("lightgbm", "cfg1"):
        log("LightGBM cfg1: starting...")
        t0 = time.time()
        params = {"num_leaves": 255, "learning_rate": 0.1, "min_child_samples": 100,
                  "objective": "binary", "metric": "binary_logloss",
                  "device": "cpu", "seed": SEED, "verbose": -1,
                  "num_threads": N_THREADS, "force_col_wise": True,
                  "subsample": 0.8, "colsample_bytree": 0.8}
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURES, reference=dtrain, free_raw_data=False)
        model = lgb.train(params, dtrain, num_boost_round=2000,
                          valid_sets=[dval], valid_names=["val"],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        train_time = time.time() - t0
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        y_pred_val = (model.predict(X_val) > 0.5).astype(int)
        test_m = compute_metrics(y_test, y_pred, 2, ["Benign", "Attack"])
        val_m = compute_metrics(y_val, y_pred_val, 2, ["Benign", "Attack"])
        importance = dict(zip(FEATURES, model.feature_importance(importance_type="gain").tolist()))
        result = {"method": "lightgbm_cfg1", "task": "2_class", "config": params,
                  "n_estimators_used": model.best_iteration if model.best_iteration > 0 else 2000,
                  "training_time_seconds": train_time,
                  "val_metrics": val_m, "test_metrics": test_m,
                  "feature_importance_top20": dict(sorted(importance.items(), key=lambda x: -x[1])[:20])}
        save_result("lightgbm", "cfg1", result)
        model.save_model(str(MODEL_ROOT / "lgbm_2_class_cfg1.txt"))
        log(f"LightGBM cfg1: acc={test_m['accuracy']:.6f} mF1={test_m['macro_f1']:.6f} time={train_time:.0f}s")
        del dtrain, dval, model
        gc.collect()
    else:
        log("LightGBM cfg1: SKIP (exists)")

    # LightGBM cfg2
    if not result_exists("lightgbm", "cfg2"):
        log("LightGBM cfg2: starting...")
        t0 = time.time()
        params = {"num_leaves": 512, "learning_rate": 0.05, "min_child_samples": 50,
                  "objective": "binary", "metric": "binary_logloss",
                  "device": "cpu", "seed": SEED, "verbose": -1,
                  "num_threads": N_THREADS, "force_col_wise": True,
                  "subsample": 0.8, "colsample_bytree": 0.8}
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURES, reference=dtrain, free_raw_data=False)
        model = lgb.train(params, dtrain, num_boost_round=3000,
                          valid_sets=[dval], valid_names=["val"],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        train_time = time.time() - t0
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        y_pred_val = (model.predict(X_val) > 0.5).astype(int)
        test_m = compute_metrics(y_test, y_pred, 2, ["Benign", "Attack"])
        val_m = compute_metrics(y_val, y_pred_val, 2, ["Benign", "Attack"])
        importance = dict(zip(FEATURES, model.feature_importance(importance_type="gain").tolist()))
        result = {"method": "lightgbm_cfg2", "task": "2_class", "config": params,
                  "n_estimators_used": model.best_iteration if model.best_iteration > 0 else 3000,
                  "training_time_seconds": train_time,
                  "val_metrics": val_m, "test_metrics": test_m,
                  "feature_importance_top20": dict(sorted(importance.items(), key=lambda x: -x[1])[:20])}
        save_result("lightgbm", "cfg2", result)
        model.save_model(str(MODEL_ROOT / "lgbm_2_class_cfg2.txt"))
        log(f"LightGBM cfg2: acc={test_m['accuracy']:.6f} mF1={test_m['macro_f1']:.6f} time={train_time:.0f}s")
        del dtrain, dval, model
        gc.collect()
    else:
        log("LightGBM cfg2: SKIP (exists)")

    log("2-CLASS LIGHTGBM COMPLETE")
    log("Note: CatBoost and MLP deferred until GPU is free from 8/34-class training")


if __name__ == "__main__":
    main()
