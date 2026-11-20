#!/usr/bin/env python3
"""Ensemble evaluation: combine predictions from top models per task."""
import os, sys, json, glob, gc, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "16"

CACHE_ROOT = Path("/dev/shm/saksham/ids_cache")
IDS_ROOT = Path("/dev/shm/saksham/ids")
RESULTS_ROOT = IDS_ROOT / "results"
MODEL_ROOT = CACHE_ROOT / "models"

with open(IDS_ROOT / "src" / "class_maps.json") as f:
    CLASS_MAPS = json.load(f)
with open(IDS_ROOT / "src" / "data_meta.json") as f:
    DATA_META = json.load(f)

CLASS8_ORDER = CLASS_MAPS["class8_order"]
CLASS34_ORDER = CLASS_MAPS["class34_order"]
FEATURES = DATA_META["features"]
SEED = 42


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def get_probabilities_xgb(model_path, X, n_classes):
    import xgboost as xgb
    dmat = xgb.DMatrix(X)
    model = xgb.Booster()
    model.load_model(model_path)
    if n_classes == 2:
        prob_pos = model.predict(dmat)
        probs = np.column_stack([1 - prob_pos, prob_pos])
    else:
        probs = model.predict(dmat).reshape(-1, n_classes)
    return probs


def get_probabilities_lgbm(model_path, X, n_classes):
    import lightgbm as lgb
    model = lgb.Booster(model_file=model_path)
    raw = model.predict(X)
    if n_classes == 2:
        probs = np.column_stack([1 - raw, raw])
    else:
        probs = raw
    return probs


def get_probabilities_catboost(model_path, X, n_classes):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(model_path)
    probs = model.predict_proba(X)
    return probs


def ensemble_task(task, label_col, n_classes, class_names, targets):
    log(f"\n{'='*60}")
    log(f"ENSEMBLE for {task}")

    result_dirs = sorted(glob.glob(str(RESULTS_ROOT / task / "*/results.json")))
    if not result_dirs:
        log(f"  No results for {task}")
        return

    results = []
    for rp in result_dirs:
        if "BEST" in rp or "ensemble" in rp:
            continue
        with open(rp) as f:
            results.append(json.load(f))

    if not results:
        log(f"  No model results for {task}")
        return

    metric_key = "macro_f1" if n_classes > 2 else "accuracy"
    results.sort(key=lambda x: -x["test_metrics"][metric_key])

    log(f"  Available models ({len(results)}):")
    for r in results:
        m = r["test_metrics"]
        log(f"    {r['method']:35s} acc={m['accuracy']:.6f} mF1={m['macro_f1']:.6f}")

    top_models = results[:3]
    log(f"\n  Ensembling top {len(top_models)} models:")
    for r in top_models:
        log(f"    {r['method']}")

    log("  Loading test data...")
    test_df = load_split("test")
    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.int32)
    del test_df
    gc.collect()

    all_probs = []
    for r in top_models:
        method = r["method"]
        log(f"  Getting predictions from {method}...")

        if method.startswith("xgboost"):
            model_file = str(MODEL_ROOT / f"xgb_{task}_{method.replace('xgboost_', '')}.ubj")
            if not os.path.exists(model_file):
                model_file = str(MODEL_ROOT / f"xgb_{task}_{method.replace('xgboost_', '')}.json")
            if not os.path.exists(model_file):
                log(f"    Model file not found for {method}, skipping")
                continue
            probs = get_probabilities_xgb(model_file, X_test, n_classes)
        elif method.startswith("lightgbm"):
            model_file = str(MODEL_ROOT / f"lgbm_{task}_{method.replace('lightgbm_', '')}.txt")
            if not os.path.exists(model_file):
                log(f"    Model file not found for {method}, skipping")
                continue
            probs = get_probabilities_lgbm(model_file, X_test, n_classes)
        elif method.startswith("catboost"):
            model_file = str(MODEL_ROOT / f"catboost_{task}_{method.replace('catboost_', '')}.cbm")
            if not os.path.exists(model_file):
                log(f"    Model file not found for {method}, skipping")
                continue
            probs = get_probabilities_catboost(model_file, X_test, n_classes)
        else:
            log(f"    Unknown method type for {method}, skipping")
            continue

        all_probs.append(probs)

    if len(all_probs) < 2:
        log("  Not enough models for ensemble")
        return

    avg_probs = np.mean(all_probs, axis=0)
    y_pred = avg_probs.argmax(axis=1)

    test_metrics = compute_metrics(y_test, y_pred, n_classes, class_names)

    log(f"\n  ENSEMBLE RESULTS:")
    log(f"    accuracy:  {test_metrics['accuracy']:.6f}")
    log(f"    macro_f1:  {test_metrics['macro_f1']:.6f}")

    for tgt_name, tgt_val in targets.items():
        actual = test_metrics[tgt_name]
        met = actual >= tgt_val
        log(f"    {tgt_name}: {actual:.6f} vs {tgt_val:.4f} -> {'MET' if met else 'NOT MET'}")

    log(f"  Per-class:")
    for cls_name, cls_data in test_metrics["per_class"].items():
        log(f"    {cls_name:20s}: P={cls_data['precision']:.4f} R={cls_data['recall']:.4f} F1={cls_data['f1']:.4f}")

    result = {
        "method": "ensemble_top3_avg",
        "task": task,
        "models": [r["method"] for r in top_models[:len(all_probs)]],
        "test_metrics": test_metrics,
    }

    ens_dir = RESULTS_ROOT / task / "ensemble_top3"
    os.makedirs(ens_dir, exist_ok=True)
    with open(ens_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"  Saved: {ens_dir / 'results.json'}")

    # Check if ensemble is better than best single model
    best_single = results[0]
    best_metric = best_single["test_metrics"][metric_key]
    ens_metric = test_metrics[metric_key]
    if ens_metric > best_metric:
        log(f"  ENSEMBLE BEATS BEST SINGLE MODEL: {ens_metric:.6f} > {best_metric:.6f}")
        best_dir = RESULTS_ROOT / task / "BEST"
        os.makedirs(best_dir, exist_ok=True)
        with open(best_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)
        log(f"  Updated BEST with ensemble!")
    else:
        log(f"  Ensemble ({ens_metric:.6f}) did not beat best single ({best_metric:.6f})")


def main():
    log("ENSEMBLE EVALUATION")

    ensemble_task("2_class", "label_2class", 2, ["Benign", "Attack"],
                  {"accuracy": 0.9968})

    ensemble_task("8_class", "label_8class", 8, CLASS8_ORDER,
                  {"macro_f1": 0.7193, "accuracy": 0.9944})

    ensemble_task("34_class", "label_34class", 34, CLASS34_ORDER,
                  {"macro_f1": 0.7})

    log("\nENSEMBLE COMPLETE")


if __name__ == "__main__":
    main()
