#!/usr/bin/env python3
"""Phase 2: Train all models for 2/8/34-class tasks.
XGBoost, LightGBM, CatBoost on GPU. Full data, no caps.
"""
import os, sys, json, time, glob, gc, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CACHE_ROOT = Path("/dev/shm/saksham/ids_cache")
IDS_ROOT = Path("/dev/shm/saksham/ids")
RESULTS_ROOT = IDS_ROOT / "results"
MODEL_ROOT = CACHE_ROOT / "models"
LOG_ROOT = IDS_ROOT / "logs"

SEED = 42
SEEDS = [42, 123, 456]  # 3 seeds for final models

# Load class maps
with open(IDS_ROOT / "src" / "class_maps.json") as f:
    CLASS_MAPS = json.load(f)

CLASS8_ORDER = CLASS_MAPS["class8_order"]
CLASS34_ORDER = CLASS_MAPS["class34_order"]


def load_split(split_name, label_col):
    """Load all parquet parts for a split, return X (float32), y (int)."""
    parts = sorted(glob.glob(str(CACHE_ROOT / split_name / "*.parquet")))
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    feature_cols = [c for c in df.columns if not c.startswith("label_")]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int32)
    del df, dfs
    gc.collect()
    return X, y


def compute_metrics(y_true, y_pred, n_classes, class_names=None):
    """Compute all metrics."""
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

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def compute_class_weights(y, n_classes, power=0.5):
    """w_c = (N / (k * n_c))^power"""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    N = len(y)
    k = n_classes
    weights = (N / (k * counts)) ** power
    return weights


def save_results(task, method, config_name, result, model_obj=None):
    """Save results to JSON."""
    out_dir = RESULTS_ROOT / task / f"{method}_{config_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {out_dir / 'results.json'}")
    return out_dir


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_ROOT / "phase2.log", "a") as f:
        f.write(line + "\n")


# ─────────────────────── XGBoost ───────────────────────

def train_xgb(X_train, y_train, X_val, y_val, X_test, y_test,
              task, n_classes, class_names, config_name, params, sample_weights=None):
    import xgboost as xgb

    log(f"  XGBoost {config_name}: starting...")
    t0 = time.time()

    if n_classes == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softmax"
        params["num_class"] = n_classes
        params["eval_metric"] = "mlogloss"

    params["tree_method"] = "hist"
    params["device"] = "cuda"
    params["seed"] = SEED
    params["verbosity"] = 1

    n_estimators = params.pop("n_estimators", 2000)
    early_stop = params.pop("early_stopping_rounds", 50)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        params, dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stop,
        verbose_eval=100,
    )

    train_time = time.time() - t0

    # Predict
    if n_classes == 2:
        y_pred_test = (model.predict(dtest) > 0.5).astype(int)
        y_pred_val = (model.predict(dval) > 0.5).astype(int)
    else:
        y_pred_test = model.predict(dtest).astype(int)
        y_pred_val = model.predict(dval).astype(int)

    val_metrics = compute_metrics(y_val, y_pred_val, n_classes, class_names)
    test_metrics = compute_metrics(y_test, y_pred_test, n_classes, class_names)

    # Feature importance
    importance = model.get_score(importance_type="gain")
    feat_names = [f"f{i}" for i in range(X_train.shape[1])]
    with open(IDS_ROOT / "src" / "data_meta.json") as f:
        meta = json.load(f)
    real_names = meta["features"]
    importance_named = {real_names[int(k[1:])]: v for k, v in importance.items() if k.startswith("f")}

    result = {
        "method": f"xgboost_{config_name}",
        "task": task,
        "config": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                   for k, v in params.items()},
        "n_estimators_used": model.best_iteration + 1 if hasattr(model, 'best_iteration') else n_estimators,
        "training_time_seconds": train_time,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_importance_top20": dict(sorted(importance_named.items(), key=lambda x: -x[1])[:20]),
    }

    save_results(task, "xgboost", config_name, result)

    # Save model
    model_path = MODEL_ROOT / f"xgb_{task}_{config_name}.json"
    model.save_model(str(model_path))

    log(f"  XGBoost {config_name}: acc={test_metrics['accuracy']:.6f} macro_f1={test_metrics['macro_f1']:.6f} time={train_time:.0f}s")

    del dtrain, dval, dtest, model
    gc.collect()

    return result


# ─────────────────────── LightGBM ───────────────────────

def train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test,
               task, n_classes, class_names, config_name, params, sample_weights=None):
    import lightgbm as lgb

    log(f"  LightGBM {config_name}: starting...")
    t0 = time.time()

    if n_classes == 2:
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"
    else:
        params["objective"] = "multiclass"
        params["num_class"] = n_classes
        params["metric"] = "multi_logloss"

    params["device"] = "gpu"
    params["seed"] = SEED
    params["verbose"] = -1
    params["force_col_wise"] = True

    n_estimators = params.pop("n_estimators", 2000)
    early_stop = params.pop("early_stopping_rounds", 50)

    with open(IDS_ROOT / "src" / "data_meta.json") as f:
        meta = json.load(f)
    feature_names = meta["features"]

    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights,
                         feature_name=feature_names, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names,
                       reference=dtrain, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(early_stop),
        lgb.log_evaluation(100),
    ]

    model = lgb.train(
        params, dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    train_time = time.time() - t0

    # Predict
    if n_classes == 2:
        y_pred_test = (model.predict(X_test) > 0.5).astype(int)
        y_pred_val = (model.predict(X_val) > 0.5).astype(int)
    else:
        y_pred_test = model.predict(X_test).argmax(axis=1)
        y_pred_val = model.predict(X_val).argmax(axis=1)

    val_metrics = compute_metrics(y_val, y_pred_val, n_classes, class_names)
    test_metrics = compute_metrics(y_test, y_pred_test, n_classes, class_names)

    importance = dict(zip(feature_names, model.feature_importance(importance_type="gain").tolist()))

    result = {
        "method": f"lightgbm_{config_name}",
        "task": task,
        "config": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                   for k, v in params.items()},
        "n_estimators_used": model.best_iteration if model.best_iteration > 0 else n_estimators,
        "training_time_seconds": train_time,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_importance_top20": dict(sorted(importance.items(), key=lambda x: -x[1])[:20]),
    }

    save_results(task, "lightgbm", config_name, result)

    model_path = MODEL_ROOT / f"lgbm_{task}_{config_name}.txt"
    model.save_model(str(model_path))

    log(f"  LightGBM {config_name}: acc={test_metrics['accuracy']:.6f} macro_f1={test_metrics['macro_f1']:.6f} time={train_time:.0f}s")

    del dtrain, dval, model
    gc.collect()

    return result


# ─────────────────────── CatBoost ───────────────────────

def train_catboost(X_train, y_train, X_val, y_val, X_test, y_test,
                   task, n_classes, class_names, config_name, params):
    from catboost import CatBoostClassifier, Pool

    log(f"  CatBoost {config_name}: starting...")
    t0 = time.time()

    with open(IDS_ROOT / "src" / "data_meta.json") as f:
        meta = json.load(f)
    feature_names = meta["features"]

    model = CatBoostClassifier(
        task_type="GPU",
        devices="0",
        random_seed=SEED,
        verbose=100,
        **params,
    )

    train_pool = Pool(X_train, label=y_train, feature_names=feature_names)
    val_pool = Pool(X_val, label=y_val, feature_names=feature_names)

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    train_time = time.time() - t0

    y_pred_test = model.predict(X_test).astype(int).flatten()
    y_pred_val = model.predict(X_val).astype(int).flatten()

    val_metrics = compute_metrics(y_val, y_pred_val, n_classes, class_names)
    test_metrics = compute_metrics(y_test, y_pred_test, n_classes, class_names)

    importance = dict(zip(feature_names, model.get_feature_importance().tolist()))

    result = {
        "method": f"catboost_{config_name}",
        "task": task,
        "config": params,
        "training_time_seconds": train_time,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_importance_top20": dict(sorted(importance.items(), key=lambda x: -x[1])[:20]),
    }

    save_results(task, "catboost", config_name, result)

    model_path = MODEL_ROOT / f"catboost_{task}_{config_name}.cbm"
    model.save_model(str(model_path))

    log(f"  CatBoost {config_name}: acc={test_metrics['accuracy']:.6f} macro_f1={test_metrics['macro_f1']:.6f} time={train_time:.0f}s")

    del train_pool, val_pool, model
    gc.collect()

    return result


# ─────────────────────── Main training loop ───────────────────────

def run_task(task, label_col, n_classes, class_names, targets):
    log(f"\n{'='*60}")
    log(f"TASK: {task} ({n_classes} classes)")
    log(f"Targets: {targets}")
    log(f"{'='*60}")

    log("Loading data...")
    X_train, y_train = load_split("train", label_col)
    X_val, y_val = load_split("val", label_col)
    X_test, y_test = load_split("test", label_col)
    log(f"Data loaded: train={len(y_train):,} val={len(y_val):,} test={len(y_test):,}")

    # Print class distribution in train
    for i in range(n_classes):
        cnt = (y_train == i).sum()
        log(f"  Class {i} ({class_names[i]}): {cnt:,} train rows")

    results = []

    # ── XGBoost Config 1 ──
    try:
        r = train_xgb(X_train, y_train, X_val, y_val, X_test, y_test,
                       task, n_classes, class_names, "cfg1",
                       {"max_depth": 12, "learning_rate": 0.1, "max_bin": 256,
                        "n_estimators": 2000, "early_stopping_rounds": 50,
                        "subsample": 0.8, "colsample_bytree": 0.8})
        results.append(r)
    except Exception as e:
        log(f"  XGBoost cfg1 FAILED: {e}")
        traceback.print_exc()

    # ── XGBoost Config 2 ──
    try:
        r = train_xgb(X_train, y_train, X_val, y_val, X_test, y_test,
                       task, n_classes, class_names, "cfg2",
                       {"max_leaves": 1024, "max_depth": 0, "grow_policy": "lossguide",
                        "learning_rate": 0.05, "max_bin": 512,
                        "n_estimators": 3000, "early_stopping_rounds": 50,
                        "subsample": 0.8, "colsample_bytree": 0.8})
        results.append(r)
    except Exception as e:
        log(f"  XGBoost cfg2 FAILED: {e}")
        traceback.print_exc()

    # ── XGBoost Config 3 (weighted) ──
    try:
        # Pick best of cfg1/cfg2
        best_xgb = max([r for r in results if "xgboost" in r["method"]],
                       key=lambda x: x["val_metrics"]["macro_f1"] if n_classes > 2 else x["val_metrics"]["accuracy"])
        best_cfg = best_xgb["config"].copy()

        weights = compute_class_weights(y_train, n_classes, power=0.5)
        sample_w = weights[y_train]

        if n_classes == 2:
            pos_count = (y_train == 1).sum()
            neg_count = (y_train == 0).sum()
            best_cfg["scale_pos_weight"] = float(neg_count / pos_count)
            sw = None
        else:
            sw = sample_w

        xgb_params = {}
        for k, v in best_cfg.items():
            if k in ("max_depth", "max_leaves", "max_bin"):
                xgb_params[k] = int(v)
            elif k in ("learning_rate", "subsample", "colsample_bytree", "scale_pos_weight"):
                xgb_params[k] = float(v)
            elif k in ("grow_policy",):
                xgb_params[k] = v
        xgb_params["n_estimators"] = 3000
        xgb_params["early_stopping_rounds"] = 50

        r = train_xgb(X_train, y_train, X_val, y_val, X_test, y_test,
                       task, n_classes, class_names, "cfg3_weighted",
                       xgb_params, sample_weights=sw)
        results.append(r)
    except Exception as e:
        log(f"  XGBoost cfg3 FAILED: {e}")
        traceback.print_exc()

    # ── LightGBM Config 1 ──
    try:
        r = train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test,
                        task, n_classes, class_names, "cfg1",
                        {"num_leaves": 512, "learning_rate": 0.1,
                         "n_estimators": 2000, "early_stopping_rounds": 50,
                         "subsample": 0.8, "colsample_bytree": 0.8,
                         "min_child_samples": 100})
        results.append(r)
    except Exception as e:
        log(f"  LightGBM cfg1 FAILED: {e}")
        traceback.print_exc()

    # ── LightGBM Config 2 ──
    try:
        r = train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test,
                        task, n_classes, class_names, "cfg2",
                        {"num_leaves": 2048, "learning_rate": 0.05,
                         "n_estimators": 3000, "early_stopping_rounds": 50,
                         "subsample": 0.8, "colsample_bytree": 0.8,
                         "min_child_samples": 100})
        results.append(r)
    except Exception as e:
        log(f"  LightGBM cfg2 FAILED: {e}")
        traceback.print_exc()

    # ── LightGBM Config 3 (weighted) ──
    try:
        best_lgbm = max([r for r in results if "lightgbm" in r["method"]],
                        key=lambda x: x["val_metrics"]["macro_f1"] if n_classes > 2 else x["val_metrics"]["accuracy"])

        weights = compute_class_weights(y_train, n_classes, power=0.5)
        sample_w = weights[y_train]

        lgbm_params = {}
        for k, v in best_lgbm["config"].items():
            if k in ("num_leaves", "min_child_samples"):
                lgbm_params[k] = int(v)
            elif k in ("learning_rate", "subsample", "colsample_bytree"):
                lgbm_params[k] = float(v)
        lgbm_params["n_estimators"] = 3000
        lgbm_params["early_stopping_rounds"] = 50

        r = train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test,
                        task, n_classes, class_names, "cfg3_weighted",
                        lgbm_params, sample_weights=sample_w.astype(np.float32))
        results.append(r)
    except Exception as e:
        log(f"  LightGBM cfg3 FAILED: {e}")
        traceback.print_exc()

    # ── CatBoost ──
    try:
        cb_params = {
            "depth": 10,
            "learning_rate": 0.1,
            "iterations": 3000,
            "early_stopping_rounds": 50,
            "auto_class_weights": "Balanced",
        }
        if n_classes == 2:
            cb_params["loss_function"] = "Logloss"
        else:
            cb_params["loss_function"] = "MultiClass"

        r = train_catboost(X_train, y_train, X_val, y_val, X_test, y_test,
                           task, n_classes, class_names, "balanced", cb_params)
        results.append(r)
    except Exception as e:
        log(f"  CatBoost FAILED: {e}")
        traceback.print_exc()

    # ── Summary ──
    log(f"\n{'─'*40}")
    log(f"TASK {task} — ALL RESULTS:")
    for r in sorted(results, key=lambda x: -(x["test_metrics"]["macro_f1"] if n_classes > 2 else x["test_metrics"]["accuracy"])):
        m = r["test_metrics"]
        log(f"  {r['method']:30s}  acc={m['accuracy']:.6f}  macro_f1={m['macro_f1']:.6f}  time={r['training_time_seconds']:.0f}s")

    if results:
        if n_classes > 2:
            best = max(results, key=lambda x: x["test_metrics"]["macro_f1"])
        else:
            best = max(results, key=lambda x: x["test_metrics"]["accuracy"])
        log(f"  BEST: {best['method']} — acc={best['test_metrics']['accuracy']:.6f} macro_f1={best['test_metrics']['macro_f1']:.6f}")

        # Save best pointer
        best_dir = RESULTS_ROOT / task / "BEST"
        os.makedirs(best_dir, exist_ok=True)
        with open(best_dir / "results.json", "w") as f:
            json.dump(best, f, indent=2)

    del X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()

    return results


def main():
    log("PHASE 2: TRAINING ALL MODELS")
    log(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    # 2-class
    all_results["2_class"] = run_task(
        "2_class", "label_2class", 2,
        ["Benign", "Attack"],
        {"accuracy": 0.9968}
    )

    # 8-class
    all_results["8_class"] = run_task(
        "8_class", "label_8class", 8,
        CLASS8_ORDER,
        {"macro_f1": 0.7193, "accuracy": 0.9944}
    )

    # 34-class
    all_results["34_class"] = run_task(
        "34_class", "label_34class", 34,
        CLASS34_ORDER,
        {"macro_f1": 0.7}
    )

    log("\n" + "="*60)
    log("PHASE 2 COMPLETE")
    log(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Print final summary
    for task in ["2_class", "8_class", "34_class"]:
        best_path = RESULTS_ROOT / task / "BEST" / "results.json"
        if best_path.exists():
            with open(best_path) as f:
                best = json.load(f)
            m = best["test_metrics"]
            log(f"  {task}: {best['method']} acc={m['accuracy']:.6f} macro_f1={m['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
