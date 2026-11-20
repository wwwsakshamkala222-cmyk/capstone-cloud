#!/usr/bin/env python3
"""Multi-seed evaluation for final models.
Retrains best configs with 3 seeds and reports mean metrics.
"""
import os, sys, json, time, glob, gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "16"

CACHE_ROOT = Path("/dev/shm/saksham/ids_cache")
IDS_ROOT = Path("/dev/shm/saksham/ids")
RESULTS_ROOT = IDS_ROOT / "results"

with open(IDS_ROOT / "src" / "class_maps.json") as f:
    CLASS_MAPS = json.load(f)
with open(IDS_ROOT / "src" / "data_meta.json") as f:
    DATA_META = json.load(f)

CLASS8_ORDER = CLASS_MAPS["class8_order"]
CLASS34_ORDER = CLASS_MAPS["class34_order"]
FEATURES = DATA_META["features"]
SEEDS = [42, 123, 456]


def load_split(split_name):
    parts = sorted(glob.glob(str(CACHE_ROOT / split_name / "*.parquet")))
    dfs = [pd.read_parquet(p) for p in parts]
    return pd.concat(dfs, ignore_index=True)


def compute_metrics(y_true, y_pred, n_classes, class_names):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    return {"accuracy": float(acc), "macro_f1": float(macro_f1),
            "per_class_f1": {class_names[i]: float(per_class_f1[i]) for i in range(n_classes)}}


def retrain_xgb_with_seed(X_train, y_train, X_val, y_val, X_test, y_test,
                           n_classes, class_names, config, seed, sample_weights=None):
    import xgboost as xgb

    params = dict(config)
    if n_classes == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softmax"
        params["num_class"] = n_classes
        params["eval_metric"] = "mlogloss"

    params["tree_method"] = "gpu_hist"
    params["seed"] = seed
    params["verbosity"] = 0

    n_est = int(params.pop("n_estimators", 2000))
    early = int(params.pop("early_stopping_rounds", 50))

    # Clean up non-xgb keys
    for k in list(params.keys()):
        if k not in ("max_depth", "max_leaves", "learning_rate", "max_bin",
                     "subsample", "colsample_bytree", "grow_policy", "scale_pos_weight",
                     "objective", "eval_metric", "tree_method", "seed", "verbosity", "num_class"):
            del params[k]

    # Ensure numeric types
    for k in ("max_depth", "max_leaves", "max_bin"):
        if k in params:
            params[k] = int(params[k])
    for k in ("learning_rate", "subsample", "colsample_bytree", "scale_pos_weight"):
        if k in params:
            params[k] = float(params[k])

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(params, dtrain, num_boost_round=n_est,
                       evals=[(dval, "val")], early_stopping_rounds=early,
                       verbose_eval=False)

    if n_classes == 2:
        y_pred = (model.predict(dtest) > 0.5).astype(int)
    else:
        y_pred = model.predict(dtest).astype(int)

    return compute_metrics(y_test, y_pred, n_classes, class_names)


def retrain_lgbm_with_seed(X_train, y_train, X_val, y_val, X_test, y_test,
                            n_classes, class_names, config, seed, sample_weights=None):
    import lightgbm as lgb

    params = dict(config)
    if n_classes == 2:
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"
    else:
        params["objective"] = "multiclass"
        params["num_class"] = n_classes
        params["metric"] = "multi_logloss"

    params["device"] = "cpu"
    params["seed"] = seed
    params["verbose"] = -1
    params["num_threads"] = 16
    params["force_col_wise"] = True

    n_est = int(params.pop("n_estimators", 2000))
    early = int(params.pop("early_stopping_rounds", 50))

    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights,
                         feature_name=FEATURES, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURES,
                       reference=dtrain, free_raw_data=False)

    model = lgb.train(params, dtrain, num_boost_round=n_est,
                       valid_sets=[dval], callbacks=[lgb.early_stopping(early)],)

    if n_classes == 2:
        y_pred = (model.predict(X_test) > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test).argmax(axis=1)

    return compute_metrics(y_test, y_pred, n_classes, class_names)


def retrain_catboost_with_seed(X_train, y_train, X_val, y_val, X_test, y_test,
                                n_classes, class_names, config, seed):
    from catboost import CatBoostClassifier, Pool

    params = dict(config)
    model = CatBoostClassifier(
        task_type="GPU", devices="0", random_seed=seed, verbose=0,
        train_dir=f"/dev/shm/saksham/ids/logs/catboost_multiseed_{seed}",
        **params,
    )
    train_pool = Pool(X_train, label=y_train, feature_names=FEATURES)
    val_pool = Pool(X_val, label=y_val, feature_names=FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    y_pred = model.predict(X_test).astype(int).flatten()
    return compute_metrics(y_test, y_pred, n_classes, class_names)


def run_multi_seed(task, label_col, n_classes, class_names):
    print(f"\n{'='*60}")
    print(f"Multi-seed eval for {task}")

    best_path = RESULTS_ROOT / task / "BEST" / "results.json"
    if not best_path.exists():
        print(f"  No best model for {task}")
        return

    with open(best_path) as f:
        best = json.load(f)

    method_type = best["method"].split("_")[0]  # xgboost, lightgbm, catboost
    config = best.get("config", {})

    print(f"  Best: {best['method']}")
    print(f"  Loading data...")

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

    # Check if sample weights were used
    sample_weights = None
    if "weighted" in best["method"] or "balanced" in best["method"]:
        counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
        counts = np.maximum(counts, 1)
        N = len(y_train)
        power = float(config.get("weight_power", 0.5))
        weights = (N / (n_classes * counts)) ** power
        sample_weights = weights[y_train].astype(np.float32)
        print(f"  Using sample weights with power={power}")

    all_metrics = []
    for seed in SEEDS:
        print(f"  Seed {seed}...")
        t0 = time.time()

        if method_type == "xgboost":
            m = retrain_xgb_with_seed(X_train, y_train, X_val, y_val, X_test, y_test,
                                       n_classes, class_names, config, seed,
                                       sample_weights=sample_weights)
        elif method_type == "lightgbm":
            m = retrain_lgbm_with_seed(X_train, y_train, X_val, y_val, X_test, y_test,
                                        n_classes, class_names, config, seed,
                                        sample_weights=sample_weights)
        elif method_type == "catboost":
            m = retrain_catboost_with_seed(X_train, y_train, X_val, y_val, X_test, y_test,
                                            n_classes, class_names, config, seed)
        else:
            print(f"  Unknown method type: {method_type}")
            continue

        all_metrics.append(m)
        print(f"    acc={m['accuracy']:.6f} macro_f1={m['macro_f1']:.6f} ({time.time()-t0:.0f}s)")

    # Compute mean and std
    mean_acc = np.mean([m["accuracy"] for m in all_metrics])
    std_acc = np.std([m["accuracy"] for m in all_metrics])
    mean_f1 = np.mean([m["macro_f1"] for m in all_metrics])
    std_f1 = np.std([m["macro_f1"] for m in all_metrics])

    mean_per_class = {}
    for cls in class_names:
        vals = [m["per_class_f1"][cls] for m in all_metrics]
        mean_per_class[cls] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    result = {
        "method": best["method"],
        "task": task,
        "seeds": SEEDS,
        "per_seed_metrics": all_metrics,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "mean_macro_f1": float(mean_f1),
        "std_macro_f1": float(std_f1),
        "mean_per_class_f1": mean_per_class,
    }

    out_path = RESULTS_ROOT / task / "BEST" / "multi_seed_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Mean accuracy: {mean_acc:.6f} ± {std_acc:.6f}")
    print(f"  Mean macro-F1: {mean_f1:.6f} ± {std_f1:.6f}")
    print(f"  Per-class F1:")
    for cls in class_names:
        d = mean_per_class[cls]
        print(f"    {cls:20s}: {d['mean']:.4f} ± {d['std']:.4f}")

    print(f"  Saved: {out_path}")

    del X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()


def main():
    print("MULTI-SEED EVALUATION (3 seeds)")

    run_multi_seed("2_class", "label_2class", 2, ["Benign", "Attack"])
    run_multi_seed("8_class", "label_8class", 8, CLASS8_ORDER)
    run_multi_seed("34_class", "label_34class", 34, CLASS34_ORDER)

    print("\nDONE")


if __name__ == "__main__":
    main()
