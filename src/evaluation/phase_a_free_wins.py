#!/usr/bin/env python3
"""Phase A: Free wins — threshold tuning, calibration, ensembles. CPU-only."""
import os, sys, json, time, glob, gc, ctypes
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

CACHE = Path("/dev/shm/saksham/ids_cache")
IDS = Path("/dev/shm/saksham/ids")
V3 = Path("/home/kishor_kumar/saksham/ids/v3")
RESULTS = V3 / "results"
MODELS = CACHE / "models"

with open(IDS / "src" / "class_maps.json") as f:
    CM = json.load(f)
with open(IDS / "src" / "data_meta.json") as f:
    DM = json.load(f)

FEATURES = DM["features"]
CLASS8 = CM["class8_order"]
CLASS34 = CM["class34_order"]
CHUNK = 250_000


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def free_mem():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass


def load_split(name):
    parts = sorted(glob.glob(str(CACHE / name / "*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def get_probs_xgb(model_path, X, n_classes):
    """Get predicted probabilities from XGBoost model, chunked."""
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(str(model_path))
    all_probs = []
    for i in range(0, len(X), CHUNK):
        chunk = X[i:i+CHUNK]
        dm = xgb.DMatrix(chunk)
        if n_classes == 2:
            p1 = model.predict(dm)
            probs = np.column_stack([1 - p1, p1])
        else:
            # For multi-class, need to set output_margin=False
            probs = model.predict(dm, output_margin=False)
            if probs.ndim == 1:
                # softmax output - reshape
                probs = probs.reshape(-1, n_classes)
        all_probs.append(probs)
        del dm
    del model
    free_mem()
    return np.vstack(all_probs).astype(np.float32)


def get_probs_lgbm(model_path, X, n_classes):
    """Get predicted probabilities from LightGBM model, chunked."""
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(model_path))
    all_probs = []
    for i in range(0, len(X), CHUNK):
        chunk = X[i:i+CHUNK]
        probs = model.predict(chunk)
        if n_classes == 2:
            if probs.ndim == 1:
                probs = np.column_stack([1 - probs, probs])
        all_probs.append(probs.astype(np.float32))
    del model
    free_mem()
    return np.vstack(all_probs)


def get_probs_catboost(model_path, X, n_classes):
    """Get predicted probabilities from CatBoost model, chunked."""
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    all_probs = []
    for i in range(0, len(X), CHUNK):
        chunk = X[i:i+CHUNK]
        probs = model.predict_proba(chunk)
        all_probs.append(probs.astype(np.float32))
    del model
    free_mem()
    return np.vstack(all_probs)


def compute_metrics(y_true, y_pred, n_classes, class_names):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    pf1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    return {
        "accuracy": float(acc), "macro_f1": float(mf1),
        "per_class_f1": {class_names[i]: float(pf1[i]) for i in range(n_classes)}
    }


# ============================================================
# A1: Per-class threshold tuning
# ============================================================
def threshold_tune(probs_val, y_val, probs_test, y_test, n_classes, class_names, task):
    """Sweep per-class multipliers to optimize macro-F1 on val, apply to test."""
    log(f"A1: Threshold tuning for {task}")

    # Baseline
    base_val = compute_metrics(y_val, probs_val.argmax(1), n_classes, class_names)
    base_test = compute_metrics(y_test, probs_test.argmax(1), n_classes, class_names)
    log(f"  Baseline val: acc={base_val['accuracy']:.6f} mF1={base_val['macro_f1']:.6f}")
    log(f"  Baseline test: acc={base_test['accuracy']:.6f} mF1={base_test['macro_f1']:.6f}")

    # Sweep multipliers
    best_mults = np.ones(n_classes, dtype=np.float32)
    best_f1 = base_val["macro_f1"]

    # Greedy per-class sweep
    for cls in range(n_classes):
        best_m = 1.0
        for m in np.arange(0.3, 3.05, 0.1):
            mults = best_mults.copy()
            mults[cls] = m
            scaled = probs_val * mults[np.newaxis, :]
            preds = scaled.argmax(1)
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_m = m
        best_mults[cls] = best_m
        if best_m != 1.0:
            log(f"  Class {class_names[cls]}: mult={best_m:.1f}")

    # Apply to test
    scaled_test = probs_test * best_mults[np.newaxis, :]
    tuned_preds = scaled_test.argmax(1)
    tuned_test = compute_metrics(y_test, tuned_preds, n_classes, class_names)

    log(f"  Tuned test: acc={tuned_test['accuracy']:.6f} mF1={tuned_test['macro_f1']:.6f}")
    log(f"  Improvement: acc={tuned_test['accuracy'] - base_test['accuracy']:+.6f} "
        f"mF1={tuned_test['macro_f1'] - base_test['macro_f1']:+.6f}")

    result = {
        "method": "threshold_tuning",
        "task": task,
        "multipliers": {class_names[i]: float(best_mults[i]) for i in range(n_classes)},
        "baseline_test": base_test,
        "tuned_test": tuned_test,
        "val_macro_f1_before": base_val["macro_f1"],
        "val_macro_f1_after": float(best_f1),
    }
    out = RESULTS / task / "threshold_tuned"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ============================================================
# A2: Probability calibration
# ============================================================
def calibrate(probs_val, y_val, probs_test, y_test, n_classes, class_names, task):
    """Isotonic and Platt calibration."""
    log(f"A2: Calibration for {task}")

    base_test = compute_metrics(y_test, probs_test.argmax(1), n_classes, class_names)
    results = {"method": "calibration", "task": task, "baseline_test": base_test}

    for method_name in ["isotonic", "sigmoid"]:
        log(f"  Trying {method_name}...")
        # Simple calibration: fit per-class isotonic/sigmoid on val probs
        # We'll use a simple approach: sklearn CalibratedClassifierCV expects an estimator,
        # but we can do manual per-class calibration
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression as LR

        cal_probs_test = np.zeros_like(probs_test)
        for cls in range(n_classes):
            y_bin = (y_val == cls).astype(int)
            p_val = probs_val[:, cls]
            p_test = probs_test[:, cls]

            if method_name == "isotonic":
                cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                cal.fit(p_val, y_bin)
                cal_probs_test[:, cls] = cal.predict(p_test)
            else:
                lr = LR(max_iter=1000)
                lr.fit(p_val.reshape(-1, 1), y_bin)
                cal_probs_test[:, cls] = lr.predict_proba(p_test.reshape(-1, 1))[:, 1]

        preds = cal_probs_test.argmax(1)
        m = compute_metrics(y_test, preds, n_classes, class_names)
        log(f"  {method_name}: acc={m['accuracy']:.6f} mF1={m['macro_f1']:.6f}")
        results[f"{method_name}_test"] = m

    out = RESULTS / task / "calibrated"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
# A3: Ensemble existing models
# ============================================================
def ensemble_models(task, label_col, n_classes, class_names, model_specs):
    """Soft-voting, weighted, and stacking ensemble."""
    log(f"A3: Ensemble for {task}")

    val_df = load_split("val")
    test_df = load_split("test")
    X_val = val_df[FEATURES].values.astype(np.float32)
    y_val = val_df[label_col].values.astype(np.int32)
    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.int32)
    del val_df, test_df
    free_mem()

    all_val_probs = []
    all_test_probs = []
    model_f1s = []

    for name, model_path, model_type in model_specs:
        log(f"  Loading {name} ({model_type})...")
        if model_type == "xgboost":
            vp = get_probs_xgb(model_path, X_val, n_classes)
            tp = get_probs_xgb(model_path, X_test, n_classes)
        elif model_type == "lightgbm":
            vp = get_probs_lgbm(model_path, X_val, n_classes)
            tp = get_probs_lgbm(model_path, X_test, n_classes)
        elif model_type == "catboost":
            vp = get_probs_catboost(model_path, X_val, n_classes)
            tp = get_probs_catboost(model_path, X_test, n_classes)
        else:
            continue

        # Validate shapes
        if vp.ndim == 1 or vp.shape[1] != n_classes:
            log(f"    SKIP: bad shape {vp.shape}")
            continue

        mf1 = f1_score(y_val, vp.argmax(1), average="macro", zero_division=0)
        log(f"    val mF1={mf1:.6f}")
        all_val_probs.append(vp)
        all_test_probs.append(tp)
        model_f1s.append(mf1)

    results = {"method": "ensemble", "task": task, "n_models": len(all_val_probs)}

    # Soft-voting
    avg_test = np.mean(all_test_probs, axis=0)
    avg_val = np.mean(all_val_probs, axis=0)
    m = compute_metrics(y_test, avg_test.argmax(1), n_classes, class_names)
    log(f"  Soft-vote: acc={m['accuracy']:.6f} mF1={m['macro_f1']:.6f}")
    results["soft_vote_test"] = m

    # Weighted voting
    weights = np.array(model_f1s)
    weights = weights / weights.sum()
    wavg_test = sum(w * p for w, p in zip(weights, all_test_probs))
    wavg_val = sum(w * p for w, p in zip(weights, all_val_probs))
    m = compute_metrics(y_test, wavg_test.argmax(1), n_classes, class_names)
    log(f"  Weighted-vote: acc={m['accuracy']:.6f} mF1={m['macro_f1']:.6f}")
    results["weighted_vote_test"] = m

    # Stacking
    log(f"  Stacking (LogReg on probs)...")
    stack_val = np.hstack(all_val_probs)
    stack_test = np.hstack(all_test_probs)
    meta = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    meta.fit(stack_val, y_val)
    stack_preds = meta.predict(stack_test)
    m = compute_metrics(y_test, stack_preds, n_classes, class_names)
    log(f"  Stacking: acc={m['accuracy']:.6f} mF1={m['macro_f1']:.6f}")
    results["stacking_test"] = m

    # Threshold tuning on best ensemble
    best_method = max(["soft_vote_test", "weighted_vote_test", "stacking_test"],
                      key=lambda k: results[k]["macro_f1"])
    log(f"  Best ensemble: {best_method} mF1={results[best_method]['macro_f1']:.6f}")

    # Apply threshold tuning on soft-vote (usually best for tuning)
    log(f"  Threshold tuning on soft-vote...")
    tuned = threshold_tune_on_probs(avg_val, y_val, avg_test, y_test, n_classes, class_names)
    results["soft_vote_tuned_test"] = tuned

    out = RESULTS / task / "ensemble"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    del all_val_probs, all_test_probs, X_val, X_test
    free_mem()
    return results


def threshold_tune_on_probs(probs_val, y_val, probs_test, y_test, n_classes, class_names):
    """Quick threshold tuning helper."""
    best_mults = np.ones(n_classes, dtype=np.float32)
    best_f1 = f1_score(y_val, probs_val.argmax(1), average="macro", zero_division=0)
    for cls in range(n_classes):
        best_m = 1.0
        for m in np.arange(0.3, 3.05, 0.1):
            mults = best_mults.copy()
            mults[cls] = m
            scaled = probs_val * mults[np.newaxis, :]
            f1 = f1_score(y_val, scaled.argmax(1), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_m = m
        best_mults[cls] = best_m
    scaled_test = probs_test * best_mults[np.newaxis, :]
    return compute_metrics(y_test, scaled_test.argmax(1), n_classes, class_names)


# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 60)
    log("PHASE A: FREE WINS (CPU-only)")
    log("=" * 60)

    # Load val and test data
    log("Loading val/test data...")
    val_df = load_split("val")
    test_df = load_split("test")
    X_val = val_df[FEATURES].values.astype(np.float32)
    X_test = test_df[FEATURES].values.astype(np.float32)

    tasks = [
        ("2_class", "label_2class", 2, ["Benign", "Attack"],
         [("lgbm_cfg2", MODELS / "lgbm_2_class_cfg2.txt", "lightgbm"),
          ("xgb_cfg2", MODELS / "xgb_2_class_cfg2.json", "xgboost"),
          ("xgb_deep", MODELS / "xgb_2_class_deep.json", "xgboost"),
          ("lgbm_cfg1", MODELS / "lgbm_2_class_cfg1.txt", "lightgbm"),
          ("catboost_bal", MODELS / "catboost_2_class_balanced.cbm", "catboost")]),
        ("34_class", "label_34class", 34, CLASS34,
         [("xgb_bal_p050", MODELS / "xgb_34_class_balanced_p050.json", "xgboost"),
          ("xgb_bal_deep", MODELS / "xgb_34_class_balanced_deep.ubj", "xgboost"),
          ("lgbm_cfg2", MODELS / "lgbm_34_class_cfg2.txt", "lightgbm") if (MODELS / "lgbm_34_class_cfg2.txt").exists() else None,
          ("catboost_bal", MODELS / "catboost_34_class_balanced.cbm", "catboost")]),
    ]

    for task, label_col, n_classes, class_names, model_specs in tasks:
        y_val = val_df[label_col].values.astype(np.int32)
        y_test = test_df[label_col].values.astype(np.int32)

        log(f"\n{'='*60}")
        log(f"TASK: {task}")
        log(f"{'='*60}")

        # Get probs from best model for A1 + A2
        best_spec = model_specs[0]
        best_name, best_path, best_type = best_spec
        log(f"Getting probs from best model: {best_name}")

        if best_type == "lightgbm":
            probs_val = get_probs_lgbm(best_path, X_val, n_classes)
            probs_test = get_probs_lgbm(best_path, X_test, n_classes)
        elif best_type == "xgboost":
            probs_val = get_probs_xgb(best_path, X_val, n_classes)
            probs_test = get_probs_xgb(best_path, X_test, n_classes)
        elif best_type == "catboost":
            probs_val = get_probs_catboost(best_path, X_val, n_classes)
            probs_test = get_probs_catboost(best_path, X_test, n_classes)

        # A1: Threshold tuning
        a1 = threshold_tune(probs_val, y_val, probs_test, y_test,
                            n_classes, class_names, task)

        # A2: Calibration
        a2 = calibrate(probs_val, y_val, probs_test, y_test,
                        n_classes, class_names, task)

        del probs_val, probs_test
        free_mem()

        # A3: Ensemble
        valid_specs = [s for s in model_specs if s is not None and Path(s[1]).exists()]
        a3 = ensemble_models(task, label_col, n_classes, class_names, valid_specs)

    del val_df, test_df, X_val, X_test
    free_mem()

    # Summary
    log("\n" + "=" * 60)
    log("PHASE A COMPLETE — Summary")
    log("=" * 60)
    for task in ["2_class", "34_class"]:
        log(f"\n{task}:")
        for sub in ["threshold_tuned", "calibrated", "ensemble"]:
            rp = RESULTS / task / sub / "results.json"
            if rp.exists():
                r = json.load(open(rp))
                if sub == "threshold_tuned":
                    log(f"  Threshold: acc={r['tuned_test']['accuracy']:.6f} mF1={r['tuned_test']['macro_f1']:.6f}")
                elif sub == "calibrated":
                    for m in ["isotonic", "sigmoid"]:
                        k = f"{m}_test"
                        if k in r:
                            log(f"  Calib-{m}: acc={r[k]['accuracy']:.6f} mF1={r[k]['macro_f1']:.6f}")
                elif sub == "ensemble":
                    for k in ["soft_vote_test", "weighted_vote_test", "stacking_test", "soft_vote_tuned_test"]:
                        if k in r:
                            log(f"  {k}: acc={r[k]['accuracy']:.6f} mF1={r[k]['macro_f1']:.6f}")

    # Check gaps
    log("\nGap check:")
    for task, target_key, target_val in [("2_class", "accuracy", 0.9968), ("34_class", "macro_f1", 0.70)]:
        best_val = 0
        best_name = ""
        for sub in ["threshold_tuned", "calibrated", "ensemble"]:
            rp = RESULTS / task / sub / "results.json"
            if rp.exists():
                r = json.load(open(rp))
                for k, v in r.items():
                    if isinstance(v, dict) and target_key in v:
                        if v[target_key] > best_val:
                            best_val = v[target_key]
                            best_name = f"{sub}/{k}"
        met = best_val >= target_val
        log(f"  {task}: best {target_key}={best_val:.6f} ({best_name}) -> {'MET' if met else 'NOT MET (need Phase B)'}")


if __name__ == "__main__":
    main()
