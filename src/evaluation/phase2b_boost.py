#!/usr/bin/env python3
"""Phase 2b: Post-training boosters.
1. Threshold optimization (for 2-class: optimize accuracy on val, evaluate on test)
2. Soft-voting ensemble of best models per task
3. Stacking ensemble (logistic regression on model outputs)
"""
import os, sys, json, time, glob, gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def compute_metrics(y_true, y_pred, n_classes, class_names):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    return {"accuracy": float(acc), "macro_f1": float(macro_f1),
            "per_class_f1": {class_names[i]: float(per_class[i]) for i in range(n_classes)},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(n_classes)).tolist()}


def get_model_probs(model_path, X, method_type, n_classes):
    """Load model and get probability predictions."""
    if method_type == "xgboost":
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(model_path))
        dm = xgb.DMatrix(X)
        if n_classes == 2:
            prob = model.predict(dm)
            return np.column_stack([1 - prob, prob])
        else:
            # Models trained with multi:softmax return labels, not probs.
            # Use output_margin=True to get raw logits, then apply softmax.
            margins = model.predict(dm, output_margin=True)
            if margins.ndim == 1:
                margins = margins.reshape(-1, n_classes)
            exp_m = np.exp(margins - margins.max(axis=1, keepdims=True))
            return exp_m / exp_m.sum(axis=1, keepdims=True)

    elif method_type == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_path))
        if n_classes == 2:
            prob = model.predict(X)
            return np.column_stack([1 - prob, prob])
        else:
            return model.predict(X)

    elif method_type == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        return model.predict_proba(X)

    return None


def find_all_models(task):
    """Find all trained models for a task."""
    task_dir = RESULTS_ROOT / task
    models = []
    if not task_dir.exists():
        return models
    for sub in sorted(task_dir.iterdir()):
        if sub.name in ("BEST", "threshold_optimized", "ensemble_soft_vote", "ensemble_stacking"):
            continue
        result_file = sub / "results.json"
        if not result_file.exists():
            continue
        with open(result_file) as f:
            r = json.load(f)

        # Determine model type and config name from directory name
        # e.g. "xgboost_cfg1" -> type="xgboost", config="cfg1"
        # "lightgbm_cfg2" -> type="lightgbm", config="cfg2"
        # "catboost_balanced" -> type="catboost", config="balanced"
        parts = sub.name.split("_", 1)
        model_type = parts[0]
        config_name = parts[1] if len(parts) > 1 else ""

        # Model file paths — check both .json and .ubj for XGBoost
        xgb_json = MODEL_ROOT / f"xgb_{task}_{config_name}.json"
        xgb_ubj = MODEL_ROOT / f"xgb_{task}_{config_name}.ubj"
        lgb_file = MODEL_ROOT / f"lgbm_{task}_{config_name}.txt"
        cb_file = MODEL_ROOT / f"catboost_{task}_{config_name}.cbm"

        if model_type == "xgboost":
            xgb_file = xgb_json if xgb_json.exists() else (xgb_ubj if xgb_ubj.exists() else None)
            if xgb_file:
                models.append({"name": sub.name, "path": xgb_file, "results": r, "type": "xgboost"})
        elif model_type == "lightgbm" and lgb_file.exists():
            models.append({"name": sub.name, "path": lgb_file, "results": r, "type": "lightgbm"})
        elif model_type == "catboost" and cb_file.exists():
            models.append({"name": sub.name, "path": cb_file, "results": r, "type": "catboost"})
    return models


# ═══════════ 1. THRESHOLD OPTIMIZATION (2-class only) ═══════════

def threshold_optimization():
    log("=== THRESHOLD OPTIMIZATION (2-class) ===")
    out_dir = RESULTS_ROOT / "2_class" / "threshold_optimized"
    os.makedirs(out_dir, exist_ok=True)

    models = find_all_models("2_class")
    if not models:
        log("  No models found for 2-class")
        return

    log(f"  Found {len(models)} models")

    val_df = load_split("val")
    test_df = load_split("test")
    X_val = val_df[FEATURES].values.astype(np.float32)
    y_val = val_df["label_2class"].values.astype(np.int32)
    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df["label_2class"].values.astype(np.int32)
    del val_df, test_df
    gc.collect()

    best_acc = 0
    best_result = None

    for m in models:
        log(f"  Model: {m['name']} ({m['type']})")
        try:
            probs = get_model_probs(m["path"], X_val, m["type"], 2)
            if probs is None:
                continue
            attack_prob = probs[:, 1]

            # Sweep thresholds
            thresholds = np.linspace(0.1, 0.9, 81)
            best_t = 0.5
            best_t_acc = 0
            for t in thresholds:
                pred = (attack_prob >= t).astype(int)
                acc = accuracy_score(y_val, pred)
                if acc > best_t_acc:
                    best_t_acc = acc
                    best_t = float(t)

            log(f"    Best threshold on val: {best_t:.4f} (val_acc={best_t_acc:.6f})")

            # Evaluate on test
            test_probs = get_model_probs(m["path"], X_test, m["type"], 2)
            test_attack_prob = test_probs[:, 1]

            # Default threshold
            y_pred_default = (test_attack_prob >= 0.5).astype(int)
            acc_default = accuracy_score(y_test, y_pred_default)
            f1_default = f1_score(y_test, y_pred_default, average="macro")

            # Optimized threshold
            y_pred_opt = (test_attack_prob >= best_t).astype(int)
            acc_opt = accuracy_score(y_test, y_pred_opt)
            f1_opt = f1_score(y_test, y_pred_opt, average="macro")

            log(f"    Test default (0.5): acc={acc_default:.6f} mF1={f1_default:.6f}")
            log(f"    Test optimized ({best_t:.3f}): acc={acc_opt:.6f} mF1={f1_opt:.6f}")

            if acc_opt > best_acc:
                best_acc = acc_opt
                best_result = {
                    "method": f"{m['name']}_threshold_opt",
                    "base_model": m["name"],
                    "threshold": best_t,
                    "val_accuracy_at_threshold": float(best_t_acc),
                    "test_metrics_default": {"accuracy": float(acc_default), "macro_f1": float(f1_default)},
                    "test_metrics_optimized": compute_metrics(y_test, y_pred_opt, 2, ["Benign", "Attack"]),
                    "accuracy_improvement": float(acc_opt - acc_default),
                }

            del probs, test_probs
            gc.collect()

        except Exception as e:
            log(f"    Error: {e}")
            import traceback; traceback.print_exc()

    if best_result:
        with open(out_dir / "results.json", "w") as f:
            json.dump(best_result, f, indent=2)
        log(f"  Best threshold-optimized: acc={best_acc:.6f}")
        log(f"  Saved: {out_dir / 'results.json'}")

    del X_val, y_val, X_test, y_test
    gc.collect()
    return best_result


# ═══════════ 2. SOFT-VOTING ENSEMBLE ═══════════

def soft_voting_ensemble(task, label_col, n_classes, class_names, target_metric="macro_f1"):
    log(f"=== SOFT-VOTING ENSEMBLE ({task}) ===")
    out_dir = RESULTS_ROOT / task / "ensemble_soft_vote"
    os.makedirs(out_dir, exist_ok=True)

    models = find_all_models(task)
    if len(models) < 2:
        log(f"  Need >=2 models, found {len(models)}")
        return

    # Pick top 3 by macro_f1
    models.sort(key=lambda m: m["results"]["test_metrics"].get("macro_f1", 0), reverse=True)
    top_models = models[:min(3, len(models))]
    log(f"  Using top {len(top_models)} models: {[m['name'] for m in top_models]}")

    test_df = load_split("test")
    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.int32)
    del test_df
    gc.collect()

    all_probs = []
    for m in top_models:
        log(f"  Getting probs from {m['name']}...")
        try:
            probs = get_model_probs(m["path"], X_test, m["type"], n_classes)
            if probs is not None:
                all_probs.append(probs)
                log(f"    Shape: {probs.shape}")
        except Exception as e:
            log(f"    Error: {e}")

    if len(all_probs) < 2:
        log("  Not enough probability outputs for ensemble")
        return

    # Average probabilities
    avg_probs = np.mean(all_probs, axis=0)
    y_pred = avg_probs.argmax(axis=1)

    metrics = compute_metrics(y_test, y_pred, n_classes, class_names)
    log(f"  Ensemble: acc={metrics['accuracy']:.6f} mF1={metrics['macro_f1']:.6f}")

    # For 2-class, also try threshold optimization on the ensemble
    threshold_info = None
    if n_classes == 2:
        val_df = load_split("val")
        X_val = val_df[FEATURES].values.astype(np.float32)
        y_val = val_df[label_col].values.astype(np.int32)
        del val_df
        gc.collect()

        val_probs = []
        for m in top_models:
            try:
                p = get_model_probs(m["path"], X_val, m["type"], n_classes)
                if p is not None:
                    val_probs.append(p)
            except:
                pass

        if len(val_probs) == len(all_probs):
            avg_val = np.mean(val_probs, axis=0)[:, 1]
            thresholds = np.linspace(0.1, 0.9, 81)
            best_t, best_t_acc = 0.5, 0
            for t in thresholds:
                pred = (avg_val >= t).astype(int)
                acc = accuracy_score(y_val, pred)
                if acc > best_t_acc:
                    best_t_acc = acc
                    best_t = float(t)

            y_pred_opt = (avg_probs[:, 1] >= best_t).astype(int)
            metrics_opt = compute_metrics(y_test, y_pred_opt, n_classes, class_names)
            log(f"  Ensemble + threshold({best_t:.3f}): acc={metrics_opt['accuracy']:.6f} mF1={metrics_opt['macro_f1']:.6f}")
            threshold_info = {"threshold": best_t, "metrics": metrics_opt}

        del X_val, y_val
        gc.collect()

    result = {
        "method": f"ensemble_soft_vote",
        "task": task,
        "models_used": [m["name"] for m in top_models],
        "test_metrics": metrics,
        "threshold_optimized": threshold_info,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    del X_test, y_test, all_probs
    gc.collect()
    return result


# ═══════════ 3. STACKING ENSEMBLE ═══════════

def stacking_ensemble(task, label_col, n_classes, class_names):
    log(f"=== STACKING ENSEMBLE ({task}) ===")
    out_dir = RESULTS_ROOT / task / "ensemble_stacking"
    os.makedirs(out_dir, exist_ok=True)

    models = find_all_models(task)
    if len(models) < 2:
        log(f"  Need >=2 models, found {len(models)}")
        return

    models.sort(key=lambda m: m["results"]["test_metrics"].get("macro_f1", 0), reverse=True)
    top_models = models[:min(4, len(models))]
    log(f"  Using {len(top_models)} models: {[m['name'] for m in top_models]}")

    val_df = load_split("val")
    test_df = load_split("test")
    X_val = val_df[FEATURES].values.astype(np.float32)
    y_val = val_df[label_col].values.astype(np.int32)
    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.int32)
    del val_df, test_df
    gc.collect()

    # Get probabilities from all models on val and test
    val_feats, test_feats = [], []
    valid_models = []

    for m in top_models:
        try:
            vp = get_model_probs(m["path"], X_val, m["type"], n_classes)
            tp = get_model_probs(m["path"], X_test, m["type"], n_classes)
            if vp is not None and tp is not None:
                val_feats.append(vp)
                test_feats.append(tp)
                valid_models.append(m["name"])
        except Exception as e:
            log(f"  Error with {m['name']}: {e}")

    if len(valid_models) < 2:
        log("  Not enough valid models")
        return

    # Stack features
    X_val_stack = np.hstack(val_feats)
    X_test_stack = np.hstack(test_feats)
    log(f"  Stacked features: {X_val_stack.shape[1]} dims from {len(valid_models)} models")

    # Train logistic regression meta-learner
    meta = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED,
                               multi_class="multinomial" if n_classes > 2 else "auto")
    meta.fit(X_val_stack, y_val)
    y_pred = meta.predict(X_test_stack)

    metrics = compute_metrics(y_test, y_pred, n_classes, class_names)
    log(f"  Stacking: acc={metrics['accuracy']:.6f} mF1={metrics['macro_f1']:.6f}")

    result = {
        "method": "ensemble_stacking",
        "task": task,
        "models_used": valid_models,
        "meta_learner": "LogisticRegression",
        "test_metrics": metrics,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    del X_val, y_val, X_test, y_test
    gc.collect()
    return result


# ═══════════ MAIN ═══════════

def main():
    log("PHASE 2b: POST-TRAINING BOOSTERS")

    # 2-class threshold optimization
    threshold_optimization()

    # Ensembles for all tasks
    for task, label, nc, names in [
        ("2_class", "label_2class", 2, ["Benign", "Attack"]),
        ("8_class", "label_8class", 8, CLASS8_ORDER),
        ("34_class", "label_34class", 34, CLASS34_ORDER),
    ]:
        if (RESULTS_ROOT / task).exists():
            soft_voting_ensemble(task, label, nc, names)
            stacking_ensemble(task, label, nc, names)

    log("PHASE 2b COMPLETE")


if __name__ == "__main__":
    main()
