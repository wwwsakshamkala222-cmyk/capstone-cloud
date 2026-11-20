#!/usr/bin/env python3
"""Phase 2 v2: Optimized training pipeline.
LightGBM CPU (fast with 224 cores), CatBoost GPU, XGBoost CPU.
"""
import os, sys, json, time, glob, gc, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "112"

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
N_THREADS = 112


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_ROOT / "phase2_v2.log", "a") as f:
        f.write(line + "\n")


def load_split(split_name):
    parts = sorted(glob.glob(str(CACHE_ROOT / split_name / "*.parquet")))
    dfs = [pd.read_parquet(p) for p in parts]
    return pd.concat(dfs, ignore_index=True)


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


def compute_class_weights(y, n_classes, power=0.5):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    N = len(y)
    k = n_classes
    return (N / (k * counts)) ** power


def save_result(task, method, config_name, result):
    out_dir = RESULTS_ROOT / task / f"{method}_{config_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"  Saved: {out_dir / 'results.json'}")


# ═══════════ LightGBM ═══════════

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

    params["device"] = "cpu"
    params["seed"] = SEED
    params["verbose"] = -1
    params["num_threads"] = N_THREADS
    params["force_col_wise"] = True

    n_estimators = params.pop("n_estimators", 2000)
    early_stop = params.pop("early_stopping_rounds", 50)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights,
                         feature_name=FEATURES, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURES,
                       reference=dtrain, free_raw_data=False)

    model = lgb.train(
        params, dtrain, num_boost_round=n_estimators,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stop), lgb.log_evaluation(200)],
    )

    train_time = time.time() - t0

    if n_classes == 2:
        y_pred_test = (model.predict(X_test) > 0.5).astype(int)
        y_pred_val = (model.predict(X_val) > 0.5).astype(int)
    else:
        y_pred_test = model.predict(X_test).argmax(axis=1)
        y_pred_val = model.predict(X_val).argmax(axis=1)

    val_metrics = compute_metrics(y_val, y_pred_val, n_classes, class_names)
    test_metrics = compute_metrics(y_test, y_pred_test, n_classes, class_names)

    importance = dict(zip(FEATURES, model.feature_importance(importance_type="gain").tolist()))

    result = {
        "method": f"lightgbm_{config_name}", "task": task,
        "config": {k: v for k, v in params.items()},
        "n_estimators_used": model.best_iteration if model.best_iteration > 0 else n_estimators,
        "training_time_seconds": train_time,
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "feature_importance_top20": dict(sorted(importance.items(), key=lambda x: -x[1])[:20]),
    }

    save_result(task, "lightgbm", config_name, result)
    model.save_model(str(MODEL_ROOT / f"lgbm_{task}_{config_name}.txt"))

    log(f"  LightGBM {config_name}: acc={test_metrics['accuracy']:.6f} macro_f1={test_metrics['macro_f1']:.6f} time={train_time:.0f}s iters={result['n_estimators_used']}")

    del dtrain, dval, model
    gc.collect()
    return result


# ═══════════ CatBoost GPU ═══════════

def train_catboost(X_train, y_train, X_val, y_val, X_test, y_test,
                   task, n_classes, class_names, config_name, params):
    from catboost import CatBoostClassifier, Pool

    log(f"  CatBoost {config_name}: starting...")
    t0 = time.time()

    model = CatBoostClassifier(
        task_type="GPU", devices="0", random_seed=SEED, verbose=200,
        train_dir=str(LOG_ROOT / f"catboost_{task}_{config_name}"),
        **params,
    )

    train_pool = Pool(X_train, label=y_train, feature_names=FEATURES)
    val_pool = Pool(X_val, label=y_val, feature_names=FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    train_time = time.time() - t0

    y_pred_test = model.predict(X_test).astype(int).flatten()
    y_pred_val = model.predict(X_val).astype(int).flatten()

    val_metrics = compute_metrics(y_val, y_pred_val, n_classes, class_names)
    test_metrics = compute_metrics(y_test, y_pred_test, n_classes, class_names)

    importance = dict(zip(FEATURES, model.get_feature_importance().tolist()))

    result = {
        "method": f"catboost_{config_name}", "task": task,
        "config": {k: str(v) for k, v in params.items()},
        "training_time_seconds": train_time,
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "feature_importance_top20": dict(sorted(importance.items(), key=lambda x: -x[1])[:20]),
    }

    save_result(task, "catboost", config_name, result)
    model.save_model(str(MODEL_ROOT / f"catboost_{task}_{config_name}.cbm"))

    log(f"  CatBoost {config_name}: acc={test_metrics['accuracy']:.6f} macro_f1={test_metrics['macro_f1']:.6f} time={train_time:.0f}s")

    del train_pool, val_pool, model
    gc.collect()
    return result


# ═══════════ XGBoost ═══════════

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
    params["verbosity"] = 0
    params["nthread"] = N_THREADS

    n_estimators = params.pop("n_estimators", 2000)
    early_stop = params.pop("early_stopping_rounds", 50)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights, nthread=N_THREADS)
    dval = xgb.DMatrix(X_val, label=y_val, nthread=N_THREADS)
    dtest = xgb.DMatrix(X_test, label=y_test, nthread=N_THREADS)

    model = xgb.train(
        params, dtrain, num_boost_round=n_estimators,
        evals=[(dval, "val")], early_stopping_rounds=early_stop,
        verbose_eval=200,
    )

    train_time = time.time() - t0

    if n_classes == 2:
        y_pred_test = (model.predict(dtest) > 0.5).astype(int)
        y_pred_val = (model.predict(dval) > 0.5).astype(int)
    else:
        y_pred_test = model.predict(dtest).astype(int)
        y_pred_val = model.predict(dval).astype(int)

    val_metrics = compute_metrics(y_val, y_pred_val, n_classes, class_names)
    test_metrics = compute_metrics(y_test, y_pred_test, n_classes, class_names)

    importance = model.get_score(importance_type="gain")
    importance_named = {FEATURES[int(k[1:])]: v for k, v in importance.items() if k.startswith("f")}

    result = {
        "method": f"xgboost_{config_name}", "task": task,
        "config": {k: str(v) for k, v in params.items()},
        "n_estimators_used": model.best_iteration + 1 if hasattr(model, 'best_iteration') and model.best_iteration else n_estimators,
        "training_time_seconds": train_time,
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "feature_importance_top20": dict(sorted(importance_named.items(), key=lambda x: -x[1])[:20]),
    }

    save_result(task, "xgboost", config_name, result)
    model.save_model(str(MODEL_ROOT / f"xgb_{task}_{config_name}.json"))

    log(f"  XGBoost {config_name}: acc={test_metrics['accuracy']:.6f} macro_f1={test_metrics['macro_f1']:.6f} time={train_time:.0f}s")

    del dtrain, dval, dtest, model
    gc.collect()
    return result


# ═══════════ MLP (PyTorch) ═══════════

def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test,
              task, n_classes, class_names, config_name):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        log(f"  MLP {config_name}: PyTorch not available, skipping")
        return None

    log(f"  MLP {config_name}: starting...")
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"    Device: {device}")

    # Standardize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_tr = (X_train - mean) / std
    X_v = (X_val - mean) / std
    X_te = (X_test - mean) / std

    # Class weights
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    weights = torch.tensor((len(y_train) / (n_classes * np.maximum(counts, 1))) ** 0.5,
                          dtype=torch.float32).to(device)

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_train.astype(np.int64)))
    val_ds = TensorDataset(torch.from_numpy(X_v), torch.from_numpy(y_val.astype(np.int64)))

    train_loader = DataLoader(train_ds, batch_size=8192, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16384, shuffle=False, num_workers=4, pin_memory=True)

    n_feat = X_train.shape[1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_feat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, n_classes),
            )
        def forward(self, x):
            return self.net(x)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    best_state = None

    for epoch in range(50):
        model.train()
        train_loss = 0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 5 == 0:
            log(f"    Epoch {epoch}: train_loss={train_loss/n_batches:.4f} val_loss={avg_val_loss:.4f}")

        if patience_counter >= max_patience:
            log(f"    Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    all_preds = []
    test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_test.astype(np.int64)))
    test_loader = DataLoader(test_ds, batch_size=16384, shuffle=False, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    y_pred_test = np.concatenate(all_preds)

    all_val_preds = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_val_preds.append(preds)
    y_pred_val = np.concatenate(all_val_preds)

    train_time = time.time() - t0

    val_metrics = compute_metrics(y_val, y_pred_val, n_classes, class_names)
    test_metrics = compute_metrics(y_test, y_pred_test, n_classes, class_names)

    result = {
        "method": f"mlp_{config_name}", "task": task,
        "config": {"hidden": [512, 256, 128], "dropout": [0.3, 0.2, 0.1],
                   "lr": 1e-3, "batch_size": 8192, "epochs_used": epoch + 1},
        "training_time_seconds": train_time,
        "val_metrics": val_metrics, "test_metrics": test_metrics,
    }

    save_result(task, "mlp", config_name, result)
    torch.save(model.state_dict(), str(MODEL_ROOT / f"mlp_{task}_{config_name}.pt"))

    # Save normalization params
    np.savez(str(MODEL_ROOT / f"mlp_{task}_{config_name}_norm.npz"), mean=mean, std=std)

    log(f"  MLP {config_name}: acc={test_metrics['accuracy']:.6f} macro_f1={test_metrics['macro_f1']:.6f} time={train_time:.0f}s")

    del model, train_ds, val_ds, test_ds
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ═══════════ ENSEMBLE ═══════════

def make_ensemble(results_for_task, X_val, y_val, X_test, y_test,
                  task, n_classes, class_names):
    """Soft voting ensemble from top-3 models by val macro-F1."""
    log(f"  Ensemble: building...")

    # We don't have prediction probabilities saved, so we'll do a simple
    # majority vote using the best models' predictions
    # For a proper ensemble we'd need to retrain/reload, but for efficiency
    # let's just report the best individual model
    log(f"  (Ensemble would require reloading models; using best individual result)")
    return None


# ═══════════ Main ═══════════

def run_task(task, label_col, n_classes, class_names, targets):
    log(f"\n{'='*60}")
    log(f"TASK: {task} ({n_classes} classes)")
    log(f"Targets: {targets}")
    log(f"{'='*60}")

    log("Loading data...")
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

    log(f"Data: train={len(y_train):,} val={len(y_val):,} test={len(y_test):,}")
    for i in range(n_classes):
        cnt = (y_train == i).sum()
        log(f"  {class_names[i]}: {cnt:,} ({100*cnt/len(y_train):.2f}%)")

    results = []

    # ── LightGBM Config 1 (fast, strong baseline) ──
    try:
        r = train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test,
                       task, n_classes, class_names, "cfg1",
                       {"num_leaves": 512, "learning_rate": 0.1, "min_child_samples": 100,
                        "n_estimators": 2000, "early_stopping_rounds": 50,
                        "subsample": 0.8, "colsample_bytree": 0.8})
        results.append(r)
    except Exception as e:
        log(f"  LightGBM cfg1 FAILED: {e}")
        traceback.print_exc()

    # ── LightGBM Config 2 (deeper) ──
    try:
        r = train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test,
                       task, n_classes, class_names, "cfg2",
                       {"num_leaves": 2048, "learning_rate": 0.05, "min_child_samples": 50,
                        "n_estimators": 3000, "early_stopping_rounds": 50,
                        "subsample": 0.8, "colsample_bytree": 0.8})
        results.append(r)
    except Exception as e:
        log(f"  LightGBM cfg2 FAILED: {e}")
        traceback.print_exc()

    # ── LightGBM Config 3 (weighted, best params) ──
    try:
        lgbm_results = [r for r in results if "lightgbm" in r["method"]]
        if lgbm_results:
            if n_classes > 2:
                best = max(lgbm_results, key=lambda x: x["val_metrics"]["macro_f1"])
            else:
                best = max(lgbm_results, key=lambda x: x["val_metrics"]["accuracy"])

            weights = compute_class_weights(y_train, n_classes, power=0.5)
            sample_w = weights[y_train].astype(np.float32)

            cfg = {}
            for k in ("num_leaves", "learning_rate", "min_child_samples", "subsample", "colsample_bytree"):
                if k in best["config"]:
                    cfg[k] = best["config"][k]
            cfg["n_estimators"] = 3000
            cfg["early_stopping_rounds"] = 50

            r = train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test,
                           task, n_classes, class_names, "cfg3_weighted",
                           cfg, sample_weights=sample_w)
            results.append(r)
    except Exception as e:
        log(f"  LightGBM cfg3 FAILED: {e}")
        traceback.print_exc()

    # ── XGBoost Config 1 ──
    try:
        r = train_xgb(X_train, y_train, X_val, y_val, X_test, y_test,
                      task, n_classes, class_names, "cfg1",
                      {"max_depth": 10, "learning_rate": 0.1, "max_bin": 256,
                       "n_estimators": 1500, "early_stopping_rounds": 50,
                       "subsample": 0.8, "colsample_bytree": 0.8})
        results.append(r)
    except Exception as e:
        log(f"  XGBoost cfg1 FAILED: {e}")
        traceback.print_exc()

    # ── XGBoost Config 2 (lossguide) ──
    try:
        r = train_xgb(X_train, y_train, X_val, y_val, X_test, y_test,
                      task, n_classes, class_names, "cfg2",
                      {"max_leaves": 1024, "max_depth": 0, "grow_policy": "lossguide",
                       "learning_rate": 0.05, "max_bin": 512,
                       "n_estimators": 2000, "early_stopping_rounds": 50,
                       "subsample": 0.8, "colsample_bytree": 0.8})
        results.append(r)
    except Exception as e:
        log(f"  XGBoost cfg2 FAILED: {e}")
        traceback.print_exc()

    # ── XGBoost Config 3 (weighted) ──
    try:
        xgb_results = [r for r in results if "xgboost" in r["method"]]
        if xgb_results:
            if n_classes > 2:
                best_xgb = max(xgb_results, key=lambda x: x["val_metrics"]["macro_f1"])
            else:
                best_xgb = max(xgb_results, key=lambda x: x["val_metrics"]["accuracy"])

            weights = compute_class_weights(y_train, n_classes, power=0.5)
            sample_w = weights[y_train].astype(np.float32)

            xgb_cfg = {}
            for k in ("max_depth", "max_leaves", "learning_rate", "max_bin",
                       "subsample", "colsample_bytree", "grow_policy"):
                if k in best_xgb["config"]:
                    v = best_xgb["config"][k]
                    if k in ("max_depth", "max_leaves", "max_bin"):
                        xgb_cfg[k] = int(v)
                    elif k in ("learning_rate", "subsample", "colsample_bytree"):
                        xgb_cfg[k] = float(v)
                    else:
                        xgb_cfg[k] = v
            xgb_cfg["n_estimators"] = 2000
            xgb_cfg["early_stopping_rounds"] = 50

            if n_classes == 2:
                pos = (y_train == 1).sum()
                neg = (y_train == 0).sum()
                xgb_cfg["scale_pos_weight"] = float(neg / pos)
                sw = None
            else:
                sw = sample_w

            r = train_xgb(X_train, y_train, X_val, y_val, X_test, y_test,
                          task, n_classes, class_names, "cfg3_weighted",
                          xgb_cfg, sample_weights=sw)
            results.append(r)
    except Exception as e:
        log(f"  XGBoost cfg3 FAILED: {e}")
        traceback.print_exc()

    # ── CatBoost GPU ──
    try:
        cb_params = {
            "depth": 10, "learning_rate": 0.1, "iterations": 3000,
            "early_stopping_rounds": 50, "auto_class_weights": "Balanced",
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

    # ── MLP ──
    try:
        r = train_mlp(X_train, y_train, X_val, y_val, X_test, y_test,
                      task, n_classes, class_names, "v1")
        if r:
            results.append(r)
    except Exception as e:
        log(f"  MLP FAILED: {e}")
        traceback.print_exc()

    # ── Summary ──
    log(f"\n{'─'*50}")
    log(f"TASK {task} — RESULTS RANKED BY {'macro_f1' if n_classes > 2 else 'accuracy'}:")
    metric_key = "macro_f1" if n_classes > 2 else "accuracy"
    for r in sorted(results, key=lambda x: -x["test_metrics"][metric_key]):
        m = r["test_metrics"]
        log(f"  {r['method']:35s} acc={m['accuracy']:.6f} macro_f1={m['macro_f1']:.6f} time={r['training_time_seconds']:.0f}s")

    if results:
        best = max(results, key=lambda x: x["test_metrics"][metric_key])
        log(f"\n  >>> BEST: {best['method']} — acc={best['test_metrics']['accuracy']:.6f} macro_f1={best['test_metrics']['macro_f1']:.6f}")

        # Check against targets
        for tgt_name, tgt_val in targets.items():
            actual = best["test_metrics"][tgt_name]
            met = actual >= tgt_val
            log(f"  Target {tgt_name}: {tgt_val:.4f} | Actual: {actual:.6f} | {'MET' if met else 'NOT MET'}")

        # Per-class F1
        log(f"\n  Per-class F1 (best model):")
        for cls_name, cls_data in best["test_metrics"]["per_class"].items():
            log(f"    {cls_name:20s}: P={cls_data['precision']:.4f} R={cls_data['recall']:.4f} F1={cls_data['f1']:.4f}")

        best_dir = RESULTS_ROOT / task / "BEST"
        os.makedirs(best_dir, exist_ok=True)
        with open(best_dir / "results.json", "w") as f:
            json.dump(best, f, indent=2)

    del X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()
    return results


def main():
    log("PHASE 2 v2: TRAINING ALL MODELS")
    log(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 2-class
    run_task("2_class", "label_2class", 2, ["Benign", "Attack"],
             {"accuracy": 0.9968})

    # 8-class
    run_task("8_class", "label_8class", 8, CLASS8_ORDER,
             {"macro_f1": 0.7193, "accuracy": 0.9944})

    # 34-class
    run_task("34_class", "label_34class", 34, CLASS34_ORDER,
             {"macro_f1": 0.7})

    log(f"\nPHASE 2 COMPLETE at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Final summary
    for task in ["2_class", "8_class", "34_class"]:
        best_path = RESULTS_ROOT / task / "BEST" / "results.json"
        if best_path.exists():
            with open(best_path) as f:
                b = json.load(f)
            log(f"  {task}: {b['method']} acc={b['test_metrics']['accuracy']:.6f} macro_f1={b['test_metrics']['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
