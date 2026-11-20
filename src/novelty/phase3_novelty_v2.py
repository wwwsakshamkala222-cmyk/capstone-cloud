#!/usr/bin/env python3
"""Phase 3 v2: Optimized novelty experiments.
Vectorized twin detection, GPU-accelerated models.
"""
import os, sys, json, time, glob, gc, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_auc_score)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CACHE_ROOT = Path("/dev/shm/saksham/ids_cache")
IDS_ROOT = Path("/dev/shm/saksham/ids")
NOVELTY_ROOT = IDS_ROOT / "novelty"
DATA_ROOT = Path("/dev/shm/saksham/ciciot2023/CSV")
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


def load_split(split_name):
    parts = sorted(glob.glob(str(CACHE_ROOT / split_name / "*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def compute_metrics(y_true, y_pred, n_classes, class_names=None):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = {}
    f = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    for i in range(n_classes):
        name = class_names[i] if class_names else str(i)
        per_class[name] = float(f[i])
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    return {"accuracy": float(acc), "macro_f1": float(macro_f1),
            "per_class_f1": per_class, "confusion_matrix": cm.tolist()}


def fast_row_hash(X):
    """Hash each row of a float32 array to bytes using views."""
    return pd.util.hash_array(X.view(np.uint8).reshape(len(X), -1).sum(axis=1).astype(np.float64))


def train_quick_xgb(X_train, y_train, X_test, n_classes, n_rounds=500):
    """Train quick XGBoost model and return predictions."""
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    if n_classes == 2:
        params = {"max_depth": 10, "learning_rate": 0.1, "objective": "binary:logistic",
                  "eval_metric": "logloss", "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0}
        model = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)
        return (model.predict(dtest) > 0.5).astype(int)
    else:
        params = {"max_depth": 10, "learning_rate": 0.1, "objective": "multi:softmax",
                  "num_class": n_classes, "eval_metric": "mlogloss",
                  "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0}
        model = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)
        return model.predict(dtest).astype(int)


# ═══════════ N1: TWIN ANALYSIS (vectorized) ═══════════

def n1_twin_analysis():
    log("N1: TWIN ANALYSIS — starting")
    out_dir = NOVELTY_ROOT / "N1_twin_analysis"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    log("  Loading data...")
    train_df = load_split("train")
    test_df = load_split("test")

    X_train = train_df[FEATURES].values
    X_test = test_df[FEATURES].values
    y_test_8 = test_df["label_8class"].values

    log(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Vectorized twin detection using row hashing
    log("  Computing train row hashes (vectorized)...")
    # Convert to bytes view for hashing
    train_bytes = X_train.astype(np.float32).tobytes()
    row_size = X_train.shape[1] * 4  # float32 = 4 bytes

    # Use pandas hashing which is much faster than Python loops
    train_hash = pd.util.hash_pandas_object(
        pd.DataFrame(X_train.astype(np.float32), columns=FEATURES),
        index=False
    ).values

    log("  Computing test row hashes...")
    test_hash = pd.util.hash_pandas_object(
        pd.DataFrame(X_test.astype(np.float32), columns=FEATURES),
        index=False
    ).values

    log("  Finding twins...")
    train_hash_set = set(train_hash)
    is_twin = np.array([h in train_hash_set for h in test_hash])

    del train_hash_set, train_hash, test_hash
    gc.collect()

    twin_pct = is_twin.mean() * 100
    log(f"  Overall twin rate: {twin_pct:.2f}% ({is_twin.sum():,}/{len(is_twin):,})")

    # Per-class twin rates
    per_class_twin = {}
    for i, name in enumerate(CLASS8_ORDER):
        mask = y_test_8 == i
        if mask.sum() > 0:
            pct = is_twin[mask].mean() * 100
            per_class_twin[name] = float(pct)
            log(f"    {name}: {pct:.2f}%")

    # Evaluate model on twin vs non-twin
    log("  Training quick XGBoost for twin analysis...")
    y_train_8 = train_df["label_8class"].values.astype(np.int32)
    X_train_f32 = X_train.astype(np.float32)
    X_test_f32 = X_test.astype(np.float32)

    y_pred = train_quick_xgb(X_train_f32, y_train_8, X_test_f32, 8)

    full_metrics = compute_metrics(y_test_8, y_pred, 8, CLASS8_ORDER)
    twin_metrics = compute_metrics(y_test_8[is_twin], y_pred[is_twin], 8, CLASS8_ORDER) if is_twin.sum() > 0 else None
    nontwin_mask = ~is_twin
    nontwin_metrics = compute_metrics(y_test_8[nontwin_mask], y_pred[nontwin_mask], 8, CLASS8_ORDER) if nontwin_mask.sum() > 0 else None

    log(f"  Full:     macro_f1={full_metrics['macro_f1']:.4f} acc={full_metrics['accuracy']:.4f}")
    if twin_metrics:
        log(f"  Twin:     macro_f1={twin_metrics['macro_f1']:.4f} acc={twin_metrics['accuracy']:.4f}")
    if nontwin_metrics:
        log(f"  Non-twin: macro_f1={nontwin_metrics['macro_f1']:.4f} acc={nontwin_metrics['accuracy']:.4f}")

    result = {
        "overall_twin_pct": float(twin_pct),
        "per_class_twin_pct": per_class_twin,
        "n_test": int(len(X_test)),
        "n_twins": int(is_twin.sum()),
        "n_non_twins": int(nontwin_mask.sum()),
        "full_test_metrics": full_metrics,
        "twin_subset_metrics": twin_metrics,
        "nontwin_subset_metrics": nontwin_metrics,
        "twin_inflation_macro_f1": float(full_metrics["macro_f1"] - nontwin_metrics["macro_f1"]) if nontwin_metrics else None,
        "twin_inflation_accuracy": float(full_metrics["accuracy"] - nontwin_metrics["accuracy"]) if nontwin_metrics else None,
        "elapsed_seconds": time.time() - t0,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    np.save(out_dir / "twin_mask.npy", is_twin)

    log(f"N1 COMPLETE in {time.time()-t0:.0f}s")
    del train_df, test_df, X_train, X_test
    gc.collect()
    return result


# ═══════════ N3: DoS vs DDoS DEEP DIVE ═══════════

def n3_dos_vs_ddos():
    log("N3: DoS vs DDoS DEEP DIVE — starting")
    out_dir = NOVELTY_ROOT / "N3_dos_vs_ddos"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    import xgboost as xgb

    train_df = load_split("train")
    test_df = load_split("test")

    dos_id = CLASS_MAPS["class8_to_id"]["DoS"]
    ddos_id = CLASS_MAPS["class8_to_id"]["DDoS"]

    train_mask = np.isin(train_df["label_8class"].values, [dos_id, ddos_id])
    test_mask = np.isin(test_df["label_8class"].values, [dos_id, ddos_id])

    X_train = train_df.loc[train_mask, FEATURES].values.astype(np.float32)
    y_train = (train_df.loc[train_mask, "label_8class"].values == ddos_id).astype(np.int32)
    X_test = test_df.loc[test_mask, FEATURES].values.astype(np.float32)
    y_test = (test_df.loc[test_mask, "label_8class"].values == ddos_id).astype(np.int32)

    log(f"  Train: {len(X_train):,} (DoS={( y_train==0).sum():,}, DDoS={(y_train==1).sum():,})")
    log(f"  Test:  {len(X_test):,}")

    del train_df, test_df
    gc.collect()

    # Count ambiguous rows
    log("  Counting ambiguous rows...")
    SAMPLE = min(5000000, len(X_train))
    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(len(X_train), SAMPLE, replace=False)

    # Use pandas hashing for speed
    sample_hashes = pd.util.hash_pandas_object(
        pd.DataFrame(X_train[sample_idx], columns=FEATURES), index=False
    ).values
    sample_labels = y_train[sample_idx]

    hash_label_df = pd.DataFrame({"hash": sample_hashes, "label": sample_labels})
    ambig = hash_label_df.groupby("hash")["label"].nunique()
    ambiguous_count = (ambig > 1).sum()
    ambiguous_pct = ambiguous_count / len(ambig) * 100
    log(f"  Ambiguous unique vectors: {ambiguous_count:,} ({ambiguous_pct:.2f}%)")

    del hash_label_df, ambig
    gc.collect()

    # XGBoost paper protocol
    log("  XGBoost binary (paper protocol)...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {"max_depth": 10, "learning_rate": 0.1, "max_bin": 256,
              "objective": "binary:logistic", "eval_metric": "logloss",
              "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0,
              "subsample": 0.8, "colsample_bytree": 0.8}

    model = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
    y_prob = model.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    macro_f1_val = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred).tolist()

    importance = model.get_score(importance_type="gain")
    importance_named = {FEATURES[int(k[1:])]: v for k, v in importance.items() if k.startswith("f")}
    top_features = dict(sorted(importance_named.items(), key=lambda x: -x[1])[:20])

    log(f"  Paper: acc={acc:.4f} AUC={auc:.4f} mF1={macro_f1_val:.4f}")

    paper_result = {"accuracy": float(acc), "auc": float(auc), "macro_f1": float(macro_f1_val),
                    "confusion_matrix": cm, "top_features": top_features}

    del dtrain, dtest, model
    gc.collect()

    # File-grouped split
    log("  Loading with file info for file-grouped split...")
    all_data = []
    dos_ddos_folders = [f for f in sorted(os.listdir(DATA_ROOT))
                        if (DATA_ROOT / f).is_dir() and
                        CLASS_MAPS["folder_to_8class"].get(f, "") in ("DoS", "DDoS")]

    for folder in dos_ddos_folders:
        csvs = sorted(glob.glob(str(DATA_ROOT / folder / "*.csv")))
        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                feat_cols = [c for c in df.columns if c not in ('label', 'Label')][:39]
                df = df[feat_cols].copy()
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                is_ddos = 1 if CLASS_MAPS["folder_to_8class"][folder] == "DDoS" else 0
                df["label"] = np.int8(is_ddos)
                df["source_file"] = os.path.basename(csv_path)
                all_data.append(df)
            except:
                pass

    full_df = pd.concat(all_data, ignore_index=True)
    del all_data
    gc.collect()

    unique_files = np.unique(full_df["source_file"].values)
    rng2 = np.random.RandomState(SEED)
    rng2.shuffle(unique_files)
    split_point = int(0.8 * len(unique_files))
    train_files = set(unique_files[:split_point])

    file_train = full_df["source_file"].isin(train_files)
    X_tr = full_df.loc[file_train, FEATURES].values.astype(np.float32)
    y_tr = full_df.loc[file_train, "label"].values.astype(np.int32)
    X_te = full_df.loc[~file_train, FEATURES].values.astype(np.float32)
    y_te = full_df.loc[~file_train, "label"].values.astype(np.int32)

    log(f"  File-grouped: train={len(X_tr):,} test={len(X_te):,}")

    dtrain2 = xgb.DMatrix(X_tr, label=y_tr)
    dtest2 = xgb.DMatrix(X_te, label=y_te)

    model2 = xgb.train(params, dtrain2, num_boost_round=500, verbose_eval=False)
    y_prob2 = model2.predict(dtest2)
    y_pred2 = (y_prob2 > 0.5).astype(int)

    acc2 = accuracy_score(y_te, y_pred2)
    auc2 = roc_auc_score(y_te, y_prob2) if len(np.unique(y_te)) > 1 else 0
    macro_f1_2 = f1_score(y_te, y_pred2, average="macro")

    log(f"  File-grouped: acc={acc2:.4f} AUC={auc2:.4f} mF1={macro_f1_2:.4f}")

    file_result = {"accuracy": float(acc2), "auc": float(auc2), "macro_f1": float(macro_f1_2),
                   "confusion_matrix": confusion_matrix(y_te, y_pred2).tolist()}

    result = {
        "paper_protocol": paper_result,
        "file_grouped": file_result,
        "ambiguous_rows_pct": float(ambiguous_pct),
        "memorization_signal": "YES" if (acc - acc2) > 0.05 else "WEAK" if (acc - acc2) > 0.01 else "NO",
        "accuracy_drop": float(acc - acc2),
        "elapsed_seconds": time.time() - t0,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    log(f"N3 COMPLETE in {time.time()-t0:.0f}s")
    del full_df
    gc.collect()
    return result


# ═══════════ N4: FINGERPRINT ABLATION ═══════════

def n4_fingerprint_ablation():
    log("N4: FINGERPRINT ABLATION — starting")
    out_dir = NOVELTY_ROOT / "N4_fingerprint_ablation"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    import xgboost as xgb

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    y_train = train_df["label_2class"].values.astype(np.int32)
    y_val = val_df["label_2class"].values.astype(np.int32)
    y_test = test_df["label_2class"].values.astype(np.int32)

    params = {"max_depth": 10, "learning_rate": 0.1, "max_bin": 256,
              "objective": "binary:logistic", "eval_metric": "logloss",
              "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0,
              "subsample": 0.8, "colsample_bytree": 0.8}

    # Baseline
    log("  Baseline (all 39 features)...")
    X_tr = train_df[FEATURES].values.astype(np.float32)
    X_va = val_df[FEATURES].values.astype(np.float32)
    X_te = test_df[FEATURES].values.astype(np.float32)

    dtrain = xgb.DMatrix(X_tr, label=y_train)
    dval = xgb.DMatrix(X_va, label=y_val)
    dtest = xgb.DMatrix(X_te, label=y_test)

    model = xgb.train(params, dtrain, num_boost_round=500,
                       evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)

    y_pred = (model.predict(dtest) > 0.5).astype(int)
    baseline_acc = accuracy_score(y_test, y_pred)
    baseline_f1 = f1_score(y_test, y_pred, average="macro")

    importance = model.get_score(importance_type="gain")
    importance_named = {FEATURES[int(k[1:])]: v for k, v in importance.items() if k.startswith("f")}
    sorted_features = sorted(importance_named.items(), key=lambda x: -x[1])
    top10 = [f[0] for f in sorted_features[:10]]

    log(f"  Baseline: acc={baseline_acc:.6f} mF1={baseline_f1:.6f}")
    log(f"  Top 10: {top10}")

    del dtrain, dval, dtest, model, X_tr, X_va, X_te
    gc.collect()

    # Without TTL + IAT
    remove_feats = ["Time_To_Live", "IAT"]
    feats_a1 = [f for f in FEATURES if f not in remove_feats]
    log(f"  Without {remove_feats} ({len(feats_a1)} features)...")

    X_tr = train_df[feats_a1].values.astype(np.float32)
    X_va = val_df[feats_a1].values.astype(np.float32)
    X_te = test_df[feats_a1].values.astype(np.float32)

    dtrain = xgb.DMatrix(X_tr, label=y_train)
    dval = xgb.DMatrix(X_va, label=y_val)
    dtest = xgb.DMatrix(X_te, label=y_test)

    model_a1 = xgb.train(params, dtrain, num_boost_round=500,
                          evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
    y_pred_a1 = (model_a1.predict(dtest) > 0.5).astype(int)
    acc_a1 = accuracy_score(y_test, y_pred_a1)
    f1_a1 = f1_score(y_test, y_pred_a1, average="macro")
    log(f"  Without TTL+IAT: acc={acc_a1:.6f} (drop={baseline_acc-acc_a1:.6f})")

    del dtrain, dval, dtest, model_a1, X_tr, X_va, X_te
    gc.collect()

    # Without top 10
    feats_a2 = [f for f in FEATURES if f not in top10]
    log(f"  Without top-10 ({len(feats_a2)} features)...")

    X_tr = train_df[feats_a2].values.astype(np.float32)
    X_va = val_df[feats_a2].values.astype(np.float32)
    X_te = test_df[feats_a2].values.astype(np.float32)

    dtrain = xgb.DMatrix(X_tr, label=y_train)
    dval = xgb.DMatrix(X_va, label=y_val)
    dtest = xgb.DMatrix(X_te, label=y_test)

    model_a2 = xgb.train(params, dtrain, num_boost_round=500,
                          evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
    y_pred_a2 = (model_a2.predict(dtest) > 0.5).astype(int)
    acc_a2 = accuracy_score(y_test, y_pred_a2)
    f1_a2 = f1_score(y_test, y_pred_a2, average="macro")
    log(f"  Without top-10: acc={acc_a2:.6f} (drop={baseline_acc-acc_a2:.6f})")

    result = {
        "baseline": {"accuracy": float(baseline_acc), "macro_f1": float(baseline_f1), "n_features": 39},
        "without_ttl_iat": {"accuracy": float(acc_a1), "macro_f1": float(f1_a1),
                            "removed": remove_feats, "n_features": len(feats_a1),
                            "accuracy_drop": float(baseline_acc - acc_a1)},
        "without_top10": {"accuracy": float(acc_a2), "macro_f1": float(f1_a2),
                          "removed": top10, "n_features": len(feats_a2),
                          "accuracy_drop": float(baseline_acc - acc_a2)},
        "feature_importance_ranking": [{"feature": f, "gain": g} for f, g in sorted_features],
        "elapsed_seconds": time.time() - t0,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    log(f"N4 COMPLETE in {time.time()-t0:.0f}s")
    del train_df, val_df, test_df
    gc.collect()
    return result


# ═══════════ N5: WINDOW-SIZE ARTIFACT ═══════════

def n5_window_size():
    log("N5: WINDOW-SIZE ARTIFACT — starting")
    out_dir = NOVELTY_ROOT / "N5_window_size"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    import xgboost as xgb

    window_100 = {"DDoS", "DoS", "Mirai"}

    train_df = load_split("train")
    test_df = load_split("test")

    y_train_8 = train_df["label_8class"].values.astype(np.int32)
    y_test_8 = test_df["label_8class"].values.astype(np.int32)

    window_train = np.array([1 if CLASS8_ORDER[y] in window_100 else 0 for y in y_train_8], dtype=np.int32)
    window_test = np.array([1 if CLASS8_ORDER[y] in window_100 else 0 for y in y_test_8], dtype=np.int32)

    X_train = train_df[FEATURES].values.astype(np.float32)
    X_test = test_df[FEATURES].values.astype(np.float32)

    log(f"  W100 train: {(window_train==1).sum():,}, W10 train: {(window_train==0).sum():,}")

    # Window classifier
    log("  Training window classifier...")
    dtrain = xgb.DMatrix(X_train, label=window_train)
    dtest = xgb.DMatrix(X_test, label=window_test)

    params_w = {"max_depth": 8, "learning_rate": 0.1, "objective": "binary:logistic",
                "eval_metric": "logloss", "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0}

    model = xgb.train(params_w, dtrain, num_boost_round=300, verbose_eval=False)
    y_prob = model.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)

    window_acc = accuracy_score(window_test, y_pred)
    window_f1 = f1_score(window_test, y_pred, average="macro")
    log(f"  Window classifier: acc={window_acc:.4f} mF1={window_f1:.4f}")

    importance = model.get_score(importance_type="gain")
    importance_named = {FEATURES[int(k[1:])]: v for k, v in importance.items() if k.startswith("f")}
    top_window_feats = [f[0] for f, _ in sorted(importance_named.items(), key=lambda x: -x[1])[:10]]
    log(f"  Top window features: {top_window_feats}")

    del dtrain, dtest, model
    gc.collect()

    # 8-class baseline
    log("  8-class baseline...")
    params8 = {"max_depth": 10, "learning_rate": 0.1, "objective": "multi:softmax",
               "num_class": 8, "eval_metric": "mlogloss",
               "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0}

    dtrain = xgb.DMatrix(X_train, label=y_train_8)
    dtest = xgb.DMatrix(X_test, label=y_test_8)
    model_full = xgb.train(params8, dtrain, num_boost_round=500, verbose_eval=False)
    y_pred_full = model_full.predict(dtest).astype(int)
    metrics_full = compute_metrics(y_test_8, y_pred_full, 8, CLASS8_ORDER)
    log(f"  Baseline: mF1={metrics_full['macro_f1']:.4f}")

    del dtrain, dtest, model_full
    gc.collect()

    # 8-class without window features
    log("  8-class without window features...")
    feats_nw = [f for f in FEATURES if f not in top_window_feats]
    X_tr_nw = train_df[feats_nw].values.astype(np.float32)
    X_te_nw = test_df[feats_nw].values.astype(np.float32)

    dtrain = xgb.DMatrix(X_tr_nw, label=y_train_8)
    dtest = xgb.DMatrix(X_te_nw, label=y_test_8)
    model_nw = xgb.train(params8, dtrain, num_boost_round=500, verbose_eval=False)
    y_pred_nw = model_nw.predict(dtest).astype(int)
    metrics_nw = compute_metrics(y_test_8, y_pred_nw, 8, CLASS8_ORDER)
    log(f"  No-window: mF1={metrics_nw['macro_f1']:.4f}")

    result = {
        "window_classifier_accuracy": float(window_acc),
        "window_classifier_f1": float(window_f1),
        "window_leaks_class": window_acc > 0.9,
        "top_window_features": top_window_feats,
        "8class_baseline": {"macro_f1": metrics_full["macro_f1"], "accuracy": metrics_full["accuracy"],
                            "per_class_f1": metrics_full["per_class_f1"]},
        "8class_no_window_features": {"macro_f1": metrics_nw["macro_f1"], "accuracy": metrics_nw["accuracy"],
                                      "per_class_f1": metrics_nw["per_class_f1"]},
        "macro_f1_change": float(metrics_nw["macro_f1"] - metrics_full["macro_f1"]),
        "elapsed_seconds": time.time() - t0,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    log(f"N5 COMPLETE in {time.time()-t0:.0f}s")
    del train_df, test_df
    gc.collect()
    return result


# ═══════════ N6: CLASS-WEIGHT SENSITIVITY ═══════════

def n6_class_weight_sweep():
    log("N6: CLASS-WEIGHT SENSITIVITY — starting")
    out_dir = NOVELTY_ROOT / "N6_class_weight_sweep"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    import xgboost as xgb

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train = train_df[FEATURES].values.astype(np.float32)
    y_train = train_df["label_8class"].values.astype(np.int32)
    X_val = val_df[FEATURES].values.astype(np.float32)
    y_val = val_df["label_8class"].values.astype(np.int32)
    X_test = test_df[FEATURES].values.astype(np.float32)
    y_test = test_df["label_8class"].values.astype(np.int32)

    del train_df, val_df, test_df
    gc.collect()

    n_classes = 8
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    N = len(y_train)

    results_list = []

    for p in [0, 0.25, 0.5, 0.75, 1.0]:
        log(f"  p={p}...")
        if p == 0:
            sample_w = None
        else:
            weights = (N / (n_classes * np.maximum(counts, 1))) ** p
            sample_w = weights[y_train].astype(np.float32)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_w)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {"max_depth": 10, "learning_rate": 0.1, "objective": "multi:softmax",
                  "num_class": 8, "eval_metric": "mlogloss",
                  "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0,
                  "subsample": 0.8, "colsample_bytree": 0.8}

        model = xgb.train(params, dtrain, num_boost_round=500,
                           evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
        y_pred = model.predict(dtest).astype(int)
        metrics = compute_metrics(y_test, y_pred, n_classes, CLASS8_ORDER)

        log(f"    acc={metrics['accuracy']:.4f} mF1={metrics['macro_f1']:.4f}")

        results_list.append({
            "power": float(p),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_f1": metrics["per_class_f1"],
        })

        del dtrain, dval, dtest, model
        gc.collect()

    result = {
        "sweep": results_list,
        "class_counts": {CLASS8_ORDER[i]: int(counts[i]) for i in range(n_classes)},
        "elapsed_seconds": time.time() - t0,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    log(f"N6 COMPLETE in {time.time()-t0:.0f}s")
    return result


# ═══════════ N2: PROTOCOL LADDER ═══════════

def n2_protocol_ladder():
    log("N2: PROTOCOL LADDER — starting")
    out_dir = NOVELTY_ROOT / "N2_protocol_ladder"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    import xgboost as xgb

    # (a) paper protocol — use Phase 2 best
    best_path = IDS_ROOT / "results" / "8_class" / "BEST" / "results.json"
    protocol_a = None
    if best_path.exists():
        with open(best_path) as f:
            protocol_a = json.load(f)["test_metrics"]
        # Convert per_class format
        if "per_class" in protocol_a and "per_class_f1" not in protocol_a:
            protocol_a["per_class_f1"] = {k: v["f1"] for k, v in protocol_a["per_class"].items()}
        log(f"  (a) Paper protocol: mF1={protocol_a.get('macro_f1', 'N/A')}")

    # Load full data with file tracking
    log("  Loading all data with file tracking...")
    all_data = []
    folders = sorted([d for d in os.listdir(DATA_ROOT) if (DATA_ROOT / d).is_dir()])

    for folder in folders:
        csvs = sorted(glob.glob(str(DATA_ROOT / folder / "*.csv")))
        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                feat_cols = [c for c in df.columns if c not in ('label', 'Label')][:39]
                df = df[feat_cols].copy()
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                label_8 = CLASS_MAPS["class8_to_id"][CLASS_MAPS["folder_to_8class"][folder]]
                df["label_8class"] = np.int8(label_8)
                df["source_file"] = os.path.basename(csv_path)
                all_data.append(df)
            except Exception as e:
                pass

    full_df = pd.concat(all_data, ignore_index=True)
    del all_data
    gc.collect()
    log(f"  Full dataset: {len(full_df):,} rows")

    X_all = full_df[FEATURES].values.astype(np.float32)
    y_all = full_df["label_8class"].values.astype(np.int32)

    params = {"max_depth": 10, "learning_rate": 0.1, "objective": "multi:softmax",
              "num_class": 8, "eval_metric": "mlogloss",
              "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0,
              "subsample": 0.8, "colsample_bytree": 0.8}

    # (b) Deduplicated
    log("  (b) Dedup protocol...")
    row_hashes = pd.util.hash_pandas_object(
        pd.DataFrame(np.column_stack([X_all, y_all.reshape(-1,1)]),
                      columns=FEATURES + ["label"]),
        index=False
    ).values
    _, unique_idx = np.unique(row_hashes, return_index=True)
    dedup_mask = np.zeros(len(X_all), dtype=bool)
    dedup_mask[unique_idx] = True
    n_dedup = dedup_mask.sum()
    log(f"    After dedup: {n_dedup:,} (removed {len(X_all)-n_dedup:,})")

    X_dedup = X_all[dedup_mask]
    y_dedup = y_all[dedup_mask]
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X_dedup))
    split = int(0.8 * len(idx))

    dtrain = xgb.DMatrix(X_dedup[idx[:split]], label=y_dedup[idx[:split]])
    dtest = xgb.DMatrix(X_dedup[idx[split:]], label=y_dedup[idx[split:]])
    model_b = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
    y_pred_b = model_b.predict(dtest).astype(int)
    protocol_b = compute_metrics(y_dedup[idx[split:]], y_pred_b, 8, CLASS8_ORDER)
    log(f"    (b) mF1={protocol_b['macro_f1']:.4f} acc={protocol_b['accuracy']:.4f}")
    del dtrain, dtest, model_b, X_dedup, y_dedup
    gc.collect()

    # (c) File-grouped
    log("  (c) File-grouped...")
    source_files = full_df["source_file"].values
    unique_files = np.unique(source_files)
    rng2 = np.random.RandomState(SEED)
    rng2.shuffle(unique_files)
    split_point = int(0.8 * len(unique_files))
    train_files = set(unique_files[:split_point])
    file_train_mask = np.array([f in train_files for f in source_files])

    dtrain = xgb.DMatrix(X_all[file_train_mask], label=y_all[file_train_mask])
    dtest_c = xgb.DMatrix(X_all[~file_train_mask], label=y_all[~file_train_mask])
    model_c = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
    y_pred_c = model_c.predict(dtest_c).astype(int)
    protocol_c = compute_metrics(y_all[~file_train_mask], y_pred_c, 8, CLASS8_ORDER)
    log(f"    (c) mF1={protocol_c['macro_f1']:.4f} acc={protocol_c['accuracy']:.4f}")

    # (d) File-grouped + twin-free
    log("  (d) File-grouped + twin-free...")
    X_train_c = X_all[file_train_mask]
    X_test_c = X_all[~file_train_mask]
    y_test_c = y_all[~file_train_mask]

    train_hashes = pd.util.hash_pandas_object(
        pd.DataFrame(X_train_c, columns=FEATURES), index=False
    ).values
    test_hashes = pd.util.hash_pandas_object(
        pd.DataFrame(X_test_c, columns=FEATURES), index=False
    ).values
    train_hash_set = set(train_hashes)
    twin_free = np.array([h not in train_hash_set for h in test_hashes])
    n_free = twin_free.sum()
    log(f"    Twin-free test: {n_free:,}/{len(X_test_c):,}")

    protocol_d = None
    if n_free > 100:
        dtest_d = xgb.DMatrix(X_test_c[twin_free])
        y_pred_d = model_c.predict(dtest_d).astype(int)
        protocol_d = compute_metrics(y_test_c[twin_free], y_pred_d, 8, CLASS8_ORDER)
        log(f"    (d) mF1={protocol_d['macro_f1']:.4f} acc={protocol_d['accuracy']:.4f}")

    del model_c, dtrain
    gc.collect()

    result = {
        "protocol_a_paper": protocol_a,
        "protocol_b_dedup": protocol_b,
        "protocol_c_file_grouped": protocol_c,
        "protocol_d_file_grouped_twin_free": protocol_d,
        "dedup_removed_pct": float(100*(1-n_dedup/len(X_all))),
        "file_grouped_twin_free_pct": float(100*n_free/len(X_test_c)) if len(X_test_c) > 0 else 0,
        "elapsed_seconds": time.time() - t0,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    log(f"N2 COMPLETE in {time.time()-t0:.0f}s")
    del full_df, X_all, y_all
    gc.collect()
    return result


# ═══════════ MAIN ═══════════

def main():
    log("PHASE 3 v2: NOVELTY EXPERIMENTS (GPU)")
    log(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    experiments = [
        ("N1", n1_twin_analysis),
        ("N3", n3_dos_vs_ddos),
        ("N4", n4_fingerprint_ablation),
        ("N5", n5_window_size),
        ("N6", n6_class_weight_sweep),
        ("N2", n2_protocol_ladder),
    ]

    for name, func in experiments:
        try:
            func()
        except Exception as e:
            log(f"{name} FAILED: {e}")
            traceback.print_exc()

    log(f"\nPHASE 3 COMPLETE at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
