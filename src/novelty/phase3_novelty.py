#!/usr/bin/env python3
"""Phase 3: Novelty experiments.
N1: Twin analysis, N2: Protocol ladder, N3: DoS vs DDoS deep dive,
N4: Fingerprint ablation, N5: Window-size artifact, N6: Class-weight sensitivity.
"""
import os, sys, json, time, glob, gc, hashlib, struct
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "16"

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
    """Load parquet split, return DataFrame with features + labels."""
    parts = sorted(glob.glob(str(CACHE_ROOT / split_name / "*.parquet")))
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    return df


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


def row_hash(row_array):
    """Hash a float32 row to uint64 for twin detection."""
    return hashlib.md5(row_array.tobytes()).digest()[:8]


# ═══════════════════════ N1: TWIN ANALYSIS ═══════════════════════

def n1_twin_analysis():
    log("N1: TWIN ANALYSIS — starting")
    out_dir = NOVELTY_ROOT / "N1_twin_analysis"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    log("  Loading train and test data...")
    train_df = load_split("train")
    test_df = load_split("test")

    X_train = train_df[FEATURES].values.astype(np.float32)
    X_test = test_df[FEATURES].values.astype(np.float32)

    log("  Hashing train rows...")
    train_hashes = set()
    CHUNK = 500000
    for i in range(0, len(X_train), CHUNK):
        chunk = X_train[i:i+CHUNK]
        for j in range(len(chunk)):
            train_hashes.add(chunk[j].tobytes())
        if (i // CHUNK) % 10 == 0:
            log(f"    Hashed {i+len(chunk):,}/{len(X_train):,} train rows")

    log(f"  Unique train hashes: {len(train_hashes):,}")

    log("  Checking test rows for twins...")
    is_twin = np.zeros(len(X_test), dtype=bool)
    for i in range(0, len(X_test), CHUNK):
        chunk = X_test[i:i+CHUNK]
        for j in range(len(chunk)):
            if chunk[j].tobytes() in train_hashes:
                is_twin[i+j] = True
        if (i // CHUNK) % 10 == 0:
            log(f"    Checked {i+len(chunk):,}/{len(X_test):,} test rows")

    del train_hashes, X_train
    gc.collect()

    twin_pct = is_twin.mean() * 100
    log(f"  Overall twin rate: {twin_pct:.2f}%")

    # Per-class twin rates (8-class)
    y_test_8 = test_df["label_8class"].values
    per_class_twin = {}
    for i, name in enumerate(CLASS8_ORDER):
        mask = y_test_8 == i
        if mask.sum() > 0:
            pct = is_twin[mask].mean() * 100
            per_class_twin[name] = float(pct)
            log(f"    {name}: {pct:.2f}% twins ({is_twin[mask].sum():,}/{mask.sum():,})")

    # Now evaluate best model on twin vs non-twin subsets
    # Load best 8-class model predictions (we'll retrain a quick XGBoost)
    log("  Training quick XGBoost for twin/non-twin evaluation...")
    import xgboost as xgb

    X_train_full = train_df[FEATURES].values.astype(np.float32)
    y_train_8 = train_df["label_8class"].values.astype(np.int32)

    dtrain = xgb.DMatrix(X_train_full, label=y_train_8)
    dtest = xgb.DMatrix(X_test, label=y_test_8)

    params = {
        "max_depth": 10, "learning_rate": 0.1, "max_bin": 256,
        "objective": "multi:softmax", "num_class": 8,
        "eval_metric": "mlogloss", "tree_method": "gpu_hist", "seed": SEED, "subsample": 0.8,
        "colsample_bytree": 0.8, "verbosity": 0,
    }
    model = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
    y_pred = model.predict(dtest).astype(int)

    # Metrics on full test
    full_metrics = compute_metrics(y_test_8, y_pred, 8, CLASS8_ORDER)

    # Metrics on twin subset
    twin_metrics = compute_metrics(y_test_8[is_twin], y_pred[is_twin], 8, CLASS8_ORDER) if is_twin.sum() > 0 else None

    # Metrics on non-twin subset
    non_twin_mask = ~is_twin
    nontwin_metrics = compute_metrics(y_test_8[non_twin_mask], y_pred[non_twin_mask], 8, CLASS8_ORDER) if non_twin_mask.sum() > 0 else None

    log(f"  Full test:     macro_f1={full_metrics['macro_f1']:.4f} acc={full_metrics['accuracy']:.4f}")
    if twin_metrics:
        log(f"  Twin subset:   macro_f1={twin_metrics['macro_f1']:.4f} acc={twin_metrics['accuracy']:.4f}")
    if nontwin_metrics:
        log(f"  Non-twin:      macro_f1={nontwin_metrics['macro_f1']:.4f} acc={nontwin_metrics['accuracy']:.4f}")

    result = {
        "overall_twin_pct": float(twin_pct),
        "per_class_twin_pct": per_class_twin,
        "n_test": int(len(X_test)),
        "n_twins": int(is_twin.sum()),
        "n_non_twins": int(non_twin_mask.sum()),
        "full_test_metrics": full_metrics,
        "twin_subset_metrics": twin_metrics,
        "nontwin_subset_metrics": nontwin_metrics,
        "twin_inflation_macro_f1": float(full_metrics["macro_f1"] - nontwin_metrics["macro_f1"]) if nontwin_metrics else None,
        "twin_inflation_accuracy": float(full_metrics["accuracy"] - nontwin_metrics["accuracy"]) if nontwin_metrics else None,
        "elapsed_seconds": time.time() - t0,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    # Save twin mask for later use
    np.save(out_dir / "twin_mask.npy", is_twin)

    log(f"N1 COMPLETE in {time.time()-t0:.0f}s")
    del train_df, test_df, X_test, model, dtrain, dtest
    gc.collect()
    return result


# ═══════════════════════ N2: PROTOCOL LADDER ═══════════════════════

def n2_protocol_ladder():
    log("N2: PROTOCOL LADDER — starting")
    out_dir = NOVELTY_ROOT / "N2_protocol_ladder"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # Protocol (a): paper protocol (shuffled, no dedup) — already done, use Phase 2 results
    log("  (a) Paper protocol: using Phase 2 results")
    best_path = IDS_ROOT / "results" / "8_class" / "BEST" / "results.json"
    if best_path.exists():
        with open(best_path) as f:
            paper_result = json.load(f)
        protocol_a = paper_result["test_metrics"]
    else:
        log("  WARNING: Phase 2 best result not found yet, will use placeholder")
        protocol_a = None

    # For protocols b/c/d we need to reload raw data with file info
    log("  Loading raw data with file tracking...")
    import xgboost as xgb

    # We'll build a simplified dataset tracking source file
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
                df["source_folder"] = folder
                all_data.append(df)
            except Exception as e:
                log(f"    Error reading {csv_path}: {e}")

    full_df = pd.concat(all_data, ignore_index=True)
    del all_data
    gc.collect()
    log(f"  Full dataset: {len(full_df):,} rows")

    feature_cols = FEATURES
    X_all = full_df[feature_cols].values.astype(np.float32)
    y_all = full_df["label_8class"].values.astype(np.int32)

    # ── Protocol (b): globally deduplicated, shuffled split ──
    log("  (b) Dedup protocol...")
    row_bytes = [X_all[i].tobytes() for i in range(len(X_all))]
    # Combine features + label for dedup
    row_keys = [X_all[i].tobytes() + bytes([y_all[i]]) for i in range(len(X_all))]
    seen = set()
    dedup_mask = np.zeros(len(X_all), dtype=bool)
    for i, key in enumerate(row_keys):
        if key not in seen:
            seen.add(key)
            dedup_mask[i] = True
    del seen, row_keys
    gc.collect()

    n_dedup = dedup_mask.sum()
    log(f"    After dedup: {n_dedup:,} rows (removed {len(X_all)-n_dedup:,}, {100*(1-n_dedup/len(X_all)):.1f}%)")

    X_dedup = X_all[dedup_mask]
    y_dedup = y_all[dedup_mask]

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X_dedup))
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    dtrain = xgb.DMatrix(X_dedup[train_idx], label=y_dedup[train_idx])
    dtest = xgb.DMatrix(X_dedup[test_idx], label=y_dedup[test_idx])

    params = {
        "max_depth": 10, "learning_rate": 0.1, "max_bin": 256,
        "objective": "multi:softmax", "num_class": 8,
        "eval_metric": "mlogloss", "tree_method": "gpu_hist", "seed": SEED, "subsample": 0.8,
        "colsample_bytree": 0.8, "verbosity": 0,
    }
    model_b = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
    y_pred_b = model_b.predict(dtest).astype(int)
    protocol_b = compute_metrics(y_dedup[test_idx], y_pred_b, 8, CLASS8_ORDER)
    log(f"    (b) Dedup: macro_f1={protocol_b['macro_f1']:.4f} acc={protocol_b['accuracy']:.4f}")
    del dtrain, dtest, model_b, X_dedup, y_dedup
    gc.collect()

    # ── Protocol (c): file-grouped split ──
    log("  (c) File-grouped split...")
    source_files = full_df["source_file"].values
    unique_files = np.unique(source_files)
    rng2 = np.random.RandomState(SEED)
    rng2.shuffle(unique_files)
    split_point = int(0.8 * len(unique_files))
    train_files = set(unique_files[:split_point])
    test_files = set(unique_files[split_point:])

    file_train_mask = np.array([f in train_files for f in source_files])
    file_test_mask = ~file_train_mask

    dtrain_c = xgb.DMatrix(X_all[file_train_mask], label=y_all[file_train_mask])
    dtest_c = xgb.DMatrix(X_all[file_test_mask], label=y_all[file_test_mask])

    model_c = xgb.train(params, dtrain_c, num_boost_round=500, verbose_eval=False)
    y_pred_c = model_c.predict(dtest_c).astype(int)
    protocol_c = compute_metrics(y_all[file_test_mask], y_pred_c, 8, CLASS8_ORDER)
    log(f"    (c) File-grouped: macro_f1={protocol_c['macro_f1']:.4f} acc={protocol_c['accuracy']:.4f}")
    del dtrain_c, model_c
    gc.collect()

    # ── Protocol (d): file-grouped + twin-free test ──
    log("  (d) File-grouped + twin-free...")
    # Hash train rows
    X_train_c = X_all[file_train_mask]
    X_test_c = X_all[file_test_mask]
    y_test_c = y_all[file_test_mask]

    train_hash_set = set()
    for i in range(0, len(X_train_c), 500000):
        chunk = X_train_c[i:i+500000]
        for j in range(len(chunk)):
            train_hash_set.add(chunk[j].tobytes())

    twin_free_mask = np.zeros(len(X_test_c), dtype=bool)
    for i in range(0, len(X_test_c), 500000):
        chunk = X_test_c[i:i+500000]
        for j in range(len(chunk)):
            if chunk[j].tobytes() not in train_hash_set:
                twin_free_mask[i+j] = True

    n_twin_free = twin_free_mask.sum()
    log(f"    Twin-free test rows: {n_twin_free:,} / {len(X_test_c):,}")

    if n_twin_free > 0:
        dtest_d = xgb.DMatrix(X_test_c[twin_free_mask])
        y_pred_d = model_c_saved_pred = xgb.train(params, xgb.DMatrix(X_train_c, label=y_all[file_train_mask]),
                                                    num_boost_round=500, verbose_eval=False).predict(dtest_d).astype(int)
        protocol_d = compute_metrics(y_test_c[twin_free_mask], y_pred_d, 8, CLASS8_ORDER)
        log(f"    (d) File-grouped+twin-free: macro_f1={protocol_d['macro_f1']:.4f} acc={protocol_d['accuracy']:.4f}")
    else:
        protocol_d = None
        log("    (d) No twin-free rows!")

    result = {
        "protocol_a_paper": protocol_a,
        "protocol_b_dedup": protocol_b,
        "protocol_c_file_grouped": protocol_c,
        "protocol_d_file_grouped_twin_free": protocol_d,
        "dedup_removed_pct": float(100*(1-n_dedup/len(X_all))),
        "file_grouped_twin_free_pct": float(100*n_twin_free/len(X_test_c)) if len(X_test_c) > 0 else 0,
        "elapsed_seconds": time.time() - t0,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    log(f"N2 COMPLETE in {time.time()-t0:.0f}s")
    del full_df, X_all, y_all
    gc.collect()
    return result


# ═══════════════════════ N3: DoS vs DDoS DEEP DIVE ═══════════════════════

def n3_dos_vs_ddos():
    log("N3: DoS vs DDoS DEEP DIVE — starting")
    out_dir = NOVELTY_ROOT / "N3_dos_vs_ddos"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    import xgboost as xgb

    # Load data, filter to DoS and DDoS only
    log("  Loading train/test for DoS vs DDoS...")
    train_df = load_split("train")
    test_df = load_split("test")

    # DoS = 3, DDoS = 2 in 8-class
    dos_id = CLASS_MAPS["class8_to_id"]["DoS"]
    ddos_id = CLASS_MAPS["class8_to_id"]["DDoS"]

    train_mask = np.isin(train_df["label_8class"].values, [dos_id, ddos_id])
    test_mask = np.isin(test_df["label_8class"].values, [dos_id, ddos_id])

    X_train = train_df.loc[train_mask, FEATURES].values.astype(np.float32)
    y_train = (train_df.loc[train_mask, "label_8class"].values == ddos_id).astype(np.int32)  # 1=DDoS, 0=DoS
    X_test = test_df.loc[test_mask, FEATURES].values.astype(np.float32)
    y_test = (test_df.loc[test_mask, "label_8class"].values == ddos_id).astype(np.int32)

    log(f"  Train: {len(X_train):,} rows (DoS={( y_train==0).sum():,}, DDoS={(y_train==1).sum():,})")
    log(f"  Test:  {len(X_test):,} rows (DoS={(y_test==0).sum():,}, DDoS={(y_test==1).sum():,})")

    del train_df, test_df
    gc.collect()

    # Count identical feature vectors with different labels
    log("  Counting ambiguous rows (same features, different labels)...")
    SAMPLE = min(5000000, len(X_train))
    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(len(X_train), SAMPLE, replace=False)
    feature_label_map = {}
    ambiguous_count = 0
    for idx in sample_idx:
        key = X_train[idx].tobytes()
        label = y_train[idx]
        if key in feature_label_map:
            if feature_label_map[key] != label:
                ambiguous_count += 1
        else:
            feature_label_map[key] = label

    ambiguous_pct = ambiguous_count / SAMPLE * 100
    log(f"  Ambiguous rows (same features, diff label): {ambiguous_count:,} ({ambiguous_pct:.2f}%)")
    del feature_label_map
    gc.collect()

    # XGBoost binary classifier: paper protocol
    log("  Training XGBoost binary (DoS vs DDoS) — paper protocol...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "max_depth": 10, "learning_rate": 0.1, "max_bin": 256,
        "objective": "binary:logistic", "eval_metric": "logloss",
        "tree_method": "gpu_hist", "seed": SEED,
        "subsample": 0.8, "colsample_bytree": 0.8, "verbosity": 0,
    }

    model = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
    y_prob = model.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred).tolist()

    importance = model.get_score(importance_type="gain")
    importance_named = {FEATURES[int(k[1:])]: v for k, v in importance.items() if k.startswith("f")}
    top_features = dict(sorted(importance_named.items(), key=lambda x: -x[1])[:20])

    log(f"  Paper protocol: acc={acc:.4f} AUC={auc:.4f} macro_f1={macro_f1:.4f}")

    paper_protocol_result = {
        "accuracy": float(acc), "auc": float(auc), "macro_f1": float(macro_f1),
        "confusion_matrix": cm, "top_features": top_features,
    }

    # File-grouped split for DoS vs DDoS
    log("  Training XGBoost binary (DoS vs DDoS) — file-grouped split...")
    # Reload with file info
    all_data = []
    dos_ddos_folders = [f for f in sorted(os.listdir(DATA_ROOT))
                        if (DATA_ROOT / f).is_dir() and
                        (CLASS_MAPS["folder_to_8class"].get(f, "") in ("DoS", "DDoS"))]

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

    dtrain2 = xgb.DMatrix(X_tr, label=y_tr)
    dtest2 = xgb.DMatrix(X_te, label=y_te)

    model2 = xgb.train(params, dtrain2, num_boost_round=500, verbose_eval=False)
    y_prob2 = model2.predict(dtest2)
    y_pred2 = (y_prob2 > 0.5).astype(int)

    acc2 = accuracy_score(y_te, y_pred2)
    auc2 = roc_auc_score(y_te, y_prob2) if len(np.unique(y_te)) > 1 else 0
    macro_f1_2 = f1_score(y_te, y_pred2, average="macro")
    cm2 = confusion_matrix(y_te, y_pred2).tolist()

    log(f"  File-grouped: acc={acc2:.4f} AUC={auc2:.4f} macro_f1={macro_f1_2:.4f}")

    file_grouped_result = {
        "accuracy": float(acc2), "auc": float(auc2), "macro_f1": float(macro_f1_2),
        "confusion_matrix": cm2,
    }

    result = {
        "paper_protocol": paper_protocol_result,
        "file_grouped": file_grouped_result,
        "ambiguous_rows_pct": float(ambiguous_pct),
        "memorization_signal": "YES" if (acc - acc2) > 0.05 else "NO",
        "accuracy_drop": float(acc - acc2),
        "elapsed_seconds": time.time() - t0,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    log(f"N3 COMPLETE in {time.time()-t0:.0f}s")
    del full_df
    gc.collect()
    return result


# ═══════════════════════ N4: FINGERPRINT ABLATION ═══════════════════════

def n4_fingerprint_ablation():
    log("N4: FINGERPRINT ABLATION — starting")
    out_dir = NOVELTY_ROOT / "N4_fingerprint_ablation"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    import xgboost as xgb

    log("  Loading train/val/test data...")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    y_train = train_df["label_2class"].values.astype(np.int32)
    y_val = val_df["label_2class"].values.astype(np.int32)
    y_test = test_df["label_2class"].values.astype(np.int32)

    # Baseline: all features
    log("  Training baseline (all 39 features)...")
    X_train = train_df[FEATURES].values.astype(np.float32)
    X_val = val_df[FEATURES].values.astype(np.float32)
    X_test = test_df[FEATURES].values.astype(np.float32)

    params = {
        "max_depth": 10, "learning_rate": 0.1, "max_bin": 256,
        "objective": "binary:logistic", "eval_metric": "logloss",
        "tree_method": "gpu_hist", "seed": SEED,
        "subsample": 0.8, "colsample_bytree": 0.8, "verbosity": 0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(params, dtrain, num_boost_round=500,
                       evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)

    y_pred = (model.predict(dtest) > 0.5).astype(int)
    baseline_acc = accuracy_score(y_test, y_pred)
    baseline_f1 = f1_score(y_test, y_pred, average="macro")
    log(f"  Baseline: acc={baseline_acc:.6f} macro_f1={baseline_f1:.6f}")

    # Get feature importance
    importance = model.get_score(importance_type="gain")
    importance_named = {}
    for k, v in importance.items():
        if k.startswith("f"):
            idx = int(k[1:])
            importance_named[FEATURES[idx]] = v
    sorted_features = sorted(importance_named.items(), key=lambda x: -x[1])
    top10 = [f[0] for f in sorted_features[:10]]
    log(f"  Top 10 features: {top10}")

    del dtrain, dval, dtest, model, X_train, X_val, X_test
    gc.collect()

    # Ablation 1: Remove Time_To_Live and IAT
    remove_ttl_iat = ["Time_To_Live", "IAT"]
    feats_no_ttl = [f for f in FEATURES if f not in remove_ttl_iat]
    log(f"  Training without {remove_ttl_iat} ({len(feats_no_ttl)} features)...")

    X_train_a1 = train_df[feats_no_ttl].values.astype(np.float32)
    X_val_a1 = val_df[feats_no_ttl].values.astype(np.float32)
    X_test_a1 = test_df[feats_no_ttl].values.astype(np.float32)

    dtrain = xgb.DMatrix(X_train_a1, label=y_train)
    dval = xgb.DMatrix(X_val_a1, label=y_val)
    dtest = xgb.DMatrix(X_test_a1, label=y_test)

    model_a1 = xgb.train(params, dtrain, num_boost_round=500,
                          evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
    y_pred_a1 = (model_a1.predict(dtest) > 0.5).astype(int)
    acc_a1 = accuracy_score(y_test, y_pred_a1)
    f1_a1 = f1_score(y_test, y_pred_a1, average="macro")
    log(f"  Without TTL+IAT: acc={acc_a1:.6f} macro_f1={f1_a1:.6f} (drop: {baseline_acc-acc_a1:.6f})")

    del dtrain, dval, dtest, model_a1, X_train_a1, X_val_a1, X_test_a1
    gc.collect()

    # Ablation 2: Remove top 10 features
    feats_no_top10 = [f for f in FEATURES if f not in top10]
    log(f"  Training without top 10 features ({len(feats_no_top10)} features)...")

    X_train_a2 = train_df[feats_no_top10].values.astype(np.float32)
    X_val_a2 = val_df[feats_no_top10].values.astype(np.float32)
    X_test_a2 = test_df[feats_no_top10].values.astype(np.float32)

    dtrain = xgb.DMatrix(X_train_a2, label=y_train)
    dval = xgb.DMatrix(X_val_a2, label=y_val)
    dtest = xgb.DMatrix(X_test_a2, label=y_test)

    model_a2 = xgb.train(params, dtrain, num_boost_round=500,
                          evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
    y_pred_a2 = (model_a2.predict(dtest) > 0.5).astype(int)
    acc_a2 = accuracy_score(y_test, y_pred_a2)
    f1_a2 = f1_score(y_test, y_pred_a2, average="macro")
    log(f"  Without top 10: acc={acc_a2:.6f} macro_f1={f1_a2:.6f} (drop: {baseline_acc-acc_a2:.6f})")

    result = {
        "baseline": {"accuracy": float(baseline_acc), "macro_f1": float(baseline_f1), "n_features": 39},
        "without_ttl_iat": {"accuracy": float(acc_a1), "macro_f1": float(f1_a1),
                            "removed": remove_ttl_iat, "n_features": len(feats_no_ttl),
                            "accuracy_drop": float(baseline_acc - acc_a1)},
        "without_top10": {"accuracy": float(acc_a2), "macro_f1": float(f1_a2),
                          "removed": top10, "n_features": len(feats_no_top10),
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


# ═══════════════════════ N5: WINDOW-SIZE ARTIFACT ═══════════════════════

def n5_window_size():
    log("N5: WINDOW-SIZE ARTIFACT — starting")
    out_dir = NOVELTY_ROOT / "N5_window_size"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    import xgboost as xgb

    # Window 100: DDoS, DoS, Mirai; Window 10: Recon, Web_Based, Spoofing, Brute_Force, Benign
    window_100 = {"DDoS", "DoS", "Mirai"}
    window_10 = {"Recon", "Web_Based", "Spoofing", "Brute_Force", "Benign"}

    log("  Loading data...")
    train_df = load_split("train")
    test_df = load_split("test")

    y_train_8 = train_df["label_8class"].values.astype(np.int32)
    y_test_8 = test_df["label_8class"].values.astype(np.int32)

    # Create window group labels
    window_train = np.array([1 if CLASS8_ORDER[y] in window_100 else 0 for y in y_train_8])
    window_test = np.array([1 if CLASS8_ORDER[y] in window_100 else 0 for y in y_test_8])

    X_train = train_df[FEATURES].values.astype(np.float32)
    X_test = test_df[FEATURES].values.astype(np.float32)

    log(f"  Window 100 train: {(window_train==1).sum():,}, Window 10 train: {(window_train==0).sum():,}")

    # Train classifier to predict window group
    log("  Training window-group classifier...")
    dtrain = xgb.DMatrix(X_train, label=window_train)
    dtest = xgb.DMatrix(X_test, label=window_test)

    params = {
        "max_depth": 8, "learning_rate": 0.1, "max_bin": 256,
        "objective": "binary:logistic", "eval_metric": "logloss",
        "tree_method": "gpu_hist", "seed": SEED,
        "verbosity": 0,
    }

    model = xgb.train(params, dtrain, num_boost_round=300, verbose_eval=False)
    y_prob = model.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)

    window_acc = accuracy_score(window_test, y_pred)
    window_f1 = f1_score(window_test, y_pred, average="macro")
    log(f"  Window classifier: acc={window_acc:.4f} macro_f1={window_f1:.4f}")

    # Get discriminating features
    importance = model.get_score(importance_type="gain")
    importance_named = {FEATURES[int(k[1:])]: v for k, v in importance.items() if k.startswith("f")}
    sorted_feats = sorted(importance_named.items(), key=lambda x: -x[1])
    top_window_feats = [f[0] for f in sorted_feats[:10]]
    log(f"  Top window-discriminating features: {top_window_feats}")

    del dtrain, dtest, model
    gc.collect()

    # Now train 8-class model WITHOUT top window-discriminating features
    log("  Training 8-class model without top window features...")
    feats_no_window = [f for f in FEATURES if f not in top_window_feats]
    X_train_nw = train_df[feats_no_window].values.astype(np.float32)
    X_test_nw = test_df[feats_no_window].values.astype(np.float32)

    dtrain = xgb.DMatrix(X_train_nw, label=y_train_8)
    dtest = xgb.DMatrix(X_test_nw, label=y_test_8)

    params8 = {
        "max_depth": 10, "learning_rate": 0.1, "max_bin": 256,
        "objective": "multi:softmax", "num_class": 8,
        "eval_metric": "mlogloss", "tree_method": "gpu_hist", "seed": SEED, "verbosity": 0,
    }

    model8 = xgb.train(params8, dtrain, num_boost_round=500, verbose_eval=False)
    y_pred8 = model8.predict(dtest).astype(int)
    metrics_no_window = compute_metrics(y_test_8, y_pred8, 8, CLASS8_ORDER)
    log(f"  8-class without window features: macro_f1={metrics_no_window['macro_f1']:.4f} acc={metrics_no_window['accuracy']:.4f}")

    # Compare with full-feature 8-class
    log("  Training 8-class baseline (all features)...")
    X_train_full = train_df[FEATURES].values.astype(np.float32)
    X_test_full = test_df[FEATURES].values.astype(np.float32)
    dtrain_full = xgb.DMatrix(X_train_full, label=y_train_8)
    dtest_full = xgb.DMatrix(X_test_full, label=y_test_8)

    model8_full = xgb.train(params8, dtrain_full, num_boost_round=500, verbose_eval=False)
    y_pred8_full = model8_full.predict(dtest_full).astype(int)
    metrics_full = compute_metrics(y_test_8, y_pred8_full, 8, CLASS8_ORDER)
    log(f"  8-class baseline: macro_f1={metrics_full['macro_f1']:.4f} acc={metrics_full['accuracy']:.4f}")

    result = {
        "window_classifier_accuracy": float(window_acc),
        "window_classifier_f1": float(window_f1),
        "window_leaks_class": window_acc > 0.9,
        "top_window_features": top_window_feats,
        "8class_baseline": {"macro_f1": metrics_full["macro_f1"], "accuracy": metrics_full["accuracy"],
                            "per_class_f1": metrics_full["per_class_f1"]},
        "8class_no_window_features": {"macro_f1": metrics_no_window["macro_f1"], "accuracy": metrics_no_window["accuracy"],
                                      "per_class_f1": metrics_no_window["per_class_f1"]},
        "macro_f1_change": float(metrics_no_window["macro_f1"] - metrics_full["macro_f1"]),
        "elapsed_seconds": time.time() - t0,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    log(f"N5 COMPLETE in {time.time()-t0:.0f}s")
    del train_df, test_df
    gc.collect()
    return result


# ═══════════════════════ N6: CLASS-WEIGHT SENSITIVITY ═══════════════════════

def n6_class_weight_sweep():
    log("N6: CLASS-WEIGHT SENSITIVITY — starting")
    out_dir = NOVELTY_ROOT / "N6_class_weight_sweep"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    import xgboost as xgb

    log("  Loading data...")
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
    k = n_classes

    results_list = []

    for p in [0, 0.25, 0.5, 0.75, 1.0]:
        log(f"  Power p={p}...")
        if p == 0:
            sample_w = None
        else:
            weights = (N / (k * np.maximum(counts, 1))) ** p
            sample_w = weights[y_train].astype(np.float32)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_w)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            "max_depth": 10, "learning_rate": 0.1, "max_bin": 256,
            "objective": "multi:softmax", "num_class": 8,
            "eval_metric": "mlogloss", "tree_method": "gpu_hist",
            "seed": SEED, "subsample": 0.8,
            "colsample_bytree": 0.8, "verbosity": 0,
        }

        model = xgb.train(params, dtrain, num_boost_round=500,
                           evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
        y_pred = model.predict(dtest).astype(int)
        metrics = compute_metrics(y_test, y_pred, n_classes, CLASS8_ORDER)

        log(f"    p={p}: acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
        for cls in CLASS8_ORDER:
            log(f"      {cls}: F1={metrics['per_class_f1'][cls]:.4f}")

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


# ═══════════════════════ MAIN ═══════════════════════

def main():
    log("PHASE 3: NOVELTY EXPERIMENTS")
    log(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        n1_twin_analysis()
    except Exception as e:
        log(f"N1 FAILED: {e}")
        traceback.print_exc()

    try:
        n3_dos_vs_ddos()
    except Exception as e:
        log(f"N3 FAILED: {e}")
        traceback.print_exc()

    try:
        n4_fingerprint_ablation()
    except Exception as e:
        log(f"N4 FAILED: {e}")
        traceback.print_exc()

    try:
        n5_window_size()
    except Exception as e:
        log(f"N5 FAILED: {e}")
        traceback.print_exc()

    try:
        n6_class_weight_sweep()
    except Exception as e:
        log(f"N6 FAILED: {e}")
        traceback.print_exc()

    try:
        n2_protocol_ladder()
    except Exception as e:
        log(f"N2 FAILED: {e}")
        traceback.print_exc()

    log(f"\nPHASE 3 COMPLETE at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    import traceback
    main()
