#!/usr/bin/env python3
"""Phase 1: Data pipeline for CICIoT2023 dataset.
Reads CSVs, assigns labels, splits, writes parquet cache.
"""
import os, sys, json, time, glob
import numpy as np
import pandas as pd
from pathlib import Path

DATA_ROOT = Path("/dev/shm/saksham/ciciot2023/CSV")
CACHE_ROOT = Path("/dev/shm/saksham/ids_cache")
IDS_ROOT = Path("/dev/shm/saksham/ids")
SEED = 42

# Class mappings: folder_name -> (label_34, label_8, label_2)
FOLDER_TO_8CLASS = {
    # DDoS (12 folders)
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-PSHACK_FLOOD": "DDoS",
    "DDoS-RSTFINFLOOD": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    # DoS (4 folders)
    "DoS-HTTP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-UDP_Flood": "DoS",
    # Mirai (3 folders)
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    # Recon (5 folders)
    "Recon-HostDiscovery": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PingSweep": "Recon",
    "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon",
    # Web_Based (6 folders)
    "Backdoor_Malware": "Web_Based",
    "BrowserHijacking": "Web_Based",
    "CommandInjection": "Web_Based",
    "SqlInjection": "Web_Based",
    "Uploading_Attack": "Web_Based",
    "XSS": "Web_Based",
    # Spoofing (2 folders)
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
    # Brute_Force (1 folder)
    "DictionaryBruteForce": "Brute_Force",
    # Benign (1 folder)
    "Benign_Final": "Benign",
}

CLASS8_ORDER = ["Benign", "Brute_Force", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web_Based"]
CLASS8_TO_ID = {name: i for i, name in enumerate(CLASS8_ORDER)}

# Build 34-class mapping: we need a consistent ordering
ALL_FOLDERS = sorted(FOLDER_TO_8CLASS.keys())
CLASS34_TO_ID = {name: i for i, name in enumerate(ALL_FOLDERS)}

def load_folder(folder_name):
    """Load all CSVs from a folder, drop NaN/inf, return DataFrame."""
    folder_path = DATA_ROOT / folder_name
    csv_files = sorted(glob.glob(str(folder_path / "*.csv")))
    if not csv_files:
        print(f"  WARNING: No CSV files in {folder_path}")
        return None

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  ERROR reading {f}: {e}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    # Get feature columns (first 39 columns based on header)
    feature_cols = list(df.columns)
    if len(feature_cols) < 39:
        print(f"  WARNING: {folder_name} has {len(feature_cols)} columns, expected >= 39")

    # Some CSVs may have a label column at the end — drop it if present
    if 'label' in feature_cols:
        feature_cols.remove('label')
    if 'Label' in feature_cols:
        feature_cols.remove('Label')

    # Use first 39 feature columns
    feature_cols = feature_cols[:39]
    df = df[feature_cols].copy()

    # Convert to numeric, coerce errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN or inf
    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    n_after = len(df)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} NaN/inf rows from {folder_name}")

    return df

def main():
    start_time = time.time()

    # Verify all folders exist
    folders = sorted([d for d in os.listdir(DATA_ROOT) if (DATA_ROOT / d).is_dir()])
    print(f"Found {len(folders)} folders: {folders}")
    assert len(folders) == 34, f"Expected 34 folders, got {len(folders)}"

    # Check all folders are in our mapping
    for f in folders:
        assert f in FOLDER_TO_8CLASS, f"Folder {f} not in mapping!"

    # Save class maps
    os.makedirs(IDS_ROOT / "src", exist_ok=True)
    with open(IDS_ROOT / "src" / "class_maps.json", "w") as fp:
        json.dump({
            "class8_to_id": CLASS8_TO_ID,
            "class34_to_id": CLASS34_TO_ID,
            "folder_to_8class": FOLDER_TO_8CLASS,
            "class8_order": CLASS8_ORDER,
            "class34_order": ALL_FOLDERS,
        }, fp, indent=2)

    # Process each folder
    all_features = None
    total_rows = 0
    class8_counts = {c: 0 for c in CLASS8_ORDER}
    class34_counts = {c: 0 for c in ALL_FOLDERS}

    # We'll collect train/val/test parts and write periodically
    train_parts = []
    val_parts = []
    test_parts = []
    part_idx = 0

    rng = np.random.RandomState(SEED)

    for i, folder in enumerate(folders):
        print(f"\n[{i+1}/34] Processing {folder}...")
        df = load_folder(folder)
        if df is None:
            continue

        if all_features is None:
            all_features = list(df.columns)
            print(f"  Features ({len(all_features)}): {all_features}")
            assert len(all_features) == 39, f"Expected 39 features, got {len(all_features)}"

        # Assign labels
        label_8 = CLASS8_TO_ID[FOLDER_TO_8CLASS[folder]]
        label_34 = CLASS34_TO_ID[folder]
        label_2 = 0 if FOLDER_TO_8CLASS[folder] == "Benign" else 1

        n = len(df)
        total_rows += n
        class8_counts[FOLDER_TO_8CLASS[folder]] += n
        class34_counts[folder] += n

        # Convert to float32
        X = df.values.astype(np.float32)

        # Random split: 75/5/20
        rand_vals = rng.rand(n)
        train_mask = rand_vals < 0.75
        val_mask = (rand_vals >= 0.75) & (rand_vals < 0.80)
        test_mask = rand_vals >= 0.80

        for mask, split_name, parts_list in [
            (train_mask, "train", train_parts),
            (val_mask, "val", val_parts),
            (test_mask, "test", test_parts),
        ]:
            if mask.sum() == 0:
                continue
            split_df = pd.DataFrame(X[mask], columns=all_features)
            split_df["label_2class"] = np.int8(label_2)
            split_df["label_8class"] = np.int8(label_8)
            split_df["label_34class"] = np.int8(label_34)
            parts_list.append(split_df)

        print(f"  {folder}: {n:,} rows | 8class={FOLDER_TO_8CLASS[folder]} | train={train_mask.sum():,} val={val_mask.sum():,} test={test_mask.sum():,}")

        # Write to disk periodically to free memory (every 5 folders)
        if (i + 1) % 5 == 0 or i == len(folders) - 1:
            for split_name, parts_list in [("train", train_parts), ("val", val_parts), ("test", test_parts)]:
                if parts_list:
                    combined = pd.concat(parts_list, ignore_index=True)
                    out_path = CACHE_ROOT / split_name / f"part_{part_idx:04d}.parquet"
                    combined.to_parquet(out_path, engine="pyarrow")
                    print(f"  Wrote {split_name} part {part_idx}: {len(combined):,} rows")
            train_parts.clear()
            val_parts.clear()
            test_parts.clear()
            part_idx += 1

    elapsed = time.time() - start_time

    # Print summary
    print(f"\n{'='*60}")
    print(f"DATA PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"Total rows: {total_rows:,}")
    print(f"\n8-class counts:")
    for cls in CLASS8_ORDER:
        pct = 100 * class8_counts[cls] / total_rows
        print(f"  {cls:15s}: {class8_counts[cls]:>12,} ({pct:6.2f}%)")
    print(f"\n34-class counts:")
    for cls in sorted(class34_counts.keys()):
        pct = 100 * class34_counts[cls] / total_rows
        print(f"  {cls:30s}: {class34_counts[cls]:>12,} ({pct:6.2f}%)")

    # Verify split sizes
    print(f"\nVerifying cached parquet files...")
    for split in ["train", "val", "test"]:
        parts = sorted(glob.glob(str(CACHE_ROOT / split / "*.parquet")))
        n_split = sum(len(pd.read_parquet(p, columns=["label_2class"])) for p in parts)
        print(f"  {split}: {n_split:,} rows across {len(parts)} parts")

    # Save metadata
    meta = {
        "total_rows": total_rows,
        "features": all_features,
        "n_features": len(all_features),
        "class8_counts": class8_counts,
        "class34_counts": class34_counts,
        "class8_to_id": CLASS8_TO_ID,
        "class34_to_id": CLASS34_TO_ID,
        "seed": SEED,
        "split_ratio": "75/5/20",
        "elapsed_seconds": elapsed,
    }
    with open(IDS_ROOT / "src" / "data_meta.json", "w") as fp:
        json.dump(meta, fp, indent=2)

    print(f"\nMetadata saved to {IDS_ROOT / 'src' / 'data_meta.json'}")
    print("PHASE 1 COMPLETE")

if __name__ == "__main__":
    main()
