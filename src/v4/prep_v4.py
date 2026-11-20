"""v4 prep: Kaggle 46-feature CICIoT2023 subset -> paper-protocol split.

Protocol (Neto et al. 2023 style): merge the Kaggle train/validation/test files, shuffle,
80/20 train/test (seed 42), then 5% of the train part as validation (selection only).
No de-duplication (to match the paper; twin % is reported separately).
StandardScaler is fit on the train part only.

Outputs (in OUT, kept off the home disk):
  X_{tr,va,te}_46.npy  float32 scaled, 46 features
  y_{tr,va,te}_{2,8,34}.npy  int labels
  meta.json  feature lists, class names, split sizes, twin stats
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SRC = Path("/dev/shm/saksham/v4/merged_raw.parquet")  # concat of Kaggle train/validation/test CSVs
OUT = Path("/dev/shm/saksham/v4")
CLASS_MAPS = Path(__file__).resolve().parents[1] / "preprocessing" / "class_maps.json"
SEED = 42

# The 8 features the paper has but our 39-feature release lacks.
EXTRA_8 = ["flow_duration", "Srate", "Drate", "urg_count", "Magnitue", "Radius", "Covariance", "Weight"]
# Kaggle label spelling -> our class_maps.json spelling
RENAME = {"BenignTraffic": "Benign_Final", "DDoS-PSHACK_Flood": "DDoS-PSHACK_FLOOD",
          "DDoS-RSTFINFlood": "DDoS-RSTFINFLOOD"}


def main():
    cm = json.loads(CLASS_MAPS.read_text())
    df = pd.read_parquet(SRC)
    feats46 = [c for c in df.columns if c != "label"]
    assert len(feats46) == 46 and all(f in feats46 for f in EXTRA_8)
    feats38 = [f for f in feats46 if f not in EXTRA_8]

    lab34 = df["label"].replace(RENAME)
    unknown = set(lab34.unique()) - set(cm["class34_to_id"])
    assert not unknown, unknown
    y34 = lab34.map(cm["class34_to_id"]).to_numpy(np.int8)
    y8 = lab34.map(cm["folder_to_8class"]).map(cm["class8_to_id"]).to_numpy(np.int8)
    y2 = (lab34 != "Benign_Final").to_numpy(np.int8)  # 0 = benign, 1 = attack

    X = df[feats46].to_numpy(np.float64)
    idx = np.arange(len(df))
    idx_trfull, idx_te = train_test_split(idx, test_size=0.20, random_state=SEED, shuffle=True)
    idx_tr, idx_va = train_test_split(idx_trfull, test_size=0.05, random_state=SEED, shuffle=True)

    scaler = StandardScaler().fit(X[idx_tr])
    for name, ids in [("tr", idx_tr), ("va", idx_va), ("te", idx_te)]:
        np.save(OUT / f"X_{name}_46.npy", scaler.transform(X[ids]).astype(np.float32))
        for k, y in [("2", y2), ("8", y8), ("34", y34)]:
            np.save(OUT / f"y_{name}_{k}.npy", y[ids])

    # Twin check: test rows whose exact raw feature vector also appears in the train part (train+val).
    def twins(cols):
        h = pd.util.hash_pandas_object(df[cols], index=False).to_numpy()
        train_hashes = set(h[idx_trfull].tolist())
        is_twin = np.fromiter((x in train_hashes for x in h[idx_te]), bool, len(idx_te))
        per8 = {c: float(is_twin[y8[idx_te] == i].mean()) for c, i in cm["class8_to_id"].items()}
        return {"test_twin_pct": float(is_twin.mean() * 100), "per_class8_twin_frac": per8,
                "dup_rows_in_full_data_pct": float(pd.Series(h).duplicated().mean() * 100)}

    meta = {"n_rows": len(df), "n_train": len(idx_tr), "n_val": len(idx_va), "n_test": len(idx_te),
            "features_46": feats46, "features_38": feats38, "extra_8": EXTRA_8,
            "class8_order": cm["class8_order"], "class34_order": cm["class34_order"],
            "twins_46": twins(feats46), "twins_38": twins(feats38)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if not k.startswith("features")}, indent=1))


if __name__ == "__main__":
    main()
