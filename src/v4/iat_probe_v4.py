"""v4 IAT probe: is IAT a capture-time proxy in the Kaggle 46-feature subset?

1. Single-feature probe: for each of the 46 features, a depth-8 decision tree trained on that one feature
   (600k-row random sample of the v4 train split, seed 0), scored on the v4 validation split (8-class).
2. Per-class IAT distribution on the raw merged data (min, 1%, 50%, 99%, max).
Writes results/v4_46feat/iat_probe.json. Uses train/validation only; the test split is not touched.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

D = Path("/dev/shm/saksham/v4")
OUT = Path(__file__).resolve().parents[2] / "results" / "v4_46feat" / "iat_probe.json"


def main():
    meta = json.loads((D / "meta.json").read_text())
    feats = meta["features_46"]
    rng = np.random.default_rng(0)
    Xtr = np.load(D / "X_tr_46.npy", mmap_mode="r")
    ytr = np.load(D / "y_tr_8.npy")
    i = np.sort(rng.choice(len(ytr), 600_000, replace=False))
    Xs, ys = np.asarray(Xtr[i]), ytr[i]
    Xva, yva = np.load(D / "X_va_46.npy"), np.load(D / "y_va_8.npy")
    maj = int(np.bincount(ys).argmax())
    probe = []
    for j, f in enumerate(feats):
        t = DecisionTreeClassifier(max_depth=8, random_state=0).fit(Xs[:, [j]], ys)
        p = t.predict(Xva[:, [j]])
        probe.append({"feature": f, "val_accuracy": float((p == yva).mean()),
                      "val_macro_f1": float(f1_score(yva, p, average="macro"))})
    probe.sort(key=lambda r: -r["val_macro_f1"])

    df = pd.read_parquet(D / "merged_raw.parquet", columns=["IAT", "label"])
    q = df.groupby("label")["IAT"].quantile([0.0, 0.01, 0.5, 0.99, 1.0]).unstack()
    q.columns = ["min", "p01", "p50", "p99", "max"]
    bands = {lab: {k: float(v) for k, v in row.items()} for lab, row in q.sort_values("p50").iterrows()}
    out = {"task": "8-class", "probe_model": "DecisionTree(max_depth=8) on one feature",
           "train_sample_rows": int(len(ys)), "val_rows": int(len(yva)),
           "majority_class_val_accuracy": float((yva == maj).mean()),
           "single_feature_probe": probe, "iat_per_class_quantiles_raw": bands}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(probe[:5], indent=1), out["majority_class_val_accuracy"])


if __name__ == "__main__":
    main()
