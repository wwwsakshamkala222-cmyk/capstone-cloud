"""Retrain the v4 best no-IAT 8-class model and save it as a deployable artifact.

v4 (src/v4/train_v4.py) saved metrics and predictions but no model files. This script rebuilds
exactly the model v4 selected on validation for 8-class without IAT:
  xgb_best 8-class (cfg3_weighted: lossguide, 1024 leaves, lr 0.05, max_bin 512, sqrt class weights),
  45 features (the paper's 46 minus IAT, which is a capture-timestamp proxy).
Same split as src/v4/prep_v4.py (80/20 seed 42, then 5% of train as validation), same scaler (fit on
train), early stopping on validation only. The test split is not touched here.

Outputs in ART (off the home disk, not committed):
  model_8c_noiat.ubj   XGBoost booster (trees up to best_iteration)
  preprocess.json      feature order, scaler mean/scale, class names, val metrics
  idx_test.npy         row indices of the v4 test split in merged_raw.parquet
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

V4 = Path("/dev/shm/saksham/v4")
ART = V4 / "prevention"
SEED = 42
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v4"))
from prep_v4 import RENAME, CLASS_MAPS  # noqa: E402  same label mapping as v4


def main():
    gpu = sys.argv[1] if len(sys.argv) > 1 else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    ART.mkdir(exist_ok=True)
    meta = json.loads((V4 / "meta.json").read_text())
    cm = json.loads(CLASS_MAPS.read_text())
    feats = [f for f in meta["features_46"] if f != "IAT"]

    df = pd.read_parquet(V4 / "merged_raw.parquet")
    lab34 = df["label"].replace(RENAME)
    y8 = lab34.map(cm["folder_to_8class"]).map(cm["class8_to_id"]).to_numpy(np.int8)
    idx = np.arange(len(df))
    idx_trfull, idx_te = train_test_split(idx, test_size=0.20, random_state=SEED, shuffle=True)
    idx_tr, idx_va = train_test_split(idx_trfull, test_size=0.05, random_state=SEED, shuffle=True)
    assert (len(idx_tr), len(idx_va), len(idx_te)) == (meta["n_train"], meta["n_val"], meta["n_test"])
    np.save(ART / "idx_test.npy", idx_te)

    X = df[feats].to_numpy(np.float64)
    mean, scale = X[idx_tr].mean(0), X[idx_tr].std(0)  # == StandardScaler fit on train
    scale[scale == 0] = 1.0
    # Sanity check: identical to the v4 prepared validation array.
    cols = [meta["features_46"].index(f) for f in feats]
    ref = np.load(V4 / "X_va_46.npy", mmap_mode="r")[:1000][:, cols]
    assert np.allclose(((X[idx_va[:1000]] - mean) / scale).astype(np.float32), ref, atol=1e-5)

    Xtr = ((X[idx_tr] - mean) / scale).astype(np.float32)
    Xva = ((X[idx_va] - mean) / scale).astype(np.float32)
    ytr, yva = y8[idx_tr], y8[idx_va]
    counts = np.bincount(ytr, minlength=8).astype(np.float64)
    w = ((len(ytr) / (8 * counts)) ** 0.5)[ytr].astype(np.float32)
    params = {"max_leaves": 1024, "max_depth": 0, "grow_policy": "lossguide", "learning_rate": 0.05,
              "max_bin": 512, "subsample": 0.8, "colsample_bytree": 0.8, "objective": "multi:softprob",
              "num_class": 8, "eval_metric": "mlogloss", "tree_method": "hist", "device": "cuda:0",
              "seed": SEED, "verbosity": 0}
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=w, feature_names=feats)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=feats)
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")],
                    early_stopping_rounds=50, verbose_eval=200)
    bst = bst[: bst.best_iteration + 1]  # keep only the trees up to the validation optimum
    p = bst.predict(dva).argmax(1)
    val = {"accuracy": float(accuracy_score(yva, p)), "macro_f1": float(f1_score(yva, p, average="macro"))}
    bst.set_param({"device": "cpu"})
    bst.save_model(ART / "model_8c_noiat.ubj")
    (ART / "preprocess.json").write_text(json.dumps({
        "features": feats, "mean": mean.tolist(), "scale": scale.tolist(),
        "class8_order": cm["class8_order"], "model": "xgb_best 8-class no-IAT (cfg3_weighted)",
        "best_iteration": int(bst.num_boosted_rounds() - 1), "val_metrics": val,
        "v4_reference_val_metrics": json.loads((V4 / "runs/xgb_best_8c_46nf.json").read_text())["val_metrics"],
    }, indent=1))
    print("val", val, "rounds", bst.num_boosted_rounds())


if __name__ == "__main__":
    main()
