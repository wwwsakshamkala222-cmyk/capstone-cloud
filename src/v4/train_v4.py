"""v4 training: one model on one task with one feature set.

usage: train_v4.py <model> <task> <fset> [gpu]
  model: rf | xgb_best | xgb_unw
  task : 2 | 8 | 34
  fset : 46 | 38 | 46n | 38n
         38 = the 46 minus the 8 paper-only features, i.e. our 39-feature release without IGMP
         (which this subset lacks). Suffix "n" also drops IAT: in this data IAT is a capture
         timestamp whose per-class bands identify the capture session (see report).

xgb_best = our prior best config per task:
  2-class  cfg2           (lossguide, 1024 leaves, lr 0.05, max_bin 512), unweighted
  8-class  cfg3_weighted  (cfg2 + sqrt class weights)
  34-class balanced_p050  (depth 12, lr 0.05, max_bin 512, sqrt class weights)
xgb_unw  = one unweighted XGB per task:
  2-class  cfg1 (depth 12, lr 0.1, max_bin 256)  -- cfg2 is already unweighted
  8-class  cfg2
  34-class depth 12, lr 0.05, max_bin 512, no weights
Early stopping on validation only. The test set is scored once at the end; it is never used
for stopping, selection or thresholds.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

D = Path("/dev/shm/saksham/v4")
OUT = D / "runs"
SEED = 42

CFG2 = {"max_leaves": 1024, "max_depth": 0, "grow_policy": "lossguide", "learning_rate": 0.05,
        "max_bin": 512, "subsample": 0.8, "colsample_bytree": 0.8}
CFG1 = {"max_depth": 12, "learning_rate": 0.1, "max_bin": 256, "subsample": 0.8, "colsample_bytree": 0.8}
D12 = {"max_depth": 12, "learning_rate": 0.05, "max_bin": 512, "subsample": 0.8, "colsample_bytree": 0.8}
XGB = {  # (params, weighted?)
    ("xgb_best", 2): (CFG2, False), ("xgb_best", 8): (CFG2, True), ("xgb_best", 34): (D12, True),
    ("xgb_unw", 2): (CFG1, False), ("xgb_unw", 8): (CFG2, False), ("xgb_unw", 34): (D12, False),
}


def metrics(y, p):
    return {"accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro")),
            "weighted_f1": float(f1_score(y, p, average="weighted"))}


def main():
    model, task, fset = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    gpu = sys.argv[4] if len(sys.argv) > 4 else "0"
    name = f"{model}_{task}c_{fset}f"
    OUT.mkdir(exist_ok=True)
    meta = json.loads((D / "meta.json").read_text())
    feats = meta["features_46"] if fset.startswith("46") else meta["features_38"]
    if fset.endswith("n"):
        feats = [f for f in feats if f != "IAT"]
    cols = [meta["features_46"].index(f) for f in feats]

    X_tr = np.load(D / "X_tr_46.npy", mmap_mode="r")[:, cols]
    X_va = np.load(D / "X_va_46.npy", mmap_mode="r")[:, cols]
    y_tr, y_va = np.load(D / f"y_tr_{task}.npy"), np.load(D / f"y_va_{task}.npy")
    n_cls = int(max(y_tr.max(), y_va.max())) + 1
    t0 = time.time()
    info = {}

    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, n_jobs=16, random_state=SEED)
        clf.fit(X_tr, y_tr)
        predict = clf.predict
        imp = clf.feature_importances_
        info["config"] = {"n_estimators": 200, "n_jobs": 16, "other": "sklearn defaults"}
    else:
        import xgboost as xgb
        params, weighted = XGB[(model, task)]
        params = dict(params, tree_method="hist", device=f"cuda:{gpu}", seed=SEED, verbosity=0)
        if task == 2:
            params.update(objective="binary:logistic", eval_metric="logloss")
        else:
            params.update(objective="multi:softmax", num_class=n_cls, eval_metric="mlogloss")
        w = None
        if weighted:
            counts = np.bincount(y_tr, minlength=n_cls).astype(np.float64)
            w = ((len(y_tr) / (n_cls * counts)) ** 0.5)[y_tr].astype(np.float32)
        dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w, feature_names=feats)
        dva = xgb.DMatrix(X_va, label=y_va, feature_names=feats)
        bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")],
                        early_stopping_rounds=50, verbose_eval=200)
        rng = (0, bst.best_iteration + 1)

        def predict(X):
            p = bst.predict(xgb.DMatrix(X, feature_names=feats), iteration_range=rng)
            return (p > 0.5).astype(int) if task == 2 else p.astype(int)

        gain = bst.get_score(importance_type="gain")
        imp = np.array([gain.get(f, 0.0) for f in feats])
        imp = imp / imp.sum()
        info["config"] = {**params, "weight_power": 0.5 if weighted else None}
        info["best_iteration"] = int(bst.best_iteration)

    info["train_seconds"] = time.time() - t0
    p_va = predict(X_va)
    info["val_metrics"] = metrics(y_va, p_va)
    # Single scoring of the held-out split for the paper comparison (not used for any choice).
    X_te = np.load(D / "X_te_46.npy", mmap_mode="r")[:, cols]
    y_te = np.load(D / f"y_te_{task}.npy")
    p_te = predict(X_te)
    info["test_metrics"] = metrics(y_te, p_te)
    info["test_confusion"] = confusion_matrix(y_te, p_te, labels=range(n_cls)).tolist()
    info["test_per_class_f1"] = f1_score(y_te, p_te, average=None, labels=range(n_cls)).tolist()
    info["feature_importance"] = dict(sorted(zip(feats, map(float, imp)), key=lambda kv: -kv[1]))
    info.update(model=model, task=task, fset=fset, name=name, n_features=len(feats))
    np.save(OUT / f"{name}_pred_te.npy", p_te.astype(np.int8))
    (OUT / f"{name}.json").write_text(json.dumps(info, indent=1))
    print(name, "val", info["val_metrics"], "test", info["test_metrics"],
          f"{info['train_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
