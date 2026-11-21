#!/usr/bin/env python3
"""Multi-seed check for the 8-class headline model, xgboost_balanced_deep (val-selected).

Same protocol as multi_seed_eval_remaining.py (seeds 42/123/456, same splits, metrics on test), with the
training settings of the original run in src/training/phase2_extra.py: depth 14, lr 0.05, max_bin 512,
subsample/colsample 0.8, sqrt (power 0.5) class weights, 5000 rounds, early stopping 100 on val.
Usage: multi_seed_8c_balanced_deep.py <seed> <gpu>   (one process per seed)
       multi_seed_8c_balanced_deep.py combine
"""
import sys, json, time, os
from pathlib import Path
OUT = Path("/dev/shm/saksham/ids/results/8_class/BEST")

if sys.argv[1] == "combine":
    import numpy as np
    ms = [json.load(open(OUT / f"ms_bd_seed{s}.json")) for s in (42, 123, 456)]
    names = list(ms[0]["per_class_f1"])
    res = {"method": "xgboost_balanced_deep", "task": "8_class", "seeds": [42, 123, 456],
           "per_seed_metrics": ms,
           "mean_accuracy": float(np.mean([m["accuracy"] for m in ms])),
           "std_accuracy": float(np.std([m["accuracy"] for m in ms])),
           "mean_macro_f1": float(np.mean([m["macro_f1"] for m in ms])),
           "std_macro_f1": float(np.std([m["macro_f1"] for m in ms])),
           "mean_per_class_f1": {c: {"mean": float(np.mean([m["per_class_f1"][c] for m in ms])),
                                     "std": float(np.std([m["per_class_f1"][c] for m in ms]))} for c in names},
           "script": "src/evaluation/multi_seed_8c_balanced_deep.py"}
    json.dump(res, open(OUT / "multi_seed_results.json", "w"), indent=2)
    print(res["mean_accuracy"], res["std_accuracy"], res["mean_macro_f1"], res["std_macro_f1"])
    sys.exit()

seed, gpu = int(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(Path(__file__).parent))
import multi_seed_eval_remaining as M  # sets CUDA_VISIBLE_DEVICES=0 on import; override below
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
import numpy as np

t0 = time.time()
tr, va, te = M.load_split("train"), M.load_split("val"), M.load_split("test")
F = M.FEATURES
Xtr, ytr = tr[F].values.astype(np.float32), tr["label_8class"].values.astype(np.int32)
Xva, yva = va[F].values.astype(np.float32), va["label_8class"].values.astype(np.int32)
Xte, yte = te[F].values.astype(np.float32), te["label_8class"].values.astype(np.int32)
del tr, va, te
counts = np.maximum(np.bincount(ytr, minlength=8).astype(np.float64), 1)
w = ((len(ytr) / (8 * counts)) ** 0.5)[ytr].astype(np.float32)
cfg = {"max_depth": 14, "learning_rate": 0.05, "max_bin": 512, "subsample": 0.8, "colsample_bytree": 0.8,
       "n_estimators": 5000, "early_stopping_rounds": 100}
m = M.retrain_xgb_with_seed(Xtr, ytr, Xva, yva, Xte, yte, 8, M.CLASS8_ORDER, cfg, seed, sample_weights=w)
m["seconds"] = time.time() - t0
json.dump(m, open(OUT / f"ms_bd_seed{seed}.json", "w"), indent=2)
print(seed, m["accuracy"], m["macro_f1"], m["seconds"])
