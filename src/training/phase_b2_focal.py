"""B2 focal reweighting for 34-class. Usage: phase_b2_focal.py [SEED]
1) p_correct on train from xgb_34_class_balanced_p050 (chunked, GPU), cached as .npy
2) cap 1M/class; w = sqrt-balanced class weight * focal (1-p)^2 (floor 1e-3), focal renormalized within class
   so class totals match the original balanced_p050 weighting (focal only re-ranks rows within a class)
3) XGB same config as balanced_p050 but max_bin=128; early stopping on val."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import *
import subprocess, xgboost as xgb
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42; k = 34; GAMMA = 2.0; FLOOR = 1e-3
tag = f"xgb_34_class_focal_s{seed}"; out = RES / "34_class" / "focal" / tag
X, y = load_xy("train", "34_class"); pc_path = PROBS / "34_class" / "p050_train_pcorrect.npy"
if pc_path.exists(): pc = np.load(pc_path)
else:
    b = xgb.Booster(); b.load_model(str(MODELS / "xgb_34_class_balanced_p050.json")); b.set_param({"device": "cuda:0"})
    pc = np.empty(len(y), np.float32)
    for i in range(0, len(y), CHUNK):
        d = xgb.DMatrix(X[i:i+CHUNK]); P = softmax(b.predict(d, output_margin=True).reshape(-1, k)); del d
        pc[i:i+CHUNK] = P[np.arange(len(P)), y[i:i+CHUNK]]
    del b; free(); np.save(pc_path, pc)
log(f"train p_correct: mean={pc.mean():.4f}  share>0.99={np.mean(pc > 0.99):.3f}")
counts = np.bincount(y, minlength=k); cw = np.sqrt(len(y) / (k * counts))  # balanced_p050 weights (full train)
rng = np.random.default_rng(seed); idx = []
for c in range(k):
    ic = np.flatnonzero(y == c); idx.append(rng.choice(ic, 1_000_000, replace=False) if len(ic) > 1_000_000 else ic)
idx = np.sort(np.concatenate(idx)); X, y, pc = X[idx], y[idx], pc[idx]; del idx; free()
foc = np.maximum((1 - pc) ** GAMMA, FLOOR); w = np.empty(len(y), np.float32)
for c in range(k):
    m = y == c; w[m] = cw[c] * counts[c] / m.sum() * foc[m] / foc[m].mean()  # keep class total = full-train balanced total
w /= w.mean(); log(f"rows={len(y):,}")
Xv, yv = load_xy("val", "34_class")
p = dict(max_depth=12, learning_rate=0.05, max_bin=128, subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
         num_class=k, eval_metric="mlogloss", tree_method="hist", device="cuda:0", seed=seed)
dtr = xgb.DMatrix(X, label=y, weight=w); dv = xgb.DMatrix(Xv, label=yv); del X; free(); t = time.time()
bst = xgb.train(p, dtr, 3000, evals=[(dv, "val")], early_stopping_rounds=50, verbose_eval=100); tt = time.time() - t
vram = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0", shell=True, capture_output=True, text=True).stdout.strip()
del dtr, dv; free()
(MODELS.parent / "v3_models").mkdir(exist_ok=True); bst.save_model(str(MODELS.parent / "v3_models" / f"{tag}.json"))
def probs(Xs):
    return np.vstack([bst.predict(xgb.DMatrix(Xs[i:i+CHUNK]), iteration_range=(0, bst.best_iteration + 1)).reshape(-1, k)
                      for i in range(0, len(Xs), CHUNK)]).astype(np.float32)
Pv = probs(Xv); del Xv; free(); Xt, yt = load_xy("test", "34_class"); Pt = probs(Xt); del Xt; free()
np.save(PROBS / "34_class" / f"{tag}_val.npy", Pv); np.save(PROBS / "34_class" / f"{tag}_test.npy", Pt)
R = {"model": "xgboost_focal", "config": {**p, "gamma": GAMMA, "floor": FLOOR, "cap_per_class": 1_000_000,
     "class_weight": "sqrt-balanced (full-train counts)", "focal_source": "xgb_34_class_balanced_p050"},
     "best_iteration": int(bst.best_iteration), "train_time_s": round(tt, 1), "vram_used_MiB_gpu0_after_fit": vram,
     "val": metrics(yv, Pv.argmax(1), "34_class"), "test": metrics(yt, Pt.argmax(1), "34_class")}
save_json(R, out / "results.json")
log(f"{tag}: val mF1={R['val']['macro_f1']:.4f} acc={R['val']['accuracy']:.4f} | test mF1={R['test']['macro_f1']:.4f} "
    f"acc={R['test']['accuracy']:.4f} | {tt:.0f}s it={R['best_iteration']} vram={vram}")
del bst; free()
