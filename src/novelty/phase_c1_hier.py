"""C1 Hierarchical 34-class: stage1 = existing 8-class XGB (cached probs), stage2 = per-family XGB experts.
Usage: phase_c1_hier.py [SEED]. Soft routing: P(sub) = P(family) * P(sub | family); hard: argmax family -> expert."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import *
import xgboost as xgb
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
C34, C8 = names("34_class"), names("8_class"); f2c = CM["folder_to_8class"]
fam_of = np.array([C8.index(f2c[c]) if c in f2c else C8.index("Benign") for c in C34])  # 34 -> 8
log(f"family sizes: { {C8[f]: int((fam_of == f).sum()) for f in range(8)} }")
df = load_split("train", FEATURES + ["label_34class"]); Xtr = df[FEATURES].to_numpy(np.float32)
ytr = df["label_34class"].to_numpy().astype(np.int64); del df; free()
Xv, yv = load_xy("val", "34_class"); Xt, yt = load_xy("test", "34_class")
S1v = np.load(PROBS / "8_class/xgb_8_class_cfg3_weighted_val.npy"); S1t = np.load(PROBS / "8_class/xgb_8_class_cfg3_weighted_test.npy")
Pv = np.zeros((len(yv), 34), np.float32); Pt = np.zeros((len(yt), 34), np.float32)
rng = np.random.default_rng(seed); info = {}
def pred(bst, X, n):
    out = []
    for i in range(0, len(X), CHUNK):
        d = xgb.DMatrix(X[i:i+CHUNK]); p = bst.predict(d); del d
        out.append(p.reshape(-1, n) if n > 2 else np.column_stack([1 - p, p]))
    return np.vstack(out).astype(np.float32)
for f in range(8):
    subs = np.flatnonzero(fam_of == f)
    if len(subs) == 1:
        Pv[:, subs[0]] = S1v[:, f]; Pt[:, subs[0]] = S1t[:, f]; continue
    idx = []
    for s in subs:
        ic = np.flatnonzero(ytr == s); idx.append(rng.choice(ic, 300_000, replace=False) if len(ic) > 300_000 else ic)
    idx = np.concatenate(idx); remap = {s: j for j, s in enumerate(subs)}
    ya = np.array([remap[v] for v in ytr[idx]]); n = len(subs)
    mv = np.isin(yv, subs); yva = np.array([remap[v] for v in yv[mv]])
    p = dict(max_depth=8, eta=0.1, max_bin=128, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
             device="cuda:0", seed=seed, objective="multi:softprob" if n > 2 else "binary:logistic")
    if n > 2: p["num_class"] = n
    dtr = xgb.DMatrix(Xtr[idx], label=ya); dva = xgb.DMatrix(Xv[mv], label=yva); t = time.time()
    bst = xgb.train(p, dtr, 300, evals=[(dva, "val")], verbose_eval=False); tt = time.time() - t
    del dtr, dva; free()
    Ev, Et = pred(bst, Xv, n), pred(bst, Xt, n)
    Pv[:, subs] = S1v[:, [f]] * Ev; Pt[:, subs] = S1t[:, [f]] * Et
    within = float((Ev[mv].argmax(1) == yva).mean())
    info[C8[f]] = {"subtypes": [C34[s] for s in subs], "train_rows": int(len(idx)), "train_s": round(tt, 1),
                   "expert_val_acc_on_true_family": within}
    log(f"expert {C8[f]} n={n} rows={len(idx):,} {tt:.0f}s within-family val acc={within:.4f}")
    bst.save_model(str(MODELS.parent / "v3_models" / f"hier_expert_{C8[f]}_s{seed}.json")); del bst, Ev, Et; free()
del Xtr, ytr, Xv, Xt; free()
np.save(PROBS / "34_class" / f"hier_s{seed}_val.npy", Pv); np.save(PROBS / "34_class" / f"hier_s{seed}_test.npy", Pt)
def hard(P, S1):
    fam = S1.argmax(1); M = np.where(fam_of[None, :] == fam[:, None], P, -1); return M.argmax(1)
R = {"seed": seed, "experts": info, "stage1": "xgb_8_class_cfg3_weighted",
     "soft_val": metrics(yv, Pv.argmax(1), "34_class"), "hard_val": metrics(yv, hard(Pv, S1v), "34_class"),
     "soft_test": metrics(yt, Pt.argmax(1), "34_class"), "hard_test": metrics(yt, hard(Pt, S1t), "34_class")}
save_json(R, V3 / "novelty" / f"hierarchical_s{seed}.json")
for kk in ["soft_val", "hard_val", "soft_test", "hard_test"]:
    log(f"{kk}: acc={R[kk]['accuracy']:.4f} mF1={R[kk]['macro_f1']:.4f}")
