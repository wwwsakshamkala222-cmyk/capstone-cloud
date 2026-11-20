"""B1 CatBoost GPU. Usage: phase_b1_catboost.py TASK CFG(c1|c2) WEIGHTS(bal|none) [SEED]
Train cap 1M rows/class (stratified). Early stopping on val. Saves .cbm, val/test probs, results.json.
Probs are prior-corrected back to the full-train class distribution (undo the cap's prior shift)."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import *
import subprocess
from catboost import CatBoostClassifier, Pool

task, cfg, wmode = sys.argv[1], sys.argv[2], sys.argv[3]; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
k = NCLS[task]; CFG = {"c1": dict(depth=10, learning_rate=0.08, iterations=1500),
                       "c2": dict(depth=8, learning_rate=0.05, iterations=2000)}[cfg]
tag = f"cb_{task}_{cfg}_{wmode}_s{seed}" + (f"_b{os.environ['CB_BORDERS']}" if "CB_BORDERS" in os.environ else ""); out = RES / task / "catboost" / tag; out.mkdir(parents=True, exist_ok=True)
log(f"== {tag} {CFG}")
X, y = load_xy("train", task); full_prior = np.bincount(y, minlength=k) / len(y)
rng = np.random.default_rng(seed); idx = []
for c in range(k):
    ic = np.flatnonzero(y == c)
    idx.append(rng.choice(ic, 1_000_000, replace=False) if len(ic) > 1_000_000 else ic)
idx = np.sort(np.concatenate(idx)); X, y = X[idx], y[idx]; del idx; free()
samp_prior = np.bincount(y, minlength=k) / len(y); log(f"train rows after cap: {len(y):,}")
# effective training prior: uniform if balanced weights, else the capped sample prior
cw = (1.0 / (k * np.maximum(samp_prior, 1e-12))) if wmode == "bal" else np.ones(k)
eff_prior = samp_prior * cw; eff_prior /= eff_prior.sum()
Xv, yv = load_xy("val", task)
# eval weights reweight val to the effective training prior so early stopping isn't fooled by prior shift
ew = (eff_prior / np.maximum(np.bincount(yv, minlength=k) / len(yv), 1e-12))[yv]
m = CatBoostClassifier(**CFG, task_type="GPU", devices="0", border_count=int(os.environ.get("CB_BORDERS", 128)), gpu_ram_part=0.11,
                       loss_function="Logloss" if k == 2 else "MultiClass", od_type="Iter", od_wait=100,
                       eval_metric="TotalF1:average=Macro", metric_period=25,
                       use_best_model=True, random_seed=seed, verbose=100, thread_count=16)
t = time.time(); m.fit(Pool(X, y, weight=cw[y]), eval_set=Pool(Xv, yv, weight=ew)); tt = time.time() - t
vram = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0", shell=True,
                      capture_output=True, text=True).stdout.strip()
del X, y; free()
MD = MODELS.parent / "v3_models"; MD.mkdir(exist_ok=True); m.save_model(str(MD / f"{tag}.cbm"))
def probs(Xs):
    P = np.vstack([m.predict_proba(Xs[i:i+CHUNK]) for i in range(0, len(Xs), CHUNK)]).astype(np.float32)
    P = P * (full_prior / eff_prior); return (P / P.sum(1, keepdims=True)).astype(np.float32)
Pv = probs(Xv); del Xv; free()
Xt, yt = load_xy("test", task); Pt = probs(Xt); del Xt; free()
np.save(PROBS / task / f"{tag}_val.npy", Pv); np.save(PROBS / task / f"{tag}_test.npy", Pt)
R = {"model": "catboost", "task": task, "config": {**CFG, "border_count": int(os.environ.get("CB_BORDERS", 128)), "weights": wmode, "seed": seed,
     "cap_per_class": 1_000_000, "prior_correction": True, "early_stop_metric": "TotalF1:average=Macro (val, reweighted to train prior)", "od_wait": 100}, "best_iteration": int(m.get_best_iteration()),
     "train_time_s": round(tt, 1), "vram_used_MiB_gpu0_after_fit": vram,
     "val": metrics(yv, Pv.argmax(1), task), "test": metrics(yt, Pt.argmax(1), task)}
save_json(R, out / "results.json")
log(f"{tag}: val acc={R['val']['accuracy']:.5f} mF1={R['val']['macro_f1']:.4f} | test acc={R['test']['accuracy']:.5f} "
    f"mF1={R['test']['macro_f1']:.4f} | {tt:.0f}s iters={R['best_iteration']} vram={vram}MiB")
del m; free()
