"""Usage: cache_probs.py TASK MODEL_FILE SPLITS(comma)  -> v3/probs/TASK/<stem>_<split>.npy"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import *
task, mfile, splits = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
k = NCLS[task]; stem = mfile.rsplit(".", 1)[0]; od = PROBS / task; od.mkdir(parents=True, exist_ok=True)
for s in splits:
    out = od / f"{stem}_{s}.npy"
    if out.exists(): log(f"skip {out.name}"); continue
    X, y = load_xy(s, task); np.save(od / f"y_{s}.npy", y)
    t = time.time(); P = predict_probs(MODELS / mfile, X, k); del X; free()
    np.save(out, P); m = metrics(y, P.argmax(1), task)
    log(f"{stem} {s}: n={len(y)} acc={m['accuracy']:.6f} mF1={m['macro_f1']:.6f} ({time.time()-t:.0f}s)")
    del P; free()
