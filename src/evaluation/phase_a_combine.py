"""Phase A: threshold tuning (A1), calibration (A2), ensembles (A3) on cached probs.
Usage: phase_a_combine.py TASK stem1,stem2,...   (first stem = best single model)
Selection strictly by validation; val-fitted methods scored by 2-fold CV on val."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import *
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

task, stems = sys.argv[1], sys.argv[2].split(","); k = NCLS[task]; D = PROBS / task
yv, yt = np.load(D / "y_val.npy").astype(np.int64), np.load(D / "y_test.npy").astype(np.int64)
Pv = {s: np.load(D / f"{s}_val.npy") for s in stems}; Pt = {s: np.load(D / f"{s}_test.npy") for s in stems}

def mf1(y, p):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k); tp = np.diag(cm)
    d = cm.sum(0) + cm.sum(1); return float(np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0)))

def tune(P, y, passes=3):
    w = np.ones(k, np.float32); grid = np.round(np.arange(0.3, 3.01, 0.1), 2); best = mf1(y, P.argmax(1))
    for _ in range(passes):
        imp = False
        for c in range(k):
            for g in grid:
                w2 = w.copy(); w2[c] = g; s = mf1(y, (P * w2).argmax(1))
                if s > best + 1e-7: best, w, imp = s, w2, True
        if not imp: break
    return w, best

def cv_score(fit_apply, P, y):
    pred = np.empty_like(y)
    for a, b in StratifiedKFold(2, shuffle=True, random_state=0).split(P, y):
        pred[b] = fit_apply(P[a], y[a], P[b]).argmax(1)
    return mf1(y, pred)

def calib(method):
    def f(Pa, ya, Pb):
        out = np.zeros_like(Pb)
        for c in range(k):
            t = (ya == c).astype(int)
            if t.sum() == 0 or t.sum() == len(t): continue
            if method == "isotonic":
                out[:, c] = IsotonicRegression(out_of_bounds="clip").fit(Pa[:, c], t).predict(Pb[:, c])
            else:
                lr = LogisticRegression(max_iter=300).fit(np.log(np.clip(Pa[:, [c]], 1e-7, 1)), t)
                out[:, c] = lr.predict_proba(np.log(np.clip(Pb[:, [c]], 1e-7, 1)))[:, 1]
        return out / np.maximum(out.sum(1, keepdims=True), 1e-9)
    return f

STACK_CAP = int(os.environ.get("STACK_CAP", 0))  # per-class row cap for the stacker fit (0 = no cap)
def stack(Pa, ya, Pb):
    full = np.bincount(ya, minlength=k) / len(ya)
    if STACK_CAP:
        r = np.random.default_rng(0)
        sel = np.concatenate([r.choice(np.flatnonzero(ya == c), min(STACK_CAP, (ya == c).sum()), replace=False)
                              for c in range(k) if (ya == c).any()])
        Pa, ya = Pa[sel], ya[sel]
    lr = LogisticRegression(max_iter=1000).fit(np.log(np.clip(Pa, 1e-7, 1)), ya)
    P = np.zeros((len(Pb), k), np.float32); P[:, lr.classes_] = lr.predict_proba(np.log(np.clip(Pb, 1e-7, 1)))
    if STACK_CAP:  # undo the cap's prior shift
        samp = np.bincount(ya, minlength=k) / len(ya); P *= np.where(samp > 0, full / np.maximum(samp, 1e-12), 0)
        P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
    return P

def tuned(Pa, ya, Pb): return Pb * tune(Pa, ya, passes=2)[0]

R = {"task": task, "models": stems, "single_val_mF1": {s: mf1(yv, Pv[s].argmax(1)) for s in stems}}
base = stems[0]; cands = {"baseline": (R["single_val_mF1"][base], Pt[base])}
log(f"baseline {base} val mF1={cands['baseline'][0]:.4f}")
w, _ = tune(Pv[base], yv); R["A1_multipliers"] = dict(zip(names(task), map(float, w)))
cands["A1_threshold"] = (cv_score(tuned, Pv[base], yv), Pt[base] * w); log(f"A1 cv-val={cands['A1_threshold'][0]:.4f}")
for m in ["isotonic", "sigmoid"]:
    f = calib(m); cands[f"A2_{m}"] = (cv_score(f, Pv[base], yv), f(Pv[base], yv, Pt[base]))
    log(f"A2 {m} cv-val={cands[f'A2_{m}'][0]:.4f}")
if len(stems) > 1:
    n = len(stems)
    cands["A3_soft"] = (mf1(yv, (sum(Pv.values()) / n).argmax(1)), sum(Pt.values()) / n)
    wt = np.array([R["single_val_mF1"][s] for s in stems]); wt /= wt.sum()
    wv = sum(a * Pv[s] for a, s in zip(wt, stems)); wtt = sum(a * Pt[s] for a, s in zip(wt, stems))
    cands["A3_weighted"] = (mf1(yv, wv.argmax(1)), wtt)
    w2, _ = tune(wv, yv); cands["A3_weighted+A1"] = (cv_score(tuned, wv, yv), wtt * w2)
    Hv = np.hstack([Pv[s] for s in stems]); Ht = np.hstack([Pt[s] for s in stems])
    cands["A3_stack"] = (cv_score(stack, Hv, yv), stack(Hv, yv, Ht))
    w3, _ = tune(stack(Hv, yv, Hv), yv)
    cands["A3_stack+A1"] = (cv_score(lambda a, ya, b: tuned(stack(a, ya, a), ya, stack(a, ya, b)), Hv, yv),
                            stack(Hv, yv, Ht) * w3)
    for c in ["A3_soft", "A3_weighted", "A3_weighted+A1", "A3_stack", "A3_stack+A1"]: log(f"{c} val={cands[c][0]:.4f}")
R["candidates"] = {}
for c, (v, P) in cands.items():
    m = metrics(yt, P.argmax(1), task); m["val_mF1_selection"] = v; R["candidates"][c] = m
best = max(cands, key=lambda c: cands[c][0]); R["selected_by_val"] = best; R["selected_test"] = R["candidates"][best]
OUTN = sys.argv[3] if len(sys.argv) > 3 else "phase_a"; np.save(D / f"{OUTN}_best_test.npy", cands[best][1])
log(f"SELECTED {best}: test acc={R['selected_test']['accuracy']:.4f} mF1={R['selected_test']['macro_f1']:.4f} "
    f"(baseline test mF1={R['candidates']['baseline']['macro_f1']:.4f})")
save_json(R, RES / task / OUTN / "results.json")
