import os, sys; sys.path.insert(0, "/home/kishor_kumar/saksham/ml_pipeline"); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import guard, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import PROBS, names
yv = np.load(PROBS/"34_class/y_val.npy"); yt = np.load(PROBS/"34_class/y_test.npy")
f, ax = plt.subplots(figsize=(12, 4)); b = np.arange(35)
ax.bar(b[:-1]-0.2, np.bincount(yv, minlength=34)/len(yv), 0.4, label=f"val n={len(yv):,}")
ax.bar(b[:-1]+0.2, np.bincount(yt, minlength=34)/len(yt), 0.4, label=f"test n={len(yt):,}")
ax.set_yscale("log"); ax.set_xticks(b[:-1]); ax.set_xticklabels(names("34_class"), rotation=90, fontsize=6); ax.legend(); ax.set_ylabel("share (log)")
f.tight_layout()
guard.fig(4, "split_class_shares", f, "Val and test have matching 34-class shares (stratified-like random 75/5/20 split from prior run); 5 classes are under 0.05% of rows, so macro-F1 is dominated by rare classes.")
from sklearn.metrics import f1_score; pv = np.load(PROBS/"34_class/xgb_34_class_balanced_p050_val.npy").argmax(1); d = dict(zip(names("34_class"), f1_score(yv, pv, average=None, labels=range(34))))
f, ax = plt.subplots(figsize=(12, 4)); k = list(d); ax.bar(range(len(k)), [d[x] for x in k], color="#4a7bb7")
ax.axhline(0.70, color="k", ls="--", lw=1, label="target mF1 0.70"); ax.set_xticks(range(len(k))); ax.set_xticklabels(k, rotation=90, fontsize=6); ax.set_ylabel("val F1"); ax.legend(); f.tight_layout()
guard.fig(12, "prior_best_34class_val_f1", f, "Validation per-class F1 of the prior-run best 34-class model (XGB balanced_p050); about 12 classes sit under 0.5 F1 (SYN-flood DoS/DDoS variants and rare web attacks), which is where the macro-F1 gap to 0.70 comes from.")
