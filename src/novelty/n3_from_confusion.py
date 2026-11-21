"""N3 (DoS vs DDoS) computed from a saved 8-class confusion matrix (rows = true, cols = predicted).

Outputs the same quantities as results/novelty/v3/dos_ddos.json: DoS/DDoS cells, their share of all errors,
8-class acc/macro-F1, and acc/macro-F1 after merging DoS+DDoS into one 'Flood' class (same predictions).
Usage: n3_from_confusion.py <results.json> [<out.json>]
"""
import json, sys
import numpy as np


def f1s(cm):
    tp = np.diag(cm).astype(float)
    p = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    r = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)


def analyse(cm, names):
    cm = np.asarray(cm, dtype=np.int64)
    i, j = names.index("DDoS"), names.index("DoS")
    err = cm.sum() - np.trace(cm)
    cells = {"DDoS->DDoS": int(cm[i, i]), "DDoS->DoS": int(cm[i, j]), "DoS->DoS": int(cm[j, j]),
             "DoS->DDoS": int(cm[j, i]), "share_of_all_8class_errors": float((cm[i, j] + cm[j, i]) / err)}
    keep = [k for k in range(len(names)) if k != j]
    m = cm.copy(); m[i, :] += m[j, :]; m[:, i] += m[:, j]
    m = m[np.ix_(keep, keep)]
    return cells, {"accuracy": float(np.trace(cm) / cm.sum()), "macro_f1": float(f1s(cm).mean())}, \
        {"accuracy": float(np.trace(m) / m.sum()), "macro_f1": float(f1s(m).mean())}


if __name__ == "__main__":
    d = json.load(open(sys.argv[1]))
    out = {"source": sys.argv[1], "method": d["method"]}
    for split in ("val", "test"):
        tm = d[f"{split}_metrics"]
        names = list(tm["per_class"].keys())
        c, e8, f7 = analyse(tm["confusion_matrix"], names)
        out[f"{split}_8class_dos_ddos_cells"], out[f"{split}_8class"], out[f"{split}_7class_flood"] = c, e8, f7
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], "w"), indent=2)
