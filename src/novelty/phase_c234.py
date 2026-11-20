"""C2/C3/C4: reuse prior-run novelty results (cited) + new DoS/DDoS confusion and 7-class Flood merge (CPU)."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import *
N = IDS / "novelty"; OUT = V3 / "novelty"; C8 = names("8_class")
src = lambda n: json.load(open(N / n / "results.json"))
save_json({"source": str(N / "N2_protocol_ladder/results.json"), "note": "Both protocols already evaluated in prior run; not re-run (C2 condition not triggered).",
           "results": src("N2_protocol_ladder")}, OUT / "protocol_ladder.json")
save_json({"source": str(N / "N5_window_size/results.json"), "note": "Prior run N5 on window-group classification; reused, not re-run.",
           "results": src("N5_window_size")}, OUT / "window_artifact.json")
R = {"binary_dos_vs_ddos_prior_run": {"source": str(N / "N3_dos_vs_ddos/results.json"), **{k: v for k, v in src("N3_dos_vs_ddos").items()}}}
D, S = C8.index("DDoS"), C8.index("DoS")
for split in ["val", "test"]:
    y = np.load(PROBS / "8_class" / f"y_{split}.npy").astype(np.int64)
    p = np.load(PROBS / "8_class" / f"xgb_8_class_cfg3_weighted_{split}.npy").argmax(1)
    m8 = metrics(y, p, "8_class"); cm = np.array(m8["confusion_matrix"])
    R[f"{split}_8class_dos_ddos_cells"] = {"DDoS->DDoS": int(cm[D, D]), "DDoS->DoS": int(cm[D, S]),
                                            "DoS->DoS": int(cm[S, S]), "DoS->DDoS": int(cm[S, D]),
                                            "share_of_all_8class_errors": float((cm[D, S] + cm[S, D]) / (cm.sum() - np.trace(cm)))}
    mp = np.array([j if j < S else j - 1 for j in [D if i == S else i for i in range(8)]])
    y7, p7 = mp[y], mp[p]; n7 = [c if c != "DDoS" else "Flood(DoS+DDoS)" for c in C8 if c != "DoS"]
    from sklearn.metrics import f1_score, accuracy_score
    pf = f1_score(y7, p7, average=None, labels=range(7), zero_division=0)
    R[f"{split}_8class"] = {"accuracy": m8["accuracy"], "macro_f1": m8["macro_f1"], "per_class_f1": m8["per_class_f1"]}
    R[f"{split}_7class_flood"] = {"accuracy": float(accuracy_score(y7, p7)), "macro_f1": float(pf.mean()),
                                  "per_class_f1": dict(zip(n7, map(float, pf)))}
    log(f"{split}: 8c acc={m8['accuracy']:.4f} mF1={m8['macro_f1']:.4f} -> 7c acc={R[f'{split}_7class_flood']['accuracy']:.4f} "
        f"mF1={pf.mean():.4f}; DoS<->DDoS share of errors={R[f'{split}_8class_dos_ddos_cells']['share_of_all_8class_errors']:.3f}")
save_json(R, OUT / "dos_ddos.json")
