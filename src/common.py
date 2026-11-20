import os, json, glob, gc, ctypes, time
os.environ.setdefault("OMP_NUM_THREADS", "16"); os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
CACHE = Path("/dev/shm/saksham/ids_cache"); MODELS = CACHE / "models"
IDS = Path("/dev/shm/saksham/ids"); V3 = IDS / "v3"; PROBS = V3 / "probs"; RES = V3 / "results"
CM = json.load(open(IDS / "src/class_maps.json")); DM = json.load(open(IDS / "src/data_meta.json"))
FEATURES = DM["features"]; CHUNK = 250_000
LABEL = {"2_class": "label_2class", "8_class": "label_8class", "34_class": "label_34class"}
NCLS = {"2_class": 2, "8_class": 8, "34_class": 34}
def names(task):
    return {"2_class": ["Benign", "Attack"], "8_class": CM["class8_order"], "34_class": CM["class34_order"]}[task]
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def free():
    gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
def load_split(split, cols=None):
    ps = sorted(glob.glob(str(CACHE / split / "*.parquet")))
    return pd.concat([pd.read_parquet(p, columns=cols) for p in ps], ignore_index=True)
def load_xy(split, task):
    df = load_split(split, FEATURES + [LABEL[task]])
    y = df[LABEL[task]].to_numpy().astype(np.int64); X = df[FEATURES].to_numpy(np.float32); del df; free()
    return X, y
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return (e / e.sum(1, keepdims=True)).astype(np.float32)
def predict_probs(path, X, k):
    path = str(path)
    if path.endswith(".cbm"):
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(); m.load_model(path)
        out = [m.predict_proba(X[i:i+CHUNK]).astype(np.float32) for i in range(0, len(X), CHUNK)]
    elif path.endswith(".txt"):
        import lightgbm as lgb
        m = lgb.Booster(model_file=path); out = []
        for i in range(0, len(X), CHUNK):
            p = m.predict(X[i:i+CHUNK]).astype(np.float32)
            out.append(np.column_stack([1-p, p]) if p.ndim == 1 else p)
    else:
        import xgboost as xgb
        m = xgb.Booster(); m.load_model(path); m.set_param({"device": "cuda:0"}); out = []
        for i in range(0, len(X), CHUNK):
            d = xgb.DMatrix(X[i:i+CHUNK]); z = m.predict(d, output_margin=True); del d
            if k == 2:
                p = (1 / (1 + np.exp(-z))).astype(np.float32); out.append(np.column_stack([1-p, p]))
            else:
                out.append(softmax(z.reshape(-1, k)))
    del m; free()
    return np.vstack(out)
def metrics(y, pred, task):
    k = NCLS[task]; nm = names(task)
    pf = f1_score(y, pred, average=None, labels=range(k), zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(pf.mean()),
            "per_class_f1": {nm[i]: float(pf[i]) for i in range(k)},
            "confusion_matrix": confusion_matrix(y, pred, labels=range(k)).tolist()}
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True); json.dump(obj, open(path, "w"), indent=1)
