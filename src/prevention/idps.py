"""Shared IDPS core: load the deployed model, classify flows, map classes to actions, generate rules.

The deployed model is the v4 best no-IAT 8-class XGBoost (IAT is excluded because in CICIoT2023 it
acts as a capture timestamp). Rules are generated as text only and never executed: the dataset has
no IP addresses, so every rule uses a placeholder source IP from the RFC 5737 documentation range.
"""
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path("/dev/shm/saksham/v4/prevention")
ACTIONS = ["ALLOW", "BLOCK", "ALERT", "BLOCK+ALERT"]


class Policy:
    def __init__(self, path=HERE / "policy.json"):
        cfg = json.loads(Path(path).read_text())
        self.actions = cfg["actions"]
        self.rate = cfg["rate_limit"]

    def action(self, cls):
        return self.actions[cls]

    def rule(self, cls, ip):
        """iptables command(s) for a BLOCK-type action, or '' for ALLOW/ALERT. Text only, never run."""
        act = self.actions[cls]
        if act == "BLOCK":
            name = f"idps_{cls.lower()}"
            return (f"iptables -A INPUT -s {ip} -m hashlimit --hashlimit-name {name} --hashlimit-mode srcip "
                    f"--hashlimit-above {self.rate['above']} --hashlimit-burst {self.rate['burst']} -j DROP")
        if act == "BLOCK+ALERT":
            return f"iptables -A INPUT -s {ip} -j DROP"
        return ""


def placeholder_ip(flow_id):
    """Deterministic fake source IP in 203.0.113.0/24 (TEST-NET-3)."""
    return f"203.0.113.{int(flow_id) % 254 + 1}"


class Detector:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, nthread=8):
        model_dir = Path(model_dir)
        pre = json.loads((model_dir / "preprocess.json").read_text())
        self.features = pre["features"]
        self.classes = pre["class8_order"]
        self.mean = np.asarray(pre["mean"], np.float64)
        self.scale = np.asarray(pre["scale"], np.float64)
        self.booster = xgb.Booster()
        self.booster.load_model(model_dir / "model_8c_noiat.ubj")
        self.booster.set_param({"device": "cpu", "nthread": nthread})
        self.meta = pre

    def predict_proba(self, X_raw):
        """X_raw: (n, n_features) raw feature values in self.features order."""
        Xs = ((np.asarray(X_raw, np.float64) - self.mean) / self.scale).astype(np.float32)
        return self.booster.inplace_predict(Xs)

    def predict(self, X_raw):
        p = self.predict_proba(np.atleast_2d(X_raw))
        return p.argmax(1), p.max(1)
