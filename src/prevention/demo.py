"""Live terminal demo of the IDPS (CPU only, no extra installs).

Streams flows one by one from results/prevention/demo_flows.csv (2,000 random rows of the v4 test
split, written by idps_simulator.py), classifies each with the deployed no-IAT 8-class model,
applies policy.json and prints one coloured line per flow, with running counters at the bottom.
Rules are generated text only; nothing is executed (CICIoT2023 has no IP addresses).

usage: python demo.py [--rows 300] [--delay 0.15] [--model-dir /dev/shm/saksham/v4/prevention]
                      [--attacks-only] [--no-color]
"""
import argparse
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from idps import DEFAULT_MODEL_DIR, Detector, Policy, placeholder_ip

REPO = Path(__file__).resolve().parents[2]
COL = {"ALLOW": "\033[32m", "BLOCK": "\033[31m", "ALERT": "\033[33m", "BLOCK+ALERT": "\033[35m"}
DIM, BOLD, RESET, CLEAR = "\033[2m", "\033[1m", "\033[0m", "\r\033[K"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--delay", type=float, default=0.15, help="seconds between flows")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--flows", default=str(REPO / "results/prevention/demo_flows.csv"))
    ap.add_argument("--attacks-only", action="store_true", help="skip most DDoS rows so rarer classes show up")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()
    tty = sys.stdout.isatty()
    if a.no_color or not tty:
        for k in COL:
            COL[k] = ""
        globals().update(DIM="", BOLD="", RESET="")

    print(f"{BOLD}CICIoT2023 IDPS demo{RESET}: loading model (no-IAT 8-class XGBoost) ...", flush=True)
    det, pol = Detector(a.model_dir, nthread=4), Policy()
    flows = pd.read_csv(a.flows)
    if a.attacks_only:  # keep 1 in 10 DDoS rows so the stream is not 73% DDoS
        flows = flows[(flows["true_class"] != "DDoS") | (flows.index % 10 == 0)]
    flows = flows.head(a.rows)
    X = flows[det.features].to_numpy(float)
    print(f"{DIM}{len(flows)} flows from the v4 test split | rules are generated, not executed | "
          f"placeholder IPs 203.0.113.0/24{RESET}\n")
    print(f"{BOLD}{'time':12} {'src (placeholder)':17} {'predicted':12} {'conf':>5}  {'true':12} "
          f"{'action':12} generated rule{RESET}")

    acts, right_act, right_cls, lat = Counter(), 0, 0, 0.0
    for i in range(len(flows)):
        t0 = time.perf_counter()
        cls_id, conf = det.predict(X[i])
        lat += (time.perf_counter() - t0) * 1e3
        pred, true = det.classes[int(cls_id[0])], flows["true_class"].iat[i]
        act = pol.action(pred)
        ip = placeholder_ip(flows["flow_id"].iat[i])
        rule = pol.rule(pred, ip) or ("alert raised" if act == "ALERT" else "-")
        ok_act = act == pol.action(true)
        acts[act] += 1
        right_act += ok_act
        right_cls += pred == true
        mark = "" if ok_act else f" {BOLD}✗ wrong action{RESET}"
        line = (f"{datetime.now():%H:%M:%S.%f}"[:12] + f" {ip:17} {pred:12} {conf[0]:5.2f}  {DIM}{true:12}{RESET} "
                f"{COL[act]}{act:12}{RESET} {DIM}{rule[:90]}{RESET}{mark}")
        n = i + 1
        footer = (f"{BOLD}flows {n}{RESET} | " + " ".join(f"{COL[k]}{k} {acts[k]}{RESET}" for k in COL)
                  + f" | action acc {right_act / n:.3f} | class acc {right_cls / n:.3f} | {lat / n:.2f} ms/flow")
        # TTY: overwrite the counter line in place; piped output: plain lines, counters at the end.
        sys.stdout.write(CLEAR + line + "\n" + footer if tty else line + "\n")
        sys.stdout.flush()
        time.sleep(a.delay)
    if not tty:
        print(footer)
    print("\n\nDone. Class mistakes that still trigger the right action (e.g. DoS vs DDoS) count as correct actions.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"{RESET}\nstopped.")
