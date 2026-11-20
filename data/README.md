# Data

No data files are committed. Everything here is reproducible from the public dataset.

## 1. CICIoT2023 (used for all reported results)

1. Request/download from the CIC site: <https://www.unb.ca/cic/datasets/iotdataset-2023.html>
   (the `CSV/` release, ~8.4 GB uncompressed, 34 sub-folders — one per class; benign is
   `Benign_Final`).
2. Place it so the folder layout is:
   ```
   <DATA_ROOT>/CSV/<ClassName>/*.csv
   ```
3. Set `DATA_ROOT` at the top of `src/preprocessing/phase1_data_pipeline.py` and run it. It
   writes a labelled parquet cache split 75/5/20 (seed 42) to `CACHE_ROOT`.

Class-name → 2/8/34-class label mappings are in `src/preprocessing/class_maps.json`; feature
list in `src/preprocessing/data_meta.json`.

## 2. Capturing your own attack traffic on AWS EC2

> **Template — fill in with the exact setup you used** (instance types, AMIs, tool versions,
> attack commands, durations). The steps below are the general procedure.

> **AWS policy:** you may only target instances you own, in your own account. AWS permits
> security testing of your own EC2 resources without pre-approval for most activities, but
> **DoS/DDoS and stress testing are prohibited** unless done under the AWS
> *Simulated Events* policy with prior approval. Keep all traffic inside a private VPC.

1. **Network** — create an isolated VPC + private subnet; security groups allowing traffic only
   between the lab instances; no internet-facing targets.
2. **Instances**
   - *Victim(s)*: Linux instances running the services you want attacked (e.g. MQTT broker,
     HTTP server, SSH) to stand in for IoT devices.
   - *Attacker*: a separate instance with the attack tools.
   - *Monitor*: either capture on the victim, or use **VPC Traffic Mirroring** to send a copy of
     the victim ENI's traffic to a monitor instance.
3. **Capture** on the victim/monitor:
   ```bash
   sudo tcpdump -i <iface> -w capture_<attack>_%Y%m%d_%H%M.pcap -G 600 -C 100
   ```
   Record one attack type per capture session (and a separate benign-only session), and log
   start/end timestamps for labelling.
4. **Feature extraction** — convert pcaps to the 39 CICIoT2023 flow features with the dataset
   authors' feature-extraction code (published alongside CICIoT2023), so columns match
   `data_meta.json`.
5. **Label** each output CSV by its capture session and place it under
   `<DATA_ROOT>/CSV/<ClassName>/`.
6. **Evaluate** — score the captured CSVs with the trained models. Treat this as a
   **file/session-grouped** test set (never mix rows from the same capture into train and test;
   see the twin-leakage caveat in the main README).
7. **Clean up** — terminate instances and delete mirror sessions/VPC to stop charges.
