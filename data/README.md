# Data

No data files are committed. Everything here is reproducible from the public dataset.

## CICIoT2023 (used for all reported results)

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
