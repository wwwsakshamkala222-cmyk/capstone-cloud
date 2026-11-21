"""Save 2/8/34-class row counts of the Kaggle 46-feature subset (merged train+validation+test files)."""
import json
from pathlib import Path

import pandas as pd

from prep_v4 import CLASS_MAPS, RENAME

cm = json.loads(CLASS_MAPS.read_text())
lab = pd.read_parquet("/dev/shm/saksham/v4/merged_raw.parquet", columns=["label"])["label"].replace(RENAME)
c34 = lab.value_counts()
c8 = lab.map(cm["folder_to_8class"]).value_counts()
out = {"source": "Kaggle himadri07/ciciot2023 (train+validation+test merged)", "n_rows": int(len(lab)),
       "class2": {"Benign": int((lab == "Benign_Final").sum()), "Attack": int((lab != "Benign_Final").sum())},
       "class8": {k: int(v) for k, v in c8.items()}, "class34": {k: int(v) for k, v in c34.items()}}
p = Path(__file__).resolve().parents[2] / "results/v4_46feat/class_counts.json"
p.write_text(json.dumps(out, indent=1))
print(out["n_rows"], out["class2"], out["class8"])
