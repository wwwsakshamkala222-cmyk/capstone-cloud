import os, sys; sys.path.insert(0, "/home/kishor_kumar/saksham/ml_pipeline"); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import guard, glob, pandas as pd
ps = sorted(glob.glob("/dev/shm/saksham/ids_cache/train/*.parquet"))
df = pd.read_parquet(ps[0]).sample(500_000, random_state=0).drop(columns=["label_2class", "label_8class"])
guard.profile(df, target="label_34class")
guard.eda_figures(df, target="label_34class")
print("ok", df.shape)
