import glob
import os
import sys

import pandas as pd

folder = sys.argv[1]
files = glob.glob(os.path.join(folder, "*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"stacked {len(files)} files -> {len(df)} rows")

df = df.drop_duplicates(subset="movie_id")
print(f"after removing repeats: {len(df)} rows")

os.makedirs("data/full", exist_ok=True)
df.to_csv("data/full/imdb.csv", index=False)
print("wrote data/full/imdb.csv")
g 