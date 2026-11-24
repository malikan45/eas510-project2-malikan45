

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data import pick_label


def sample_file(name, genre_col):
    df = pd.read_csv(os.path.join(config.FULL_DIR, name))
    df["_label"] = df[genre_col].apply(pick_label)
    df = df[df["_label"].notna()]
    # shuffle, then keep the first N rows of each genre
    df = df.sample(frac=1, random_state=config.SEED)
    out = df.groupby("_label").head(config.SAMPLE_PER_GENRE).drop(columns="_label")
    out.to_csv(os.path.join(config.SAMPLE_DIR, name), index=False)
    print(f"{name}: wrote {len(out)} rows")


if __name__ == "__main__":
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    sample_file(config.IMDB_FILE, config.IMDB_COLS["genre"])
    sample_file(config.RT_FILE, config.RT_COLS["genre"])
