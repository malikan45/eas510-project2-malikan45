"""Load the IMDb and Rotten Tomatoes files and turn them into (title, year, text, label)."""

import os
import re

import pandas as pd

import config


def clean_text(text):
    text = str(text)
    text = text.replace("See full summary", " ")   # IMDb truncation marker
    text = re.sub(r"[^A-Za-z ]", " ", text)        # keep letters only
    return re.sub(r"\s+", " ", text).strip().lower()


def pick_label(genre_string):
    """'Action, Comedy, Horror' -> 'Horror' (highest priority mapped genre)."""
    mapped = set()
    for g in str(genre_string).split(","):
        g = g.strip().lower()
        if g in config.GENRE_MAP:
            mapped.add(config.GENRE_MAP[g])
    for label in config.GENRE_PRIORITY:
        if label in mapped:
            return label
    return None


def get_year(value):
    match = re.search(r"(19|20)\d\d", str(value))
    return int(match.group()) if match else None


def load_source(path, cols, source_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. See README section 'Getting the data'.")

    raw = pd.read_csv(path)
    print(f"[{source_name}] raw rows: {len(raw)}")

    df = pd.DataFrame({
        "title": raw[cols["title"]].astype(str).str.strip(),
        "year": raw[cols["year"]].apply(get_year),
        "text": raw[cols["text"]].apply(clean_text),
        "label": raw[cols["genre"]].apply(pick_label),
    })

    df = df[df["label"].notna()]
    df = df[df["text"].str.split().str.len() >= 10]        # drop empty / "add a plot" rows
    df = df.drop_duplicates(subset=["title", "year"])      # same movie listed twice
    df["source"] = source_name

    print(f"[{source_name}] rows after cleaning: {len(df)}")
    return df.reset_index(drop=True)


def load_all(full=False):
    folder = config.FULL_DIR if full else config.SAMPLE_DIR
    imdb = load_source(os.path.join(folder, config.IMDB_FILE), config.IMDB_COLS, "imdb")
    rt = load_source(os.path.join(folder, config.RT_FILE), config.RT_COLS, "rt")
    return imdb, rt


def genre_agreement(imdb, rt):
    """For movies in both sources, how often do they get the same label?"""
    both = imdb.merge(rt, on=["title", "year"], suffixes=("_imdb", "_rt"))
    if len(both) == 0:
        return 0, float("nan")
    agree = (both["label_imdb"] == both["label_rt"]).mean()
    return len(both), agree
