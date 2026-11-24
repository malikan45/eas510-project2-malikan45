import argparse
import os
import random
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.data import genre_agreement, load_all
from src.evaluate import save_comparison, save_confusion, save_report, score
from src.models import make_models


def split(df):
    return train_test_split(df, test_size=config.TEST_SIZE,
                            stratify=df["label"], random_state=config.SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use full data in data/full/")
    args = parser.parse_args()

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    start = time.time()

    out_dir = "results/full" if args.full else "results/sample"
    os.makedirs(out_dir, exist_ok=True)

    # ---- data ----
    imdb, rt = load_all(full=args.full)
    n_both, agree = genre_agreement(imdb, rt)
    print(f"Movies in both sources: {n_both}, same label in both: {agree:.1%}")

    imdb_train, imdb_test = split(imdb)
    rt_train, rt_test = split(rt)

    # For the cross-source test, drop RT test movies that also appear in IMDb train
    seen = set(zip(imdb_train["title"], imdb_train["year"]))
    rt_test_cross = rt_test[[(t, y) not in seen for t, y in zip(rt_test["title"], rt_test["year"])]]

    experiments = {
        "imdb_to_imdb": (imdb_train, imdb_test),
        "rt_to_rt": (rt_train, rt_test),
        "imdb_to_rt": (imdb_train, rt_test_cross),
    }

    # ---- train + evaluate ----
    rows = []
    for exp_name, (train, test) in experiments.items():
        print(f"\n=== {exp_name}: train={len(train)}, test={len(test)} ===")
        for model_name, model in make_models().items():
            model.fit(train["text"], train["label"])
            pred = model.predict(test["text"])

            metrics = score(test["label"], pred)
            rows.append({"experiment": exp_name, "model": model_name,
                         "n_train": len(train), "n_test": len(test), **metrics})
            print(f"  {model_name:20s} acc={metrics['accuracy']:.3f}  macro_f1={metrics['macro_f1']:.3f}")

            tag = f"{exp_name}_{model_name}"
            save_report(test["label"], pred, os.path.join(out_dir, f"report_{tag}.txt"))
            if model_name != "majority_class":
                save_confusion(test["label"], pred, tag, os.path.join(out_dir, f"confusion_{tag}.png"))

    # ---- save summary ----
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    save_comparison(results, os.path.join(out_dir, "model_comparison.png"))

    print(f"\nSaved results to {out_dir}/  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
