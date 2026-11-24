# Movie Genre Prediction from Plot Descriptions


Predicts a movie's genre from its plot description using TF-IDF features and three classifiers (Naive Bayes, Logistic Regression, Random Forest), compared against a majority-class baseline. Models are tested within IMDb, within Rotten Tomatoes, and across sources.

## Results summary

_To be filled in after the full run. See `results/full/metrics.csv`._

## Environment

- Tested with Python 3.13 on Linux
- Packages pinned in `requirements.txt`

## Run the sample (under 1 minute on CPU)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Outputs go to `results/sample/`:
- `metrics.csv`: accuracy and macro precision/recall/F1 for every model and experiment
- `model_comparison.png`: macro-F1 bar chart
- `confusion_*.png`, `report_*.txt`: per-model confusion matrices and per-genre reports

## Getting the data (full run)

Both datasets come from Kaggle (free account required). Downloaded on: _YYYY-MM-DD_.

1. **IMDb:** "IMDb Movie Dataset: All Movies by Genre" (rajugc)
   https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre
   Unzip it, then stack the per-genre CSVs into one file:
   `python data/combine_imdb.py path/to/unzipped_folder`
2. **Rotten Tomatoes:** "Rotten Tomatoes movies and critic reviews dataset" (stefanoleone992)
   https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset
   Copy `rotten_tomatoes_movies.csv` into `data/full/`.
3. Run `python main.py --full`. Results go to `results/full/`.

`data/full/` is git-ignored. Only the small sample is committed.

### How the sample was made

`python data/make_sample.py` shuffles each full file with seed 42 and keeps up to 300 movies per genre (after genre mapping). That gives balanced classes, unlike the full data, so sample scores are not comparable to the full results.

## Data licensing

Both datasets were scraped from IMDb / Rotten Tomatoes by the Kaggle uploaders. The sample is included for non-commercial coursework only. See each Kaggle page for its license terms.
