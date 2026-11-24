"""Each model is a Pipeline: TF-IDF -> classifier.
Because TF-IDF is inside the pipeline, it is fit on training data only (no leakage)."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import config


def make_models():
    def pipe(clf):
        return Pipeline([("tfidf", TfidfVectorizer(**config.TFIDF)), ("clf", clf)])

    return {
        "majority_class": pipe(DummyClassifier(strategy="most_frequent")),
        "naive_bayes": pipe(MultinomialNB()),
        "logistic_regression": pipe(LogisticRegression(random_state=config.SEED, **config.LOGREG)),
        "random_forest": pipe(RandomForestClassifier(random_state=config.SEED, **config.RANDOM_FOREST)),
    }
  