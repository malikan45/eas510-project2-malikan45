"""Metrics and plots."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, precision_recall_fscore_support)


def score(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(p, 4),
        "macro_recall": round(r, 4),
        "macro_f1": round(f1, 4),
    }


def save_report(y_true, y_pred, path):
    with open(path, "w") as f:
        f.write(classification_report(y_true, y_pred, zero_division=0))


def save_confusion(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, xticks_rotation=45,
                                            normalize="true", values_format=".2f", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_comparison(results, path):
    """Grouped bar chart of macro-F1 per model per experiment."""
    table = results.pivot(index="model", columns="experiment", values="macro_f1")
    ax = table.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_ylabel("Macro F1 (higher is better)")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison by experiment")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
