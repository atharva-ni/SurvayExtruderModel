"""
Metrics plus confusion matrix, ROC and precision-recall curves for a labeled CSV.
Plots are saved next to this script.
Usage: python visualization/Reportwithgraph.py [--input data/eval_to_label.csv] [--mode learned]
"""

import os
import argparse

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
)

from ReportMatrics import evaluate, ROOT, MODES

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot(y, is_survey, scores, mode: str, show: bool = False):
    # Confusion matrix
    cm = confusion_matrix(y, is_survey, labels=[1, 0])
    ConfusionMatrixDisplay(cm, display_labels=["Survey", "Not survey"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix ({mode})")
    plt.savefig(os.path.join(OUT_DIR, f"confusion_matrix_{mode}.png"), dpi=200, bbox_inches="tight")

    if len(set(scores)) > 2:  # curves need continuous scores
        fpr, tpr, _ = roc_curve(y, scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y, scores):.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — survey detection ({mode})")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(OUT_DIR, f"roc_{mode}.png"), dpi=200, bbox_inches="tight")

        precision, recall, _ = precision_recall_curve(y, scores)
        plt.figure()
        plt.plot(recall, precision, label=f"AP = {average_precision_score(y, scores):.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve — survey detection ({mode})")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(OUT_DIR, f"precision_recall_{mode}.png"), dpi=200, bbox_inches="tight")

    print(f"🖼️  Plots saved to '{OUT_DIR}'")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join(ROOT, "data", "eval_to_label.csv"))
    parser.add_argument("--model", default=os.path.join(ROOT, "distilbert_survey_model"))
    parser.add_argument("--mode", default="learned", choices=MODES)
    parser.add_argument("--show", action="store_true", help="Also open the plots in a window")
    args = parser.parse_args()

    _, y, is_survey, scores = evaluate(args.input, args.model, args.mode)
    plot(y, is_survey, scores, args.mode, args.show)
