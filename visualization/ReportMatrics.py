"""
Print classification metrics (survey = positive class) for a labeled CSV.
Usage: python visualization/ReportMatrics.py [--input data/eval_to_label.csv] [--mode learned]
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from classifier import prepare_frame, classify_frame, MODES  # noqa: E402


def load_labeled(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    label_col = "Label" if "Label" in df.columns else "label"
    df = df[df[label_col].isin(["0", "1"])].reset_index(drop=True)
    df["is_survey"] = (df[label_col] == "0").astype(int)  # label 0 = survey
    return df


def evaluate(path: str, model_path: str, mode: str):
    df = load_labeled(path)
    frame = prepare_frame(df)
    is_survey, scores = classify_frame(frame, model_path=model_path, mode=mode)
    y = df["is_survey"].values

    print(f"\n📊 Evaluation: {mode} ({len(df)} labeled papers from '{path}')")
    print(f"Accuracy : {accuracy_score(y, is_survey):.3f}")
    print(f"Precision: {precision_score(y, is_survey, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y, is_survey, zero_division=0):.3f}")
    print(f"F1 Score : {f1_score(y, is_survey, zero_division=0):.3f}")
    print("\nClassification Report:")
    print(classification_report(y, is_survey, labels=[1, 0], target_names=["survey", "not survey"], zero_division=0))
    return df, y, np.asarray(is_survey), np.asarray(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join(ROOT, "data", "eval_to_label.csv"))
    parser.add_argument("--model", default=os.path.join(ROOT, "distilbert_survey_model"))
    parser.add_argument("--mode", default="learned", choices=MODES)
    args = parser.parse_args()
    evaluate(args.input, args.model, args.mode)
