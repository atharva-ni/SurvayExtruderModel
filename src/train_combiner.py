"""
Fit the learned hybrid (src/combiner.py) and cross-validate it on author profiles
===============================================================================
    python main.py train-combiner                # fit and save to the model directory
    python main.py train-combiner --cv 5         # also save repeated 10-fold CV predictions for the
                                                 # hand-labeled papers (used by the evaluation)

Sources: the validation split (title only), LLM-labeled profile
papers (data/llm_labels.csv), and hand-labeled profile papers (data/eval_to_label.csv).
In cross-validation the hand-labeled papers are split into 10 folds; each fold is
predicted by a combiner fitted on the other sources plus the remaining nine folds,
with C and the threshold chosen inside those nine folds (nested).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from classifier import prepare_frame
from combiner import SurveyCombiner, cls_embeddings, combiner_features, weighted_prec_rec
from inference import load_model, load_config, predict_survey_proba
from text_utils import paper_text
from train import SPLIT_FILE

LLM_CSV = "data/llm_labels.csv"
EVAL_CSV = "data/eval_to_label.csv"
CV_FILE = "combiner_cv_predictions.json"
VALIDATION_SHARE = 0.25


def _proba(frame, model_path):
    tokenizer, model = load_model(model_path)
    texts = [paper_text(t, a) for t, a in zip(frame["Title"], frame["Abstract"])]
    return predict_survey_proba(texts, tokenizer, model, max_length=load_config(model_path).get("max_length", 384))


def matrix(frame, model_path):
    return combiner_features(frame, _proba(frame, model_path), cls_embeddings(frame, model_path))


def load_sources(model_path: str, llm_csv: str = LLM_CSV, eval_csv: str = EVAL_CSV) -> dict:
    split = pd.read_csv(os.path.join(model_path, SPLIT_FILE))
    # Title-only copies of the validation split teach the combiner records without an abstract,
    # which the profile papers (all with abstracts) do not cover
    val = split[split["Split"] == "val"].reset_index(drop=True).assign(Abstract="")
    llm = pd.read_csv(llm_csv, dtype={"Label": str})
    llm = llm[llm["Label"].isin(["0", "1"])].reset_index(drop=True)
    hand = pd.read_csv(eval_csv, dtype={"Label": str})
    hand = hand[hand["Label"].isin(["0", "1"])].reset_index(drop=True)

    print(f"Combiner sources: {len(val)} validation rows, {len(llm)} LLM-labeled, {len(hand)} hand-labeled")
    X_val, X_llm, X_hand = (matrix(prepare_frame(d), model_path) for d in (val, llm, hand))
    y_val = (val["Label"] == 0).astype(int).values
    y_llm = (llm["Label"] == "0").astype(int).values
    y_hand = (hand["Label"] == "0").astype(int).values
    w_llm = llm["StratumWeight"].values.astype(float)
    w_llm = w_llm / w_llm.mean()
    # Journal papers differ from profile papers; their total weight is a quarter of the LLM-labeled total
    w_val = np.full(len(y_val), VALIDATION_SHARE * w_llm.sum() / len(y_val))
    return {
        "X_fixed": np.vstack([X_val, X_llm]),
        "y_fixed": np.r_[y_val, y_llm],
        "w_fixed": np.r_[w_val, w_llm],
        "X_hand": X_hand, "y_hand": y_hand, "w_hand": hand["StratumWeight"].values.astype(float),
        "strata": hand["Stratum"].astype(str).values, "hand_ids": hand["id"].astype(str).values,
    }


def cross_validate(src: dict, reps: int = 5, seed: int = 42) -> np.ndarray:
    """Out-of-fold predictions (reps x n_hand) for the hand-labeled papers."""
    X, y, w, strata = src["X_hand"], src["y_hand"], src["w_hand"], src["strata"]
    groups = np.char.add(strata, y.astype(str))
    out = np.zeros((reps, len(y)), dtype=int)
    for r in range(reps):
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=seed + r).split(X, groups):
            comb = SurveyCombiner.fit_selected(src["X_fixed"], src["y_fixed"], src["w_fixed"],
                                               X[tr], y[tr], w[tr], strata[tr], seed=seed)
            out[r, te] = comb.predict(X[te])
        prec, rec = weighted_prec_rec(y, out[r], w)
        print(f"   CV repetition {r + 1}/{reps}: precision (rw) {100 * prec:.1f}%, recall (rw) {100 * rec:.1f}%")
    return out


def run(model_path: str = "./distilbert_survey_model", cv: int = 0) -> SurveyCombiner:
    src = load_sources(model_path)
    comb = SurveyCombiner.fit_selected(src["X_fixed"], src["y_fixed"], src["w_fixed"],
                                       src["X_hand"], src["y_hand"], src["w_hand"], src["strata"])
    comb.save(model_path)
    print(f"Combiner saved (C = {comb.C}, threshold = {comb.threshold:.2f})")
    if cv:
        oof = cross_validate(src, reps=cv)
        path = os.path.join(model_path, CV_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ids": src["hand_ids"].tolist(), "predictions": oof.tolist()}, f)
        print(f"Cross-validated predictions saved to '{path}'")
    return comb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="./distilbert_survey_model")
    parser.add_argument("--cv", type=int, default=0, help="Repetitions of 10-fold CV to report (0 = none)")
    args = parser.parse_args()
    run(args.model, args.cv)


if __name__ == "__main__":
    main()
