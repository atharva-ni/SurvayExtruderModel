"""
Learned Hybrid (profile-fitted)
===============================
Logistic regression over the DistilBERT survey logit, metadata features, and
DistilBERT's [CLS] representation. It is fitted on three sources:

  * the validation split of the training dataset (venue labels), title only,
    so that records without an abstract are covered; its total weight is a
    quarter of the LLM-labeled papers' total;
  * LLM-labeled papers from author profiles (data/llm_labels.csv);
  * the hand-labeled author-profile papers (data/eval_to_label.csv).

Author profiles contain magazine articles and survey-like research papers that
the venue-labeled journals do not; the profile papers teach the combiner where
the survey/research boundary lies for them. Within a source, rows are weighted
by their sampling stratum; the LLM- and hand-labeled weights are normalized to a mean of 1.
The regularization strength is chosen by cross-validation on the hand-labeled
papers, and the threshold maximizes their stratum-weighted F1.
"""

import os
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hybrid import FEATURES, build_features
from inference import load_model, get_device, load_config
from text_utils import MAGAZINE_VENUE, paper_text

COMBINER_FILE = "survey_combiner.joblib"
FLAG_FEATURES = ["magazine", "arxiv", "type_review", "this_article"]
C_GRID = (0.01, 0.1, 1.0)
THRESHOLDS = np.linspace(0.05, 0.95, 91)


# ============================================================================
# FEATURES
# ============================================================================

def cls_embeddings(frame: pd.DataFrame, model_path: str, batch_size: int = 32) -> np.ndarray:
    """DistilBERT's final [CLS] hidden state for each paper (title + abstract)."""
    tokenizer, model = load_model(model_path)
    max_length = load_config(model_path).get("max_length", 384)
    texts = [paper_text(t, a) for t, a in zip(frame["Title"], frame["Abstract"])]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(texts[i:i + batch_size], truncation=True, padding=True, max_length=max_length,
                              return_tensors="pt").to(get_device())
            out.append(model.distilbert(**batch).last_hidden_state[:, 0, :].float().cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, model.config.dim))


def combiner_features(frame: pd.DataFrame, survey_proba: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    X = build_features(frame, survey_proba)[FEATURES].copy()
    X["magazine"] = frame["Venue"].map(lambda v: int(bool(MAGAZINE_VENUE.search(v)))).values
    X["arxiv"] = frame["Venue"].str.contains("arxiv", case=False).astype(int).values
    X["type_review"] = frame["Type"].str.contains(r"\breview\b", case=False).astype(int).values
    X["this_article"] = frame["Abstract"].str.contains(r"\bthis (?:article|magazine)\b", case=False).astype(int).values
    return np.hstack([X.values.astype(float), embeddings])


# ============================================================================
# WEIGHTED METRICS
# ============================================================================

def weighted_prec_rec(y: np.ndarray, p: np.ndarray, w: np.ndarray):
    tp = np.sum(w * ((p == 1) & (y == 1)))
    prec = tp / max(np.sum(w * (p == 1)), 1e-12)
    rec = tp / max(np.sum(w * (y == 1)), 1e-12)
    return prec, rec


def weighted_f1(y, p, w) -> float:
    prec, rec = weighted_prec_rec(y, p, w)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def best_threshold(y, scores, w) -> float:
    return float(THRESHOLDS[np.argmax([weighted_f1(y, (scores >= t).astype(int), w) for t in THRESHOLDS])])


# ============================================================================
# COMBINER
# ============================================================================

def _pipeline(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


class SurveyCombiner:
    def __init__(self, model=None, threshold: float = 0.5, C: float = 0.1):
        self.model, self.threshold, self.C = model, threshold, C

    @classmethod
    def fit_selected(cls, X_fixed: np.ndarray, y_fixed: np.ndarray, w_fixed: np.ndarray,
                     X_hand: np.ndarray, y_hand: np.ndarray, w_hand: np.ndarray, strata: Sequence[str],
                     C_grid=C_GRID, seed: int = 42) -> "SurveyCombiner":
        """
        X_fixed: rows always used for training (validation split, LLM labels), with normalized weights.
        X_hand: hand-labeled rows. C and the threshold are chosen by 5-fold CV over the hand-labeled
        rows (the fixed rows stay in every training fold), then the model is refit on everything.
        """
        groups = np.char.add(np.asarray(strata, dtype=str), y_hand.astype(str))
        folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X_hand, groups))
        best = None
        for C in C_grid:
            oof = np.zeros(len(y_hand))
            for tr, te in folds:
                X = np.vstack([X_fixed, X_hand[tr]])
                y = np.r_[y_fixed, y_hand[tr]]
                w = np.r_[w_fixed, w_hand[tr] / w_hand[tr].mean()]
                m = _pipeline(C).fit(X, y, logisticregression__sample_weight=w)
                oof[te] = m.predict_proba(X_hand[te])[:, 1]
            th = best_threshold(y_hand, oof, w_hand)
            f1 = weighted_f1(y_hand, (oof >= th).astype(int), w_hand)
            if best is None or f1 > best[0]:
                best = (f1, C, th)
        _, C, th = best
        X = np.vstack([X_fixed, X_hand])
        y = np.r_[y_fixed, y_hand]
        w = np.r_[w_fixed, w_hand / w_hand.mean()]
        model = _pipeline(C).fit(X, y, logisticregression__sample_weight=w)
        return cls(model, th, C)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = survey."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def save(self, model_dir: str) -> None:
        joblib.dump({"model": self.model, "threshold": self.threshold, "C": self.C},
                    os.path.join(model_dir, COMBINER_FILE))

    @classmethod
    def load(cls, model_dir: str) -> Optional["SurveyCombiner"]:
        path = os.path.join(model_dir, COMBINER_FILE)
        if not os.path.exists(path):
            return None
        state = joblib.load(path)
        return cls(state["model"], state["threshold"], state["C"])
