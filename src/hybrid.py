"""
Hybrid Survey Classifiers
=========================
Two ways of combining the DistilBERT survey probability with keyword heuristics:

  * OR rule (as in the paper): survey if DistilBERT OR the title keyword filter says so.
  * Learned combiner: a logistic regression over the DistilBERT logit and
    interpretable cues (title keywords, survey / research phrasing in the
    abstract, reference count). It is fitted on the validation split, which
    DistilBERT was not trained on.

Reference counts and abstracts are often missing in author profiles (Semantic
Scholar returns neither for many publishers), so the combiner is trained with
reference counts randomly masked, on validation papers with and without their
abstracts, and uses explicit 'missing' indicators.
"""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from text_utils import (
    PAPER_TITLE_KEYWORDS, TITLE_SURVEY_TERMS, ABSTRACT_SURVEY_CUES, ABSTRACT_RESEARCH_CUES, clean_text,
)

COMBINER_FILE = "hybrid_combiner.joblib"
FEATURES = ["bert_logit", "title_terms", "abstract_survey_cues", "abstract_research_cues",
            "abstract_missing", "log_refs", "refs_missing"]
MIN_ABSTRACT_WORDS = 10


def keyword_is_survey(title: str) -> bool:
    """Title keyword filter from the paper."""
    return bool(PAPER_TITLE_KEYWORDS.search(clean_text(title)))


def or_rule(survey_proba: np.ndarray, titles, threshold: float) -> np.ndarray:
    """1 = survey if DistilBERT or the title keyword filter flags it."""
    keyword = np.array([keyword_is_survey(t) for t in titles])
    return ((survey_proba >= threshold) | keyword).astype(int)


def build_features(df: pd.DataFrame, survey_proba: np.ndarray, use_refs: bool = True) -> pd.DataFrame:
    titles = df["Title"].map(clean_text)
    abstracts = df["Abstract"].map(clean_text)
    p = np.clip(survey_proba, 1e-4, 1 - 1e-4)

    refs = pd.to_numeric(df.get("ReferenceCount", pd.Series(0, index=df.index)), errors="coerce").fillna(0).values
    has_refs = (refs > 0) & use_refs

    return pd.DataFrame({
        "bert_logit": np.log(p / (1 - p)),
        "title_terms": titles.map(lambda t: int(bool(TITLE_SURVEY_TERMS.search(t)))).values,
        "abstract_survey_cues": abstracts.map(lambda a: min(len(ABSTRACT_SURVEY_CUES.findall(a)), 3)).values,
        "abstract_research_cues": abstracts.map(lambda a: min(len(ABSTRACT_RESEARCH_CUES.findall(a)), 3)).values,
        "abstract_missing": abstracts.map(lambda a: int(len(a.split()) < MIN_ABSTRACT_WORDS)).values,
        "log_refs": np.where(has_refs, np.log1p(refs), 0.0),
        "refs_missing": (~has_refs).astype(int),
    }, index=df.index)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold on P(survey) that maximizes survey-class F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = scores >= t
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        fn = np.sum(~pred & (y_true == 1))
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


class LearnedHybrid:
    def __init__(self, model: Optional[LogisticRegression] = None, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    def fit(self, df: pd.DataFrame, survey_proba: np.ndarray, is_survey: np.ndarray,
            refs_mask_rate: float = 0.5, seed: int = 42) -> "LearnedHybrid":
        X = build_features(df, survey_proba)
        # Randomly hide reference counts so the model also works when they are missing
        rng = np.random.default_rng(seed)
        mask = rng.random(len(X)) < refs_mask_rate
        X.loc[mask, "log_refs"] = 0.0
        X.loc[mask, "refs_missing"] = 1

        self.model = LogisticRegression(C=1.0, max_iter=1000)
        cv_scores = cross_val_predict(self.model, X[FEATURES], is_survey, cv=5, method="predict_proba")[:, 1]
        self.threshold = best_f1_threshold(is_survey, cv_scores)
        self.model.fit(X[FEATURES], is_survey)
        return self

    def predict_proba(self, df: pd.DataFrame, survey_proba: np.ndarray, use_refs: bool = True) -> np.ndarray:
        X = build_features(df, survey_proba, use_refs=use_refs)
        return self.model.predict_proba(X[FEATURES])[:, 1]

    def predict(self, df: pd.DataFrame, survey_proba: np.ndarray, use_refs: bool = True) -> np.ndarray:
        """1 = survey."""
        return (self.predict_proba(df, survey_proba, use_refs) >= self.threshold).astype(int)

    def coefficients(self) -> dict:
        return dict(zip(FEATURES, np.round(self.model.coef_[0], 3)))

    def save(self, model_dir: str) -> None:
        joblib.dump({"model": self.model, "threshold": self.threshold}, os.path.join(model_dir, COMBINER_FILE))

    @classmethod
    def load(cls, model_dir: str) -> Optional["LearnedHybrid"]:
        path = os.path.join(model_dir, COMBINER_FILE)
        if not os.path.exists(path):
            return None
        state = joblib.load(path)
        return cls(state["model"], state["threshold"])
