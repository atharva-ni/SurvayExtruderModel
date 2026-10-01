"""
Evaluation of our two models: the fine-tuned DistilBERT and the learned hybrid built on it
=========================================================================================
The report (reports/evaluation.md and .json) has three parts:

  1. Main results: both models on the held-out test split (with and without abstracts)
     and on the hand-labeled author-profile papers (data/eval_to_label.csv; for the learned
     hybrid, nested cross-validated predictions from `python main.py train --hybrid-only --cv 5`,
     because the hybrid is fitted on these papers). The learned hybrid is the main model.
  2. Our models and other methods on the same papers: TF-IDF + SVM, the title keyword
     filter, the original tool's title phrases, the indexer's document type, the
     Smyth & Cunningham (2025) rule, and on the hand-labeled papers the LLM labels.
  3. Survey exclusion per author: detected surveys, metrics without them, the survey share
     corrected for the hybrid's measured error rates (adjusted classify-and-count), and the
     papers each other method would exclude.
"""

import os
import re
import glob
import json
import argparse
from collections import Counter
from numbers import Real
from datetime import date

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tabulate import tabulate

from text_utils import paper_text, clean_text, keyword_is_survey
from inference import load_model, load_config, predict_survey_proba
from classifier import prepare_frame, calculate_indices, categorize
from train import SPLIT_FILE
from learned_hybrid import LearnedHybrid, cls_embeddings, hybrid_features
from train_hybrid import CV_FILE


def metrics(y_true, y_pred, weights=None) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    out = {"acc": accuracy_score(y_true, y_pred), "prec": p, "rec": r, "f1": f}
    if weights is not None:
        # Undo the stratified sampling: estimates for the real mix of papers in author profiles
        tp = np.sum(weights * ((y_true == 1) & (y_pred == 1)))
        out["acc_reweighted"] = float(np.sum(weights * (y_true == y_pred)) / np.sum(weights))
        out["prec_reweighted"] = float(tp / max(1e-9, np.sum(weights * (y_pred == 1))))
        out["rec_reweighted"] = float(tp / max(1e-9, np.sum(weights * (y_true == 1))))
        out["survey_rate_true"] = float(np.sum(weights * y_true) / np.sum(weights))
        out["survey_rate_pred"] = float(np.sum(weights * y_pred) / np.sum(weights))
    return out


N_BOOT = 2000
STRATUM_DEFINITIONS = {
    "title_kw": "survey term in the title",
    "abstract_cue": "survey phrasing in the abstract, none in the title",
    "random": "everything else",
}


def rw_prec_rec_rate(y, p, w):
    tp = np.sum(w * ((y == 1) & (p == 1)))
    return (tp / max(1e-9, np.sum(w * (p == 1))), tp / max(1e-9, np.sum(w * (y == 1))),
            np.sum(w * y) / np.sum(w))


def stratified_bootstrap(y, preds: dict, w, strata, seed=42) -> dict:
    """95% CIs of the reweighted precision / recall / survey rate, resampling within each stratum."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    samples = [np.concatenate([rng.choice(g, size=len(g)) for g in groups]) for _ in range(N_BOOT)]
    out = {}
    for name, p in preds.items():
        stats = np.array([rw_prec_rec_rate(y[i], p[i], w[i]) for i in samples])
        out[name] = {"prec_rw_ci": np.percentile(stats[:, 0], [2.5, 97.5]).tolist(),
                     "rec_rw_ci": np.percentile(stats[:, 1], [2.5, 97.5]).tolist(),
                     "rate_ci": np.percentile(stats[:, 2], [2.5, 97.5]).tolist()}
    return out


def stratified_bootstrap_reps(y, rep_preds, w, strata, seed=42) -> dict:
    """As stratified_bootstrap, for the average over repeated cross-validation predictions."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    samples = [np.concatenate([rng.choice(g, size=len(g)) for g in groups]) for _ in range(N_BOOT)]
    stats = np.array([np.mean([rw_prec_rec_rate(y[i], p[i], w[i]) for p in rep_preds], axis=0) for i in samples])
    return {"prec_rw_ci": np.percentile(stats[:, 0], [2.5, 97.5]).tolist(),
            "rec_rw_ci": np.percentile(stats[:, 1], [2.5, 97.5]).tolist(),
            "rate_ci": np.percentile(stats[:, 2], [2.5, 97.5]).tolist()}


def learned_cv_predictions(model_path: str, ev: pd.DataFrame):
    """Nested cross-validated predictions of the learned hybrid (reps x papers) for ev, or None."""
    cv_path = os.path.join(model_path, CV_FILE)
    if not os.path.exists(cv_path):
        return None
    with open(cv_path, encoding="utf-8") as f:
        cv = json.load(f)
    pos = {i: k for k, i in enumerate(cv["ids"])}
    idx = [pos.get(str(i)) for i in ev["id"]]
    return None if None in idx else np.array(cv["predictions"])[:, idx]


def error_rates(model_path: str, labeled_csv: str, seed: int = 42):
    """
    Recall (TPR) and false-positive rate of the learned hybrid on the hand-labeled profile papers, reweighted to
    the real mix of papers, from its cross-validated predictions; with stratified bootstrap samples of both.
    """
    if not labeled_csv or not os.path.exists(labeled_csv):
        return None
    ev = pd.read_csv(labeled_csv, dtype={"Label": str})
    ev = ev[ev["Label"].isin(["0", "1"])].reset_index(drop=True)
    reps = learned_cv_predictions(model_path, ev)
    if reps is None:
        return None
    y = (ev["Label"] == "0").astype(int).values
    w, strata = ev["StratumWeight"].values.astype(float), ev["Stratum"].values

    def rates(i):
        tpr = np.mean([np.sum(w[i] * (p[i] == 1) * (y[i] == 1)) / np.sum(w[i] * (y[i] == 1)) for p in reps])
        fpr = np.mean([np.sum(w[i] * (p[i] == 1) * (y[i] == 0)) / np.sum(w[i] * (y[i] == 0)) for p in reps])
        return tpr, fpr

    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    samples = np.array([rates(np.concatenate([rng.choice(g, size=len(g)) for g in groups])) for _ in range(N_BOOT)])
    tpr, fpr = rates(np.arange(len(y)))
    return {"tpr": tpr, "fpr": fpr, "samples": samples, "n": len(y), "source": labeled_csv}


def corrected_shares(flagged: float, cites_flagged: float, tpr, fpr):
    """
    Adjusted classify-and-count: the true survey share of papers, pi = (q - FPR) / (TPR - FPR), and of citations,
    PPV * (citation share of flagged papers) + FOR * (citation share of the others), with PPV = TPR pi / q and
    FOR = (1 - TPR) pi / (1 - q). Works on arrays of (TPR, FPR) samples as well.
    """
    q = flagged
    pi = np.clip((q - fpr) / np.maximum(tpr - fpr, 1e-9), 0, 1)
    ppv = np.clip(tpr * pi / q, 0, 1) if q > 0 else 0.0
    omit = np.clip((1 - tpr) * pi / (1 - q), 0, 1) if q < 1 else 0.0
    return pi, ppv * cites_flagged + omit * (1 - cites_flagged)


def corrected_table(table: pd.DataFrame, rates: dict) -> tuple:
    """Detected vs. estimated true survey share of papers and citations, per author and on average."""
    rows, data = [], []
    per_author = table[table["Author"] != "Average"]
    boot_pi, boot_c = [], []
    for _, r in per_author.iterrows():
        q, c = r["Papers excluded"] / 100, r["Citation reduction"] / 100
        pi, cit = corrected_shares(q, c, rates["tpr"], rates["fpr"])
        s_pi, s_c = corrected_shares(q, c, rates["samples"][:, 0], rates["samples"][:, 1])
        boot_pi.append(s_pi)
        boot_c.append(s_c)
        data.append({"Author": r["Author"], "flagged": q, "flagged_citations": c, "papers": float(pi),
                     "papers_ci": np.percentile(s_pi, [2.5, 97.5]).tolist(), "citations": float(cit),
                     "citations_ci": np.percentile(s_c, [2.5, 97.5]).tolist()})
    avg_pi, avg_c = np.mean(boot_pi, axis=0), np.mean(boot_c, axis=0)
    data.append({"Author": "Average", "flagged": float(np.mean([d["flagged"] for d in data])),
                 "flagged_citations": float(np.mean([d["flagged_citations"] for d in data])),
                 "papers": float(np.mean([d["papers"] for d in data])),
                 "papers_ci": np.percentile(avg_pi, [2.5, 97.5]).tolist(),
                 "citations": float(np.mean([d["citations"] for d in data])),
                 "citations_ci": np.percentile(avg_c, [2.5, 97.5]).tolist()})
    for d in data:
        rows.append([d["Author"], f"{100 * d['flagged']:.1f}%", f"{100 * d['papers']:.1f}% ({ci_str(d['papers_ci'])})",
                     f"{100 * d['flagged_citations']:.1f}%",
                     f"{100 * d['citations']:.1f}% ({ci_str(d['citations_ci'])})"])
    text = (tabulate(rows, headers=["Author", "Papers flagged", "Est. true survey share of papers (95% CI)",
                                    "Citations flagged", "Est. true survey share of citations (95% CI)"],
                     tablefmt="github") +
            f"\n\nAdjusted classify-and-count with the learned hybrid's reweighted recall ({100 * rates['tpr']:.1f}%) "
            f"and false-positive rate ({100 * rates['fpr']:.2f}%) from its cross-validated predictions on the "
            f"{rates['n']} hand-labeled papers (`{rates['source']}`); CIs: {N_BOOT} stratified bootstrap resamples "
            "of those papers. Assumes the error rates measured on the hand-labeled papers (Cohort B's "
            "Semantic Scholar profiles, papers with abstracts) hold for every profile, and that citations do not "
            "depend on whether a paper is misclassified.")
    return text, data


def ci_str(ci) -> str:
    return f"{100 * ci[0]:.0f}–{100 * ci[1]:.0f}%"


def sampling_design(ev_all: pd.DataFrame, ev: pd.DataFrame) -> list:
    """Per stratum: pool size in the profiles, papers sampled / labeled, surveys found, weight."""
    rows = []
    for stratum, g in ev_all.groupby("Stratum"):
        weight = float(g["StratumWeight"].iloc[0])
        labeled = ev[ev["Stratum"] == stratum]
        rows.append({"stratum": stratum, "definition": STRATUM_DEFINITIONS.get(stratum, ""),
                     "pool": int(round(weight * len(g))), "sampled": len(g), "labeled": len(labeled),
                     "surveys": int((labeled["Label"] == "0").sum()), "weight": weight})
    return rows


def label_agreement(ev_all: pd.DataFrame) -> dict:
    """Agreement and Cohen's kappa between the blind human labels and the LLM labels (survey vs. research)."""
    both = ev_all[ev_all["LabelClaude"].isin(["0", "1"]) & ev_all["LabelHumanBlind"].isin(["0", "1"])]
    agree = float((both["LabelClaude"] == both["LabelHumanBlind"]).mean())
    p1, p2 = (both["LabelClaude"] == "0").mean(), (both["LabelHumanBlind"] == "0").mean()
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    return {"n": len(both), "agreement": agree, "kappa": float((agree - expected) / (1 - expected)),
            "disagreements": int((both["LabelClaude"] != both["LabelHumanBlind"]).sum())}


def sampling_table(rows: list) -> str:
    total_w = sum(r["weight"] * r["labeled"] for r in rows)
    body = [[r["stratum"], r["definition"], r["pool"], r["sampled"], r["labeled"], r["surveys"], r["weight"],
             f"{100 * r['weight'] * r['surveys'] / total_w:.1f}%"] for r in rows]
    return (tabulate(body, headers=["Stratum", "Definition", "Pool", "Sampled", "Labeled", "Surveys", "Weight",
                                    "Share of est. survey rate"], tablefmt="github") +
            "\n\nPool = profile papers with an abstract of at least 30 words, deduplicated by title, not in the "
            "training set. Weight = pool / sampled. Estimated survey rate = weighted surveys / weighted labeled papers.")


def survey_proba_for(frame: pd.DataFrame, model_path: str) -> np.ndarray:
    tokenizer, model = load_model(model_path)
    texts = [paper_text(t, a) for t, a in zip(frame["Title"], frame["Abstract"])]
    return predict_survey_proba(texts, tokenizer, model, max_length=load_config(model_path).get("max_length", 384))


# ============================================================================
# OTHER METHODS (for comparison with the learned hybrid)
# ============================================================================

def indexer_is_review(frame: pd.DataFrame) -> np.ndarray:
    """Baseline: the document type assigned by the indexer (OpenAlex / Semantic Scholar) says 'review'."""
    return frame["Type"].str.contains(r"\breview\b", case=False, regex=True).astype(int).values


# Original Google Scholar Survey Excluder (github.com/nimaafraz/Google-Scholar-Survey-Excluder, 2024):
# case-insensitive substring match of these phrases in the title
ORIGINAL_TOOL_PHRASES = ["survey", "review", "directions", "challenges", "trends", "approaches", "opportunities",
                         "concepts", "roadmap", "Advances in", "Analysis of"]
ORIGINAL_TOOL_PATTERN = re.compile("|".join(map(re.escape, ORIGINAL_TOOL_PHRASES)), re.IGNORECASE)


def original_tool(frame: pd.DataFrame) -> np.ndarray:
    """Baseline: the original title-phrase filter of the Google Scholar Survey Excluder."""
    return frame["Title"].map(lambda t: bool(ORIGINAL_TOOL_PATTERN.search(t))).astype(int).values


# Smyth & Cunningham (arXiv:2511.07490, Section 4.1.2 and Appendix A.2): a review if the indexer type is
# Review or MetaAnalysis AND (the title or abstract contains a strict review phrase OR the venue is a review venue)
SMYTH_CUNNINGHAM_PHRASES = [
    "literature review", "literature survey", "literature study", "literature analysis", "comparative review",
    "comparative survey", "comparative study", "comparative analysis", "bibliographic review", "bibliographic survey",
    "bibliographic study", "bibliographic analysis", "bibliometric review", "bibliometric survey",
    "bibliometric study", "bibliometric analysis", "scientometric review", "scientometric survey",
    "scientometric study", "scientometric analysis", "systematic review", "systematic survey", "systematic study",
    "systematic analysis", "systematic literature review", "systematic literature survey",
    "systematic literature study", "systematic literature analysis", "meta analysis", "meta review", "meta study",
    "meta-analysis", "meta-review", "meta-study", "mapping study", "systematic mapping", "emerging trends",
    "comprehensive survey", "contemporary survey", "systematization of knowledge", "systematisation of knowledge",
    "tutorial", "review of", "research directions", "a review", "case studies", "overview of", "summary of",
]
SMYTH_CUNNINGHAM_PATTERN = re.compile(r"\b(?:" + "|".join(map(re.escape, SMYTH_CUNNINGHAM_PHRASES)) + r")\b",
                                      re.IGNORECASE)
REVIEW_VENUE = re.compile(r"review|survey|trends in", re.IGNORECASE)


def smyth_cunningham_rule(frame: pd.DataFrame) -> np.ndarray:
    """Baseline: indexer type Review/MetaAnalysis and (a review phrase or a review venue)."""
    typed = frame["Type"].str.contains(r"\b(?:review|meta-?analysis)\b", case=False, regex=True)
    phrase = [bool(SMYTH_CUNNINGHAM_PATTERN.search(clean_text(t) + " " + clean_text(a)))
              for t, a in zip(frame["Title"], frame["Abstract"])]
    venue = frame["Venue"].map(lambda v: bool(REVIEW_VENUE.search(v)))
    return (typed & (np.array(phrase) | venue)).astype(int).values


MAIN = "Learned hybrid (ours)"
DISTILBERT = "Fine-tuned DistilBERT (ours)"
COHORT_B = "Cohort B (prolific survey authors)"
COHORT_A = "Cohort A (comparison authors)"


def all_methods(frame, proba, model_path, svm, vectorizer, learned=None, saved_hybrid=True) -> dict:
    """
    Return {method name: is_survey predictions} for one dataset: the learned hybrid first, then the other methods.
    learned: predictions of the learned hybrid to use instead of the saved hybrid (e.g. cross-validated
    predictions for papers the hybrid was fitted on); by default the saved hybrid is applied
    (saved_hybrid=False leaves the learned hybrid out).
    """
    threshold = load_config(model_path)["threshold"]
    texts = [paper_text(t, a) for t, a in zip(frame["Title"], frame["Abstract"])]

    if learned is None and saved_hybrid:
        hybrid = LearnedHybrid.load(model_path)
        if hybrid is not None:
            learned = hybrid.predict(hybrid_features(frame, proba, cls_embeddings(frame, model_path)))
    preds = {} if learned is None else {MAIN: np.asarray(learned, dtype=int)}
    preds.update({
        DISTILBERT: (proba >= threshold).astype(int),
        "TF-IDF + SVM": (svm.predict(vectorizer.transform(texts)) == 1).astype(int),
        "Keyword only": np.array([keyword_is_survey(t) for t in frame["Title"]], dtype=int),
        "Original tool": original_tool(frame),
    })
    if (frame["Type"].str.strip() != "").any():
        preds["Indexer type (Review)"] = indexer_is_review(frame)
        preds["Smyth & Cunningham rule"] = smyth_cunningham_rule(frame)
    return preds


def fmt_table(rows: dict, extra_cols=()) -> list:
    table = []
    for method, m in rows.items():
        row = [method] + [f"{100 * m[k]:.1f}%" for k in ("acc", "prec", "rec", "f1")]
        row += [f"{100 * m[k]:.1f}%" if k in m else "" for k in extra_cols]
        table.append(row)
    return table


def _title_key(t) -> str:
    return " ".join(str(t).lower().split())


def author_impact(authors_glob: str, model_path: str,
                  labeled_csvs=("data/llm_labels.csv", "data/eval_to_label.csv")) -> pd.DataFrame:
    """
    Exclude the surveys detected by the learned hybrid and recalculate the metrics. Books/editorials are not
    surveys and stay in both profiles. 'In hybrid training' counts the profile papers that are among the
    labeled papers the learned hybrid was fitted on. Also returns, per other method, which papers it would exclude.
    """
    hybrid = LearnedHybrid.load(model_path)
    threshold = load_config(model_path)["threshold"]
    labeled = set()
    for path in labeled_csvs:
        if os.path.exists(path):
            labeled |= set(pd.read_csv(path, usecols=["Title"])["Title"].map(_title_key))
    rows, comparison = [], {}
    for path in sorted(glob.glob(authors_glob)):
        df = pd.read_csv(path)
        df["citationCount"] = df["citationCount"].fillna(0).astype(int)
        frame = prepare_frame(df)
        proba = survey_proba_for(frame, model_path)
        learned = hybrid.predict(hybrid_features(frame, proba, cls_embeddings(frame, model_path)))
        category = categorize(frame, learned)

        # Same profile, other methods: which papers each one would exclude
        excluded_by = {
            MAIN: category == "survey",
            DISTILBERT: categorize(frame, (proba >= threshold).astype(int)) == "survey",
            "Original tool": original_tool(frame) == 1,
            "Indexer type (Review)": indexer_is_review(frame) == 1,
            "Smyth & Cunningham rule": smyth_cunningham_rule(frame) == 1,
        }
        h0, _ = calculate_indices(df)
        for method, mask in excluded_by.items():
            comparison.setdefault(method, {})[os.path.basename(path)] = {
                "excluded": 100 * mask.mean(),
                "citations": 100 * df.loc[mask, "citationCount"].sum() / max(1, df["citationCount"].sum()),
                "h_drop": h0 - calculate_indices(df[~mask])[0],
            }

        survey = category == "survey"
        h0, i0 = calculate_indices(df)
        h1, i1 = calculate_indices(df[~survey])
        total_cites = max(1, df["citationCount"].sum())
        names = Counter(n.strip() for s in df.get("authors", pd.Series(dtype=str)).dropna() for n in str(s).split(";"))
        row = {
            "Author": names.most_common(1)[0][0] if names else os.path.basename(path),
            "File": os.path.basename(path),
            "Papers": len(df),
            "Surveys": int(survey.sum()),
            "Non-papers": int((category == "non-paper").sum()),
            "In hybrid training": int(frame["Title"].map(_title_key).isin(labeled).sum()),
            "Citations": int(df["citationCount"].sum()),
            "Citations*": int(df.loc[~survey, "citationCount"].sum()),
            "Papers excluded": 100 * survey.mean(),
            "Citation reduction": 100 * df.loc[survey, "citationCount"].sum() / total_cites,
            "h-index": h0,
            "h-index*": h1,
            "h-index drop": h0 - h1,
            "h-index change": 100 * (h1 - h0) / max(1, h0),
            "i10-index reduction": 100 * (1 - i1 / max(1, i0)),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        avg = {c: "" for c in table.columns}
        avg["Author"] = "Average"
        for col in table.columns:
            if col not in ("Author", "File") and pd.api.types.is_numeric_dtype(table[col]):
                avg[col] = table[col].mean()
        table = pd.concat([table, pd.DataFrame([avg])], ignore_index=True)
    return table, comparison


def method_comparison_table(comparison: dict) -> str:
    """Per method: average share of papers / citations excluded and h-index drop per method."""
    files = list(next(iter(comparison.values())).keys())
    rows = []
    for method, per_author in comparison.items():
        vals = list(per_author.values())
        rows.append([method,
                     f"{np.mean([v['excluded'] for v in vals]):.1f}%",
                     f"{np.mean([v['citations'] for v in vals]):.1f}%",
                     f"{np.mean([v['h_drop'] for v in vals]):.1f}"] +
                    [per_author[f]["h_drop"] for f in files])
    headers = ["Method", "Papers excluded", "Citations", "Δh (avg)"] + [f"Δh {os.path.splitext(f)[0]}" for f in files]
    return tabulate(rows, headers=headers, tablefmt="github")


def cohort_comparison_table(survey: dict, comparison: dict) -> str:
    """Per method: average share of papers / citations excluded and h-index drop in each cohort."""
    avg = lambda per_author, k: np.mean([v[k] for v in per_author.values()])
    rows = [[method,
             f"{avg(survey[method], 'excluded'):.1f}%", f"{avg(comparison[method], 'excluded'):.1f}%",
             f"{avg(survey[method], 'citations'):.1f}%", f"{avg(comparison[method], 'citations'):.1f}%",
             f"{avg(survey[method], 'h_drop'):.1f}", f"{avg(comparison[method], 'h_drop'):.1f}"]
            for method in survey if method in comparison]
    return tabulate(rows, headers=["Method", "Papers: Cohort B", "Papers: Cohort A", "Citations: Cohort B",
                                   "Citations: Cohort A", "Δh: Cohort B", "Δh: Cohort A"], tablefmt="github")


def main_results_table(results: dict) -> str:
    """Both of our models on every evaluation set in this report."""
    rows = []
    for key, name in (("test_split", "Held-out test split (journal papers, title + abstract)"),
                      ("test_split_title_only", "Held-out test split, title only")):
        for model in (MAIN, DISTILBERT):
            if key in results and model in results[key]["results"]:
                m = results[key]["results"][model]
                rows.append([name, model, results[key]["n"], f"{100 * m['acc']:.1f}%", f"{100 * m['prec']:.1f}%",
                             f"{100 * m['rec']:.1f}%", f"{100 * m['f1']:.1f}%"])
    for model in (MAIN, DISTILBERT):
        if "hand_labeled" in results and model in results["hand_labeled"]["results"]:
            m = results["hand_labeled"]["results"][model]
            p, r = m["prec_reweighted"], m["rec_reweighted"]
            rows.append(["Author-profile papers, blind-labeled (reweighted to the real mix)", model,
                         results["hand_labeled"]["n"], f"{100 * m['acc_reweighted']:.1f}%",
                         f"{100 * p:.1f}% ({ci_str(m['prec_rw_ci'])})", f"{100 * r:.1f}% ({ci_str(m['rec_rw_ci'])})",
                         f"{100 * 2 * p * r / (p + r):.1f}%"])
    return (tabulate(rows, headers=["Evaluation set", "Model", "Papers", "Accuracy", "Precision", "Recall", "F1"],
                     tablefmt="github") +
            "\n\nSurvey = positive class. The learned hybrid is the main model. Author-profile papers: the learned "
            "hybrid is fitted on these papers, so it is scored by nested cross-validation; 95% CIs from stratified "
            "bootstrap. Papers from venues not used in training: reports/external_evaluation.md.")


def run_evaluation(
    model_path: str = "./distilbert_survey_model",
    eval_set: str = "data/eval_to_label.csv",
    authors_glob: str = "data/proauthor_s2/*.csv",
    report_dir: str = "reports",
    comparison_glob: str = None,
    rates_set: str = "data/eval_to_label.csv",
) -> dict:
    """rates_set: hand-labeled papers whose error rates correct the per-author survey shares."""
    split_path = os.path.join(model_path, SPLIT_FILE)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"{split_path} not found — train the model first (python main.py train)")
    split = pd.read_csv(split_path)
    train_df = split[split["Split"] == "train"]
    test_df = split[split["Split"] == "test"].reset_index(drop=True)

    # TF-IDF + SVM, trained on the same training split
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = vectorizer.fit_transform([paper_text(t, a) for t, a in zip(train_df["Title"], train_df["Abstract"])])
    svm = LinearSVC(C=1.0).fit(X_train, (train_df["Label"] == 0).astype(int))

    results = {"date": str(date.today()), "model": os.path.abspath(model_path),
               "distilbert_threshold": load_config(model_path)["threshold"]}
    hybrid = LearnedHybrid.load(model_path)
    if hybrid is not None:
        results["learned_hybrid"] = {"C": hybrid.C, "threshold": hybrid.threshold}
    headers = ["Method", "Acc.", "Prec.", "Recall", "F1"]
    sections = []

    # ---- Held-out test split, with and without abstracts (many profile papers have no abstract) ----
    y = (test_df["Label"] == 0).astype(int).values
    for key, title, df in (("test_split", "held-out test split", test_df),
                           ("test_split_title_only", "held-out test split, title only", test_df.assign(Abstract=""))):
        print(f"📊 Evaluating on the {title} ({len(df)} papers)...")
        frame = prepare_frame(df)
        preds = all_methods(frame, survey_proba_for(frame, model_path), model_path, svm, vectorizer)
        rows = {name: metrics(y, p) for name, p in preds.items()}
        results[key] = {"n": len(y), "surveys": int(y.sum()), "results": rows}
        sections.append((f"Our models and other methods — {title} (n = {len(y)}, {y.sum()} surveys)",
                         tabulate(fmt_table(rows), headers=headers, tablefmt="github")))

    # ---- Hand-labeled author-profile papers ----
    if eval_set and os.path.exists(eval_set):
        ev_all = pd.read_csv(eval_set, dtype={"Label": str})
        ev = ev_all[ev_all["Label"].isin(["0", "1"])].reset_index(drop=True)
        if len(ev):
            print(f"📊 Evaluating on the hand-labeled set ({len(ev)} papers)...")
            frame = prepare_frame(ev)
            y = (ev["Label"] == "0").astype(int).values
            w = ev["StratumWeight"].values.astype(float)
            strata = ev["Stratum"].values
            # The learned hybrid is fitted on these papers: use its cross-validated predictions, never the saved hybrid
            reps = learned_cv_predictions(model_path, ev)
            if reps is None:
                print(f"No cross-validated predictions in '{os.path.join(model_path, CV_FILE)}' "
                      "(python main.py train --hybrid-only --cv 5); the learned hybrid is left out")
            else:
                results["learned_hybrid_cv_repetitions"] = [
                    dict(zip(("prec_rw", "rec_rw"), rw_prec_rec_rate(y, r, w)[:2])) for r in reps]
            preds = all_methods(frame, survey_proba_for(frame, model_path), model_path, svm, vectorizer,
                                learned=None if reps is None else reps[0], saved_hybrid=False)
            if "LabelClaude" in ev.columns and ev["LabelClaude"].isin(["0", "1"]).all():
                # Zero-shot LLM: the LLM's labels scored against the author's labels
                preds["LLM labels (Claude)"] = (ev["LabelClaude"] == "0").astype(int).values
            rows = {name: metrics(y, p, w) for name, p in preds.items()}
            cis = stratified_bootstrap(y, preds, w, strata)
            for name in preds:
                rows[name].update(cis[name])
            confident = ev["Confidence"].isin(["H", "M"]).values if "Confidence" in ev.columns else None
            if confident is not None:
                for name, p in preds.items():
                    rows[name]["f1_confident"] = metrics(y[confident], p[confident])["f1"]
            if reps is not None:
                # Average the learned hybrid over the CV repetitions (bootstrap CIs of the average)
                ms = [metrics(y, p, w) for p in reps]
                rows[MAIN] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
                if confident is not None:
                    rows[MAIN]["f1_confident"] = float(np.mean([metrics(y[confident], p[confident])["f1"]
                                                                for p in reps]))
                rows[MAIN].update(stratified_bootstrap_reps(y, reps, w, strata))
            results["hand_labeled_sampling"] = sampling_design(ev_all, ev)

            labeled_by = ", ".join(sorted(ev["LabeledBy"].dropna().unique())) if "LabeledBy" in ev.columns else "unknown"
            results["hand_labeled"] = {"n": len(y), "surveys": int(y.sum()), "labeled_by": labeled_by, "results": rows}
            names = {"f1_confident": "F1 (H+M labels)", "prec_reweighted": "Prec. (rw)",
                     "rec_reweighted": "Recall (rw)", "survey_rate_pred": "Flagged (rw)"}
            extra = [c for c in names if c in next(iter(rows.values()))]
            table = fmt_table(rows, extra)
            for row, m in zip(table, rows.values()):
                row += [ci_str(m["prec_rw_ci"]), ci_str(m["rec_rw_ci"])]
            first = next(iter(rows.values()))
            rate = f"{100 * first.get('survey_rate_true', 0):.1f}% (95% CI {ci_str(first['rate_ci'])})"
            sections.append((
                f"Our models and other methods — author-profile papers, blind-labeled (n = {len(y)}, {y.sum()} "
                f"surveys; labels: {labeled_by})",
                tabulate(table, headers=headers + [names[c] for c in extra] + ["Prec. (rw) 95% CI",
                                                                               "Recall (rw) 95% CI"],
                         tablefmt="github") +
                "\n\n(rw) = reweighted by sampling stratum to the real mix of papers in the profiles; "
                f"estimated true survey rate: {rate}. CIs: {N_BOOT} bootstrap resamples within each stratum." +
                ("" if reps is None else
                 f" Learned hybrid: fitted on these papers, so scored by nested 10-fold cross-validation, "
                 f"averaged over {len(reps)} repetitions (reweighted precision per repetition: " +
                 ", ".join(f"{100 * r['prec_rw']:.1f}%" for r in results["learned_hybrid_cv_repetitions"]) +
                 "; recall: " +
                 ", ".join(f"{100 * r['rec_rw']:.1f}%" for r in results["learned_hybrid_cv_repetitions"]) + ").")))
            sections.append(("Author-profile papers — sampling design", sampling_table(results["hand_labeled_sampling"])))
            if {"LabelClaude", "LabelHumanBlind"} <= set(ev_all.columns):
                results["label_agreement"] = label_agreement(ev_all)
                a = results["label_agreement"]
                n_blind = int((ev_all["LabelHumanBlind"] != "").sum())
                rest = ("All papers were labeled blind by the first author." if n_blind == len(ev_all) else
                        f"{n_blind} papers were labeled blind by the first author; the others carry the LLM label, "
                        "verified by the first author.")
                sections.append(("Author-profile papers — label agreement: first author (blind) vs. LLM", (
                    f"{a['n']} papers labeled survey or research by both: agreement {100 * a['agreement']:.1f}%, "
                    f"Cohen's kappa {a['kappa']:.2f}; {a['disagreements']} disagreements (the first author's label "
                    f"is used). {rest}")))

    # ---- Survey exclusion per author: Cohort B (survey authors) and Cohort A (comparison) ----
    rates = error_rates(model_path, rates_set)
    if rates is not None:
        results["error_rates"] = {"tpr": rates["tpr"], "fpr": rates["fpr"], "source": rates["source"]}
    cohorts, corrected = {}, {}
    for key, label, pattern in (("author_impact", COHORT_B, authors_glob),
                                ("comparison_impact", COHORT_A, comparison_glob)):
        if not (pattern and glob.glob(pattern)):
            continue
        print(f"📊 Recalculating author metrics ({label})...")
        table, comparison = author_impact(pattern, model_path)
        cohorts[label] = comparison
        results[key] = table.to_dict(orient="records")
        results[key + "_by_method"] = comparison
        show = table.drop(columns=["File"]).copy()
        for col in show.columns:
            if col in ("Papers excluded", "Citation reduction", "i10-index reduction", "h-index change"):
                show[col] = show[col].map(lambda v: f"{v:.1f}%" if isinstance(v, Real) else v)
            elif col in ("Papers", "Surveys", "Non-papers", "In hybrid training", "Citations", "Citations*",
                         "h-index", "h-index*", "h-index drop"):
                show[col] = show[col].map(lambda v: v if not isinstance(v, Real) else
                                          f"{int(v):,}" if float(v).is_integer() else f"{v:,.1f}")
        sections.append((f"Survey exclusion per author — {label} ({pattern})",
                         tabulate(show.values.tolist(), headers=list(show.columns), tablefmt="github") +
                         "\n\n* Without the surveys detected by the learned hybrid. Books/editorials are not counted "
                         "as surveys. 'In hybrid training': profile papers among the labeled papers the learned "
                         "hybrid was fitted on."))
        if rates is not None:
            text, data = corrected_table(table, rates)
            corrected[label] = data
            results[key + "_corrected"] = data
            sections.append((f"Survey share corrected for classification error — {label}", text))
        sections.append((f"Our models and other methods — papers each would exclude, {label} (averages over authors)",
                         method_comparison_table(comparison) +
                         "\n\nBoth of our models include the title-only and non-paper rules."))
    if len(cohorts) == 2:
        note = ""
        if len(corrected) == 2:
            s, c = corrected[COHORT_B][-1], corrected[COHORT_A][-1]
            note = (f"\n\nCorrected for classification error: surveys are {100 * s['papers']:.1f}% "
                    f"({ci_str(s['papers_ci'])}) vs. {100 * c['papers']:.1f}% ({ci_str(c['papers_ci'])}) of papers "
                    f"and {100 * s['citations']:.1f}% ({ci_str(s['citations_ci'])}) vs. {100 * c['citations']:.1f}% "
                    f"({ci_str(c['citations_ci'])}) of citations. The two cohorts' intervals share the same "
                    "error-rate uncertainty, so they are not independent.")
        sections.append(("Cohort B vs. Cohort A (averages over authors)",
                         cohort_comparison_table(cohorts[COHORT_B], cohorts[COHORT_A]) + note))

    # ---- Report: main results first ----
    sections.insert(0, ("Main results — our models", main_results_table(results)))
    os.makedirs(report_dir, exist_ok=True)
    lh = results.get("learned_hybrid", {})
    md = [f"# Survey Excluder evaluation ({results['date']})", "",
          f"Our models, both in `{model_path}`: the fine-tuned DistilBERT (threshold "
          f"{results['distilbert_threshold']:.2f}) and the learned hybrid built on it, the main model "
          f"(src/learned_hybrid.py; C = {lh.get('C')}, threshold {lh.get('threshold', float('nan')):.2f}).", ""]
    for title, table in sections:
        md += [f"## {title}", "", table, ""]
        print(f"\n{title}\n{table}")
    with open(os.path.join(report_dir, "evaluation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(report_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n✅ Report saved to '{os.path.join(report_dir, 'evaluation.md')}'")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the learned hybrid and compare it with other methods")
    parser.add_argument("--model", type=str, default="./distilbert_survey_model")
    parser.add_argument("--eval-set", type=str, default="data/eval_to_label.csv")
    parser.add_argument("--authors", type=str, default="data/proauthor_s2/*.csv")
    parser.add_argument("--report-dir", type=str, default="reports")
    parser.add_argument("--comparison-authors", type=str, default=None,
                        help="Glob of comparison-cohort profiles (authors who rarely publish surveys)")
    args = parser.parse_args()
    run_evaluation(args.model, args.eval_set, args.authors, args.report_dir, args.comparison_authors)
