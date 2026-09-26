"""
Evaluation: regenerates the paper's tables
==========================================
Table II  - classification performance (survey = positive class) on
            (a) the held-out test split of the training dataset, and
            (b) the hand-labeled author-profile set (data/eval_to_label.csv).
Table IV  - impact of excluding detected surveys on each author profile.

Methods: keyword only (title), TF-IDF + SVM, DistilBERT only, OR hybrid (paper),
learned hybrid (title + abstract only), learned hybrid + reference count, and
optionally a baseline model (e.g. the old synthetic-data model with its OR rule).
Results are written to reports/evaluation.md and reports/evaluation.json.
"""

import os
import glob
import json
import argparse
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tabulate import tabulate

from text_utils import paper_text
from inference import load_model, load_config, predict_survey_proba
from hybrid import LearnedHybrid, or_rule, keyword_is_survey
from classifier import prepare_frame, calculate_indices, categorize, is_magazine_without_survey_signal
from train import SPLIT_FILE


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


def survey_proba_for(frame: pd.DataFrame, model_path: str) -> np.ndarray:
    tokenizer, model = load_model(model_path)
    texts = [paper_text(t, a) for t, a in zip(frame["Title"], frame["Abstract"])]
    return predict_survey_proba(texts, tokenizer, model, max_length=load_config(model_path).get("max_length", 384))


def all_methods(frame, proba, model_path, svm, vectorizer, baseline=None) -> dict:
    """Return {method name: is_survey predictions} for one dataset."""
    threshold = load_config(model_path)["threshold"]
    combiner = LearnedHybrid.load(model_path)
    texts = [paper_text(t, a) for t, a in zip(frame["Title"], frame["Abstract"])]

    preds = {
        "Keyword only (title)": np.array([keyword_is_survey(t) for t in frame["Title"]], dtype=int),
        "TF-IDF + SVM": (svm.predict(vectorizer.transform(texts)) == 1).astype(int),
        "DistilBERT only": (proba >= threshold).astype(int),
        "Hybrid (OR, paper)": or_rule(proba, frame["Title"], threshold),
        "Learned hybrid (title+abstract)": combiner.predict(frame, proba, use_refs=False),
        "Learned hybrid (+ reference count)": combiner.predict(frame, proba, use_refs=True),
    }
    magazine_only = is_magazine_without_survey_signal(frame)
    if magazine_only.any():
        preds["Learned hybrid + magazine rule"] = preds["Learned hybrid (+ reference count)"] * (~magazine_only)
    if baseline is not None:
        name, base_proba, base_threshold = baseline
        preds[name] = or_rule(base_proba, frame["Title"], base_threshold)
    return preds


def fmt_table(rows: dict, extra_cols=()) -> list:
    table = []
    for method, m in rows.items():
        row = [method] + [f"{100 * m[k]:.1f}%" for k in ("acc", "prec", "rec", "f1")]
        row += [f"{100 * m[k]:.1f}%" if k in m else "" for k in extra_cols]
        table.append(row)
    return table


def author_impact(authors_glob: str, model_path: str) -> pd.DataFrame:
    """
    Table IV: exclude detected surveys (learned hybrid) and recalculate metrics.
    Books/editorials are not surveys and stay in both profiles. Magazine articles
    without explicit survey framing are kept; the '+ magazine' columns show the
    effect of excluding them as well.
    """
    combiner = LearnedHybrid.load(model_path)
    rows = []
    for path in sorted(glob.glob(authors_glob)):
        df = pd.read_csv(path)
        df["citationCount"] = df["citationCount"].fillna(0).astype(int)
        frame = prepare_frame(df)
        proba = survey_proba_for(frame, model_path)
        category = categorize(frame, combiner.predict(frame, proba, use_refs=True))

        survey = category == "survey"
        survey_or_magazine = survey | (category == "magazine-overview")
        h0, i0 = calculate_indices(df)
        h1, i1 = calculate_indices(df[~survey])
        h2, _ = calculate_indices(df[~survey_or_magazine])
        total_cites = max(1, df["citationCount"].sum())

        names = Counter(n.strip() for s in df.get("authors", pd.Series(dtype=str)).dropna() for n in str(s).split(";"))
        rows.append({
            "Author": names.most_common(1)[0][0] if names else os.path.basename(path),
            "File": os.path.basename(path),
            "Papers": len(df),
            "Surveys": int(survey.sum()),
            "Magazine overviews": int((category == "magazine-overview").sum()),
            "Non-papers": int((category == "non-paper").sum()),
            "Papers excluded": 100 * survey.mean(),
            "Citation reduction": 100 * df.loc[survey, "citationCount"].sum() / total_cites,
            "h-index": f"{h0} → {h1}",
            "h-index drop": h0 - h1,
            "i10-index reduction": 100 * (1 - i1 / max(1, i0)),
            "Citation reduction (+ magazine)": 100 * df.loc[survey_or_magazine, "citationCount"].sum() / total_cites,
            "h-index drop (+ magazine)": h0 - h2,
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        avg = {"Author": "Average", "File": "", "Papers": "", "Surveys": "", "Magazine overviews": "",
               "Non-papers": "", "h-index": ""}
        for col in ("Papers excluded", "Citation reduction", "h-index drop", "i10-index reduction",
                    "Citation reduction (+ magazine)", "h-index drop (+ magazine)"):
            avg[col] = table[col].mean()
        table = pd.concat([table, pd.DataFrame([avg])], ignore_index=True)
    return table


def run_evaluation(
    model_path: str = "./distilbert_survey_model",
    eval_set: str = "data/eval_to_label.csv",
    authors_glob: str = "data/proauthor/*.csv",
    baseline_model: str = None,
    report_dir: str = "reports",
) -> dict:
    split_path = os.path.join(model_path, SPLIT_FILE)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"{split_path} not found — train the model first (python main.py train)")
    split = pd.read_csv(split_path)
    train_df = split[split["Split"] == "train"]
    test_df = split[split["Split"] == "test"].reset_index(drop=True)

    # TF-IDF + SVM baseline trained on the same training split
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = vectorizer.fit_transform([paper_text(t, a) for t, a in zip(train_df["Title"], train_df["Abstract"])])
    svm = LinearSVC(C=1.0).fit(X_train, (train_df["Label"] == 0).astype(int))

    results = {"date": str(date.today()), "model": os.path.abspath(model_path),
               "threshold": load_config(model_path)["threshold"],
               "hybrid_coefficients": {k: float(v) for k, v in LearnedHybrid.load(model_path).coefficients().items()}}
    sections = []

    def baseline_for(frame):
        if not baseline_model:
            return None
        return (f"Baseline: {os.path.basename(os.path.normpath(baseline_model))} (OR, th=0.8)",
                survey_proba_for(frame, baseline_model), 0.8)

    # ---- (a) In-domain test split ----
    print(f"📊 Evaluating on the held-out test split ({len(test_df)} papers)...")
    frame = prepare_frame(test_df)
    y = (test_df["Label"] == 0).astype(int).values
    preds = all_methods(frame, survey_proba_for(frame, model_path), model_path, svm, vectorizer, baseline_for(frame))
    rows = {name: metrics(y, p) for name, p in preds.items()}
    results["test_split"] = {"n": len(y), "surveys": int(y.sum()), "results": rows}
    sections.append((f"Table II(a) — held-out test split (n = {len(y)}, {y.sum()} surveys)",
                     tabulate(fmt_table(rows), headers=["Method", "Acc.", "Prec.", "Recall", "F1"], tablefmt="github")))

    # ---- (a2) Same test split, title only (many profile papers have no abstract) ----
    title_only = test_df.assign(Abstract="")
    frame = prepare_frame(title_only)
    preds = all_methods(frame, survey_proba_for(frame, model_path), model_path, svm, vectorizer, baseline_for(frame))
    rows = {name: metrics(y, p) for name, p in preds.items()}
    results["test_split_title_only"] = {"n": len(y), "surveys": int(y.sum()), "results": rows}
    sections.append((f"Table II(a2) — same test split, title only (abstract removed)",
                     tabulate(fmt_table(rows), headers=["Method", "Acc.", "Prec.", "Recall", "F1"], tablefmt="github")))

    # ---- (b) Hand-labeled author-profile set ----
    if eval_set and os.path.exists(eval_set):
        ev = pd.read_csv(eval_set, dtype={"Label": str})
        ev = ev[ev["Label"].isin(["0", "1"])].reset_index(drop=True)
        if len(ev):
            print(f"📊 Evaluating on the hand-labeled set ({len(ev)} papers)...")
            frame = prepare_frame(ev)
            y = (ev["Label"] == "0").astype(int).values
            w = ev["StratumWeight"].values if "StratumWeight" in ev.columns else None
            preds = all_methods(frame, survey_proba_for(frame, model_path), model_path, svm, vectorizer,
                                baseline_for(frame))
            rows = {name: metrics(y, p, w) for name, p in preds.items()}

            confident = ev["Confidence"].isin(["H", "M"]).values if "Confidence" in ev.columns else None
            if confident is not None:
                for name, p in preds.items():
                    rows[name]["f1_confident"] = metrics(y[confident], p[confident])["f1"]

            labeled_by = ", ".join(sorted(ev["LabeledBy"].dropna().unique())) if "LabeledBy" in ev.columns else "unknown"
            results["hand_labeled"] = {"n": len(y), "surveys": int(y.sum()), "labeled_by": labeled_by, "results": rows}
            names = {"f1_confident": "F1 (H+M labels)", "prec_reweighted": "Prec. (rw)",
                     "rec_reweighted": "Recall (rw)", "survey_rate_pred": "Flagged (rw)"}
            extra = [c for c in names if c in next(iter(rows.values()))]
            headers = ["Method", "Acc.", "Prec.", "Recall", "F1"] + [names[c] for c in extra]
            sections.append((
                f"Table II(b) — hand-labeled author-profile papers (n = {len(y)}, {y.sum()} surveys; labels: {labeled_by})",
                tabulate(fmt_table(rows, extra), headers=headers, tablefmt="github") +
                "\n\n(rw) = reweighted by sampling stratum to the real mix of papers in the profiles; "
                f"estimated true survey rate: {100 * next(iter(rows.values())).get('survey_rate_true', 0):.1f}%"))

    # ---- Table IV ----
    if authors_glob and glob.glob(authors_glob):
        print("📊 Recalculating author metrics (Table IV)...")
        table = author_impact(authors_glob, model_path)
        results["author_impact"] = table.to_dict(orient="records")
        show = table.drop(columns=["File"]).copy()
        for col in ("Papers excluded", "Citation reduction", "i10-index reduction", "Citation reduction (+ magazine)"):
            show[col] = show[col].map(lambda v: f"{v:.2f}%")
        for col in ("h-index drop", "h-index drop (+ magazine)"):
            show[col] = show[col].map(lambda v: f"{v:.1f}" if isinstance(v, float) else v)
        sections.append((f"Table IV — impact of excluding detected surveys (learned hybrid; {authors_glob})",
                         tabulate(show.values.tolist(), headers=list(show.columns), tablefmt="github") +
                         "\n\nBooks/editorials are not counted as surveys. '(+ magazine)' also excludes magazine "
                         "articles that the classifier flagged but that do not present themselves as surveys."))

    # ---- Report ----
    os.makedirs(report_dir, exist_ok=True)
    md = [f"# Survey classifier evaluation ({results['date']})", "",
          f"Model: `{model_path}` — DistilBERT threshold {results['threshold']:.2f}", "",
          f"Learned hybrid coefficients: `{results['hybrid_coefficients']}`", ""]
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
    parser = argparse.ArgumentParser(description="Regenerate Table II and Table IV")
    parser.add_argument("--model", type=str, default="./distilbert_survey_model")
    parser.add_argument("--eval-set", type=str, default="data/eval_to_label.csv")
    parser.add_argument("--authors", type=str, default="data/proauthor/*.csv")
    parser.add_argument("--baseline-model", type=str, default=None)
    parser.add_argument("--report-dir", type=str, default="reports")
    args = parser.parse_args()
    run_evaluation(args.model, args.eval_set, args.authors, args.baseline_model, args.report_dir)
