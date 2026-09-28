"""
External evaluation on the Kaggle arXiv test set (unseen venues)
================================================================
Runs every method from evaluate.py on data/kaggle_arxiv_test.csv (built by
build_kaggle_test.py) and writes reports/external_evaluation.md / .json:

  * accuracy, precision, recall, F1 with bootstrap 95% confidence intervals,
  * precision expected at the survey rate of real author profiles (7.3%),
  * the paired bootstrap difference in F1 between each hybrid and DistilBERT only,
  * recall on surveys per venue and results per topic group.
"""

import os
import json
import argparse
from datetime import date

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from tabulate import tabulate

from text_utils import paper_text
from classifier import prepare_frame
from evaluate import metrics, all_methods, survey_proba_for
from train import SPLIT_FILE

PROFILE_SURVEY_RATE = 0.080  # estimated survey rate of author-profile papers (reports/evaluation.md)
N_BOOT = 2000


def f1_score(y, p) -> float:
    tp = np.sum((y == 1) & (p == 1))
    return 2 * tp / max(1, 2 * tp + np.sum((y == 0) & (p == 1)) + np.sum((y == 1) & (p == 0)))


def precision_at_rate(y, p, rate=PROFILE_SURVEY_RATE) -> float:
    tpr = np.mean(p[y == 1] == 1)
    fpr = np.mean(p[y == 0] == 1)
    return tpr * rate / max(1e-9, tpr * rate + fpr * (1 - rate))


def bootstrap(y, preds: dict, reference: str, seed=42) -> dict:
    """95% CIs for F1 and for the paired F1 difference against the reference method."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    out = {}
    ref = np.array([f1_score(y[i], preds[reference][i]) for i in idx])
    for name, p in preds.items():
        f1s = np.array([f1_score(y[i], p[i]) for i in idx])
        diff = f1s - ref
        out[name] = {"f1_ci": np.percentile(f1s, [2.5, 97.5]).tolist(),
                     "diff_vs_ref": float(np.mean(diff)),
                     "diff_ci": np.percentile(diff, [2.5, 97.5]).tolist()}
    return out


def pct(v) -> str:
    return f"{100 * v:.1f}%"


def run(test_csv: str, model_path: str, baseline_model: str, report_dir: str) -> dict:
    df = pd.read_csv(test_csv)
    y = (df["Label"] == 0).astype(int).values

    split = pd.read_csv(os.path.join(model_path, SPLIT_FILE))
    train_df = split[split["Split"] == "train"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = vectorizer.fit_transform([paper_text(t, a) for t, a in zip(train_df["Title"], train_df["Abstract"])])
    svm = LinearSVC(C=1.0).fit(X_train, (train_df["Label"] == 0).astype(int))

    def baseline_for(frame):
        if not baseline_model or not os.path.isdir(baseline_model):
            return None
        return (f"Earlier model: {os.path.basename(os.path.normpath(baseline_model))} (OR, th=0.8)",
                survey_proba_for(frame, baseline_model), 0.8)

    results = {"date": str(date.today()), "test_set": test_csv, "n": len(y), "surveys": int(y.sum())}
    sections = []
    for variant, frame_df in (("title + abstract", df), ("title only", df.assign(Abstract=""))):
        print(f"📊 Evaluating ({variant}) on {len(y)} papers...")
        frame = prepare_frame(frame_df)
        preds = all_methods(frame, survey_proba_for(frame, model_path), model_path, svm, vectorizer,
                            baseline_for(frame))
        boot = bootstrap(y, preds, reference="DistilBERT only")
        rows = {}
        for name, p in preds.items():
            m = metrics(y, p)
            m["prec_at_profile_rate"] = precision_at_rate(y, p)
            m.update(boot[name])
            rows[name] = m
        results[variant] = rows

        table = [[name, pct(m["acc"]), pct(m["prec"]), pct(m["rec"]), pct(m["f1"]),
                  f"{pct(m['f1_ci'][0])}–{pct(m['f1_ci'][1])}",
                  "" if name == "DistilBERT only" else
                  f"{100 * m['diff_vs_ref']:+.1f} ({100 * m['diff_ci'][0]:+.1f} to {100 * m['diff_ci'][1]:+.1f})",
                  pct(m["prec_at_profile_rate"])] for name, m in rows.items()]
        sections.append((f"Test set — {variant} (n = {len(y)}, {y.sum()} surveys)", tabulate(
            table, headers=["Method", "Acc.", "Prec.", "Recall", "F1", "F1 95% CI", "ΔF1 vs DistilBERT (95% CI)",
                            f"Prec. at {100 * PROFILE_SURVEY_RATE:.1f}% surveys"], tablefmt="github")))

        if variant == "title + abstract":
            main = ["DistilBERT only", "Validation-fitted hybrid (+ reference count)", "Learned hybrid"]
            by_venue = []
            for venue, g in df[df["Label"] == 0].groupby("Venue"):
                i = g.index.values
                by_venue.append([venue, len(i)] + [pct(preds[n][i].mean()) for n in main])
            by_venue.sort(key=lambda r: -r[1])
            sections.append(("Recall on surveys per venue (title + abstract)",
                             tabulate(by_venue, headers=["Venue", "Surveys"] + main, tablefmt="github")))
            by_group = []
            for group, g in df.groupby("Group"):
                i = g.index.values
                by_group.append([group, len(i)] + [pct(f1_score(y[i], preds[n][i])) for n in main])
            sections.append(("F1 per topic group (title + abstract)",
                             tabulate(by_group, headers=["Group", "Papers"] + main, tablefmt="github")))

    os.makedirs(report_dir, exist_ok=True)
    md = [f"# External evaluation on Kaggle arXiv papers from unseen venues ({results['date']})", "",
          f"Test set: `{test_csv}` — text (title, abstract) from the Kaggle arXiv snapshot, labels from the venue "
          "where each paper was published; none of the venues is used in training. Model: "
          f"`{model_path}`. CIs: {N_BOOT} bootstrap resamples. 'Prec. at {100 * PROFILE_SURVEY_RATE:.1f}% surveys' "
          "is the precision expected when surveys are as rare as in real author profiles.", ""]
    for title, table in sections:
        md += [f"## {title}", "", table, ""]
        print(f"\n{title}\n{table}")
    with open(os.path.join(report_dir, "external_evaluation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(report_dir, "external_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n✅ Report saved to '{os.path.join(report_dir, 'external_evaluation.md')}'")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", default="data/kaggle_arxiv_test.csv")
    parser.add_argument("--model", default="./distilbert_survey_model")
    parser.add_argument("--baseline-model", default="./distilbert_survey_model_synthetic")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    run(args.test, args.model, args.baseline_model, args.report_dir)
