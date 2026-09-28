"""
Hand-Labeled Evaluation Set Sampler
===================================
Samples papers from real author profiles (e.g. data/proauthor/*.csv) for
manual labeling. This labeled set is the honest test set: it comes from the
real-world distribution the classifier is applied to, not from the training
venues.

Papers that also appear in the training dataset are excluded. Sampling is
stratified so the rare, hard cases are well represented:
  * title_kw     - survey terms in the title
  * abstract_cue - survey phrasing in the abstract but not in the title
  * random       - everything else (mostly research, plus creatively titled surveys)
The 'Stratum' column is kept so results can be reweighted to the true mix.
"""

import os
import re
import glob
import argparse

import pandas as pd

TITLE_KW = re.compile(
    r"\b(?:survey|surveys|review|reviews|tutorial|overview|taxonomy|state[- ]of[- ]the[- ]art|"
    r"systematic mapping|primer|roadmap|comparative study|literature)\b",
    re.IGNORECASE,
)

ABSTRACT_CUE = re.compile(
    r"\b(?:this (?:survey|review|tutorial|article reviews|paper reviews|paper surveys)|we (?:survey|review)|"
    r"(?:comprehensive|systematic|extensive) (?:overview|survey|review)|we provide an overview|"
    r"state[- ]of[- ]the[- ]art (?:research|approaches|techniques|solutions)|literature review|"
    r"open (?:research )?(?:issues|challenges)|future research directions)\b",
    re.IGNORECASE,
)

MIN_ABSTRACT_WORDS = 30


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


def make_eval_set(
    input_glob: str = "data/proauthor/*.csv",
    training_csv: str = "data/real_dataset.csv",
    output_csv: str = "data/eval_to_label.csv",
    n_title_kw: int = 120,
    n_abstract_cue: int = 120,
    n_random: int = 160,
    seed: int = 42,
) -> pd.DataFrame:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files match '{input_glob}'")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["SourceFile"] = os.path.basename(path)
        frames.append(df)
    papers = pd.concat(frames, ignore_index=True)

    papers["title"] = papers["title"].fillna("").astype(str)
    papers["abstract"] = papers["abstract"].fillna("").astype(str)
    papers = papers[papers["abstract"].str.split().str.len() >= MIN_ABSTRACT_WORDS]

    # Co-authored papers appear in several profiles; keep one copy
    papers["_key"] = papers["title"].map(norm_title)
    papers = papers[papers["_key"] != ""].drop_duplicates("_key")

    # Exclude anything the model is trained on
    if os.path.exists(training_csv):
        train_keys = set(pd.read_csv(training_csv, usecols=["Title"])["Title"].map(norm_title))
        before = len(papers)
        papers = papers[~papers["_key"].isin(train_keys)]
        print(f"🧹 Removed {before - len(papers)} papers that are in the training set '{training_csv}'")
    else:
        print(f"⚠️  Training set '{training_csv}' not found; overlap with training data NOT removed")

    if "references" in papers.columns:
        papers["ReferenceCount"] = papers["references"].fillna("").map(
            lambda s: len([r for r in str(s).split(";") if r.strip()])
        )

    title_hit = papers["title"].str.contains(TITLE_KW)
    abstract_hit = ~title_hit & papers["abstract"].str.contains(ABSTRACT_CUE)
    papers["Stratum"] = "random"
    papers.loc[abstract_hit, "Stratum"] = "abstract_cue"
    papers.loc[title_hit, "Stratum"] = "title_kw"

    wanted = {"title_kw": n_title_kw, "abstract_cue": n_abstract_cue, "random": n_random}
    samples = []
    for stratum, n in wanted.items():
        pool = papers[papers["Stratum"] == stratum]
        samples.append(pool.sample(n=min(n, len(pool)), random_state=seed))
    sample = pd.concat(samples).sample(frac=1, random_state=seed)

    # Stratum weight = pool size / sampled size, for reweighting to the true mix
    pool_sizes = papers["Stratum"].value_counts()
    sample_sizes = sample["Stratum"].value_counts()
    sample["StratumWeight"] = sample["Stratum"].map(pool_sizes / sample_sizes).round(3)

    cols = ["id", "title", "abstract", "venue", "type", "year", "citationCount", "ReferenceCount",
            "SourceFile", "Stratum", "StratumWeight"]
    out = sample[[c for c in cols if c in sample.columns]].rename(columns={"title": "Title", "abstract": "Abstract"})
    out["Label"] = ""   # 0 = survey/review/tutorial, 1 = original research, 'skip' = not a paper
    out["Notes"] = ""

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"\n📋 Candidate pool after filtering: {len(papers)} papers")
    print(pool_sizes.to_string())
    print(f"\n✅ Saved {len(out)} papers to label in '{output_csv}'")
    print(sample_sizes.to_string())
    print("\n✍️  Fill the 'Label' column: 0 = survey/review/tutorial, 1 = original research,")
    print("   'skip' = editorials, errata, books and other non-papers.")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample author-profile papers for manual labeling")
    parser.add_argument("--input", type=str, default="data/proauthor/*.csv", help="Glob of author profile CSVs")
    parser.add_argument("--training", type=str, default="data/real_dataset.csv", help="Training CSV to exclude")
    parser.add_argument("--output", type=str, default="data/eval_to_label.csv")
    parser.add_argument("--n-title-kw", type=int, default=120)
    parser.add_argument("--n-abstract-cue", type=int, default=120)
    parser.add_argument("--n-random", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    make_eval_set(args.input, args.training, args.output, args.n_title_kw, args.n_abstract_cue, args.n_random, args.seed)
