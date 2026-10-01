# Survey Excluder: model and evaluation

Code and data for the paper *A Semantic Classification Framework for Identifying Survey Papers and
Mitigating Citation Inflation* (A. Nighot, N. Afraz). Survey Excluder detects survey and review papers
in a publication list from bibliographic metadata alone, and recalculates the h-index, i10-index and
citation count without them. A web interface is in the companion repository
[SurvayExtruderUI](https://github.com/atharva-ni/SurvayExtruderUI).

## How it works

Two models are developed here:

1. **Fine-tuned DistilBERT**: DistilBERT fine-tuned on the venue-labeled dataset; it estimates, from title and
   abstract, how likely a paper is a survey.
2. **Learned hybrid** (the main model, built on the first): a logistic regression that combines that score with DistilBERT's text representation, title keywords, survey or
   research phrasing in the abstract, a missing-abstract flag, the reference count, and magazine, arXiv and
   indexer-review flags. It is fitted on the validation split (title only) and on author-profile papers: 1,441
   labeled by an LLM (`data/llm_labels.csv`, rules in `data/llm_labeling_codebook.md`) and 323 labeled by an author
   (`data/eval_to_label.csv`).

**Decision rules** then place each paper in one category:

| Category | Meaning | In the adjusted metrics |
|---|---|---|
| `survey` | Survey, tutorial or review | Removed |
| `non-paper` | Book, editorial, erratum (from publication type or title) | Kept, never counted as a survey |
| `research` | Original research | Kept |

Papers with only a title count as surveys only if the title says so.

The other methods in the reports (TF-IDF + SVM, the title keyword filter, the original
[Google Scholar Survey Excluder](https://github.com/nimaafraz/Google-Scholar-Survey-Excluder) rules, the indexer's
document type and the Smyth & Cunningham rule) are there for comparison only and live in `src/evaluate.py`.

## Results (learned hybrid; fine-tuned DistilBERT in brackets)

| Test | Result | Report |
|---|---|---|
| Held-out test split (1,925 journal papers) | 93.6% accuracy, 93.2% F1 (95.3%, 95.2%) | `reports/evaluation.md` |
| arXiv papers from 45 venues not used in training (970) | 88.7% F1 (92.0%) | `reports/external_evaluation.md` |
| Author-profile papers, blind-labeled by an author (323) | 90% precision, 80% recall, nested cross-validation (58%, 92%) | `reports/evaluation.md` |
| Google Scholar top-20 papers not seen in training (43) | 40 correct (93%) | `reports/author_verification.md` |
| Five prolific survey authors (Cohort B) vs. five comparison authors (Cohort A) | surveys: 10.0% vs. 5.4% of papers (11.7% vs. 5.9% corrected for classification error), 31.9% vs. 20.5% of citations | `reports/cohort/evaluation.md` |

The adjusted metrics are an additional view of a publication record, suited to aggregate or cohort-level
analysis. At this accuracy they should not be used to judge an individual researcher.

## Setup

```bash
pip install -r requirements.txt     # tested with Python 3.14; install the CUDA build of PyTorch first for GPU use
cp .env.example .env                # add an OpenAlex API key (free) and, optionally, a Semantic Scholar key
python -m pytest tests              # unit tests (no model needed)
```

The trained model used in the paper (about 260 MB) is not stored in this repository. Download
`distilbert_survey_model.zip` from the [v1.0 release](https://github.com/atharva-ni/SurveyExcluderModel/releases/tag/v1.0)
and unzip it in the repository root, or train it yourself with `python main.py train` (DistilBERT, then the learned
hybrid); either way it ends up in `distilbert_survey_model/`.

## Usage

```bash
# Fetch an author's publications (OpenAlex keeps one merged profile per author)
python main.py extract --source openalex --name "Author Name" --output data/author.csv

# Classify them and compare h-index, i10-index and citations with and without surveys
python main.py classify --input data/author.csv
```

`classify` writes the research papers to `data/Non-Survey-Papers.csv` and the surveys and non-papers to
`data/Survey-Papers.csv`. Each row gets `Category`, `SurveyScore` (the learned hybrid's survey probability) and
`Keep` (1 = kept in the survey-excluded metrics, 0 = excluded as a survey).

## Reproducing the paper

| Paper | Command |
|---|---|
| Training data (Table II, Fig. 2) | `python main.py build-dataset` · `python visualization/Bargraphplot.py` |
| Model | `python main.py train` |
| Learned hybrid only, with its cross-validation on author profiles | `python main.py train --hybrid-only --cv 5` |
| Test split and author profiles (Tables III, V) | `python main.py evaluate` |
| Unseen venues (Table IV) | `python src/build_kaggle_test.py` · `python src/evaluate_external.py` |
| Cohorts (Tables VI to VIII) | `python src/build_cohort.py` · `python main.py evaluate --eval-set none --authors "data/cohort_openalex/survey/*.csv" --comparison-authors "data/cohort_openalex/comparison/*.csv" --report-dir reports/cohort` · `python src/check_cohort_authors.py` |
| Google Scholar check | `python src/verify_authors.py` |

## Data in this repository

| File | Contents |
|---|---|
| `data/real_dataset.csv`, `data/real_dataset_split.csv` | Training data (9,624 papers from 23 journals, labeled by venue) and the train/validation/test split |
| `data/hard_cases_to_label.csv` | Survey-like papers from research journals, set aside from training |
| `data/eval_to_label.csv` | 328 author-profile papers: the author's blind labels (`Label`), the LLM labels (`LabelClaude`) and the sampling strata and weights |
| `data/llm_labels.csv`, `data/llm_labeling_codebook.md` | 1,519 author-profile papers labeled by an LLM (1,441 used; 78 not papers) (Claude Opus 5.5) for fitting the learned hybrid, with the sampling strata and weights and the labeling rules |
| `data/proauthor/`, `data/proauthor_s2/`, `data/author_ids.json` | Semantic Scholar profiles of the five survey authors (the second with merged author IDs) |
| `data/kaggle_arxiv_test.csv`, `data/kaggle_arxiv_errors_to_review.csv` | Unseen-venue test set and its misclassified papers |
| `data/cohort_openalex/`, `data/cohort_openalex.json` | OpenAlex profiles of both cohorts (retrieved 28 September 2026), author IDs and the selection record |
| `data/scholar_reference.json` | Google Scholar metrics and top-20 papers per survey author |

Metadata comes from [OpenAlex](https://openalex.org) (CC0), the [Semantic Scholar API](https://www.semanticscholar.org/product/api)
and the arXiv metadata snapshot on [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) (CC0).
Abstracts remain the property of their publishers; they are included for reproducibility.

## Repository layout

```text
main.py            command-line entry point (extract, build-dataset, train, classify, evaluate)
src/               classifier, training, dataset builders and evaluation scripts
tests/             unit tests
data/              datasets behind the reported results
reports/           generated evaluation reports
visualization/     training-data figure and its plotting script
```
