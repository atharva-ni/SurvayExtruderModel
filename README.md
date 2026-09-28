# Survey Excluder: model and evaluation

Code and data for the paper *A Semantic Classification Framework for Identifying Survey Papers and
Mitigating Citation Inflation* (A. Nighot, N. Afraz). Survey Excluder detects survey and review papers
in a publication list from bibliographic metadata alone, and recalculates the h-index, i10-index and
citation count without them.

## How it works

1. A fine-tuned **DistilBERT** model estimates, from title and abstract, how likely a paper is a survey.
2. A **learned hybrid** (logistic regression) combines that score with title keywords, survey or research
   phrasing in the abstract, a missing-abstract flag and the reference count.
3. **Decision rules** place each paper in one category:

| Category | Meaning | In the adjusted metrics |
|---|---|---|
| `survey` | Survey, tutorial or review | Removed |
| `magazine-overview` | Magazine article flagged by the model that does not present itself as a survey | Kept (optionally removed) |
| `non-paper` | Book, editorial, erratum (from publication type or title) | Kept, never counted as a survey |
| `research` | Original research | Kept |

Papers with only a title count as surveys only if the title says so.

## Results

| Test | Result | Report |
|---|---|---|
| Held-out test split (1,925 journal papers) | 95.4% accuracy, 95.3% F1 | `reports/evaluation.md` |
| arXiv papers from 45 venues not used in training (970) | 91.8% F1 | `reports/external_evaluation.md` |
| Author-profile papers, blind-labeled by an author (323) | 72% precision, 63% recall (default system) | `reports/evaluation.md` |
| Google Scholar top-20 papers not seen in training (69) | 66 correct (96%) | `reports/author_verification.md` |
| Five prolific survey authors vs. five comparison authors | surveys: 9.8% vs. 4.9% of papers, 28.7% vs. 15.9% of citations | `reports/cohort/evaluation.md` |

The adjusted metrics are an additional view of a publication record, suited to aggregate or cohort-level
analysis. At this accuracy they should not be used to judge an individual researcher.

## Setup

```bash
pip install -r requirements.txt     # tested with Python 3.14; install the CUDA build of PyTorch first for GPU use
cp .env.example .env                # add an OpenAlex API key (free) and, optionally, a Semantic Scholar key
python -m pytest tests              # unit tests (no model needed)
```

The trained model (about 260 MB) is not stored in this repository. Train it with `python main.py train`;
it is saved to `distilbert_survey_model/`.

## Usage

```bash
# Fetch an author's publications (OpenAlex keeps one merged profile per author)
python main.py extract --source openalex --name "Author Name" --output data/author.csv

# Classify them and compare h-index, i10-index and citations with and without surveys
python main.py classify --input data/author.csv
```

`classify` writes the research papers to `data/Non-Survey-Papers.csv` and the surveys and non-papers to
`data/Survey-Papers.csv`. Each row gets `Category`, `SurveyScore` and `Prediction` (0 = survey, 1 = other).
Options: `--mode learned|or|model|keyword` and `--exclude-magazine-overviews`.

## Reproducing the paper

| Paper | Command |
|---|---|
| Training data (Table II, Fig. 2) | `python main.py build-dataset` · `python visualization/Bargraphplot.py` |
| Model | `python main.py train` (thresholds only: `python src/train.py --retune-thresholds`) |
| Test split and author profiles (Tables III, V) | `python src/evaluate.py --baseline-model ./distilbert_survey_model_synthetic --authors "data/proauthor_s2/*.csv"` |
| Unseen venues (Table IV) | `python src/build_kaggle_test.py` · `python src/evaluate_external.py` |
| Cohorts (Tables VI, VII) | `python src/build_cohort.py` · `python src/evaluate.py --eval-set none --authors "data/cohort_openalex/survey/*.csv" --comparison-authors "data/cohort_openalex/comparison/*.csv" --report-dir reports/cohort` · `python src/check_cohort_authors.py` |
| Google Scholar check | `python src/verify_authors.py` |

The "earlier model" baseline (`distilbert_survey_model_synthetic`) is the previous, synthetic-data version of
the classifier and is not distributed; without it, the evaluation scripts skip that row.

## Data in this repository

| File | Contents |
|---|---|
| `data/real_dataset.csv`, `data/real_dataset_split.csv` | Training data (9,624 papers from 23 journals, labeled by venue) and the train/validation/test split |
| `data/hard_cases_to_label.csv` | Survey-like papers from research journals, set aside from training |
| `data/eval_to_label.csv` | 328 author-profile papers: the author's blind labels (`Label`), the LLM labels (`LabelClaude`) and the sampling strata and weights |
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
visualization/     plotting scripts and the training-data figure
```
