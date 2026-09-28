# Academic Paper Classifier & Author Metrics Pipeline

A comprehensive machine learning pipeline designed to automatically classify scientific literature (distinguishing between **Survey/Review papers** and **Original Research papers**) and compute academic metrics (such as **h-index** and **i10-index**) using hybrid ML classification and keyword-matching logic.

---

## 🚀 Key Features

* **Data Extraction**: Seamlessly fetch publication data for any author from the **Semantic Scholar API**.
* **Real Training Data**: Builds a labeled dataset from real publications via **OpenAlex** (survey-only journals vs. topic- and year-matched research journals).
* **Hybrid Classification**: A fine-tuned **DistilBERT** classifier combined with keyword heuristics — either the paper's OR rule or a **learned hybrid** (logistic regression over the DistilBERT score, title keywords, survey/research phrasing and reference count).
* **Honest Evaluation**: Held-out test split plus a hand-labeled set of real author-profile papers; regenerates Table II and Table IV.
* **Hyperparameter Tuning**: Automated optimization using **Optuna** for DistilBERT hyperparameter search.
* **Academic Metric Analytics**: Calculates key metrics like **h-index** and **i10-index** on the filtered set of original research papers.
* **CUDA Optimization**: GPU-accelerated training and inference with mixed precision.
* **Unified CLI**: Run the entire pipeline through a single entrypoint script (`main.py`).

---

## 📁 Repository Structure

```text
├── data/                      # Directory for data (ignored by git, kept via .gitkeep)
│   └── proauthor/             # Subfolder for raw author files
├── src/                       # Main source code
│   ├── extract_semantic.py    # Semantic Scholar API client to fetch author profile & papers
│   ├── build_dataset.py       # Real training dataset builder (OpenAlex)
│   ├── make_eval_set.py       # Samples author-profile papers for hand labeling
│   ├── text_utils.py          # Shared text cleaning and keyword/cue patterns
│   ├── inference.py           # Model loading and batched survey-probability inference
│   ├── hybrid.py              # OR rule and learned hybrid combiner
│   ├── train.py               # Fine-tuning, threshold selection, hybrid fitting, Optuna tuning
│   ├── classifier.py          # Author-profile classification & metrics recalculation
│   └── evaluate.py            # Regenerates Table II and Table IV
├── reports/                   # Evaluation reports (evaluation.md / .json)
├── visualization/             # Directory containing visualization scripts and plots
├── .env                       # Local environment file (API keys, ignored by git)
├── .env.example               # Template for environment configuration variables
├── .gitignore                 # Standard git exclude file for model checkpoints & datasets
├── main.py                    # Unified command-line interface entry point
└── README.md                  # This documentation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. If using GPU acceleration, install a version of PyTorch compiled with CUDA.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Dependencies
Install the pinned versions the results were produced with:

```bash
pip install -r requirements.txt
```
`optuna` is only needed for `train --tune` and is installed separately.

### 3. Environment Variables
Copy the `.env.example` template to `.env` and fill in your Semantic Scholar API key and Author ID:

```bash
cp .env.example .env
```

Open `.env` and configure:
```ini
SEMANTIC_SCHOLAR_API_KEY=your_actual_api_key_here
SEMANTIC_SCHOLAR_AUTHOR_ID=144019071
```

---

## 📖 Command Line Usage

Use the unified entrypoint `main.py` to run any step of the pipeline.

### 📥 1. Extract Paper Data
Fetch publications for an author from the Semantic Scholar API:
```bash
python main.py extract --author 144019071 --output data/nima.csv
```
Semantic Scholar often splits prolific authors across several IDs (e.g. D. Niyato: 1713586 and 2266084696). Pass them all to merge the profile; duplicate titles are merged. The verified IDs for the five authors in the paper are in `data/author_ids.json`, and their merged profiles in `data/proauthor_s2/`:
```bash
python main.py extract --author 1713586,2266084696,2340230621 --output data/proauthor_merged/niyato.csv
```
Or use **OpenAlex**, which keeps one merged profile per author (set `OPENALEX_API_KEY` in `.env` — free at openalex.org — for large profiles):
```bash
python main.py extract --source openalex --name "Dusit Niyato" --output data/proauthor_openalex/niyato.csv
```
Both record each paper's publication type, used to recognise books and editorials. To add types to an older Semantic Scholar CSV:
```bash
python main.py extract --add-types data/proauthor/auth1.csv
```
No single source is complete for prolific authors, so combine them (one row per title, highest citation count, all publication types):
```bash
python src/combine_profiles.py semantic=data/proauthor/auth1.csv openalex=data/proauthor_openalex/hanzo.csv --output data/proauthor_combined/hanzo.csv
```
Check profiles and classifications against Google Scholar (metrics and top-20 papers in `data/scholar_reference.json`); writes `reports/author_verification.md`:
```bash
python src/verify_authors.py
```

---

### 🧱 Build a Real Training Dataset
Build a labeled dataset from real publications via **OpenAlex** (no API key needed):
```bash
python main.py build-dataset --output data/real_dataset.csv
```
Surveys come from survey-only journals (IEEE COMST, ACM CSUR, AI Review, Computer Science Review). Non-surveys come from research journals in the same topic group (e.g. JSAC, TWC, TPAMI, TSE), sampled to match the surveys' year distribution. Research-venue papers that look like surveys are written to `data/hard_cases_to_label.csv` instead of being used for training.

Then sample papers from real author profiles for a hand-labeled test set (training papers are excluded):
```bash
python main.py make-eval-set --input "data/proauthor/*.csv"
```
Fill the `Label` column of `data/eval_to_label.csv` (0 = survey, 1 = research, `skip` = non-paper).

---

### 🏋️ 2. Train or Tune the Classifier

#### Train DistilBERT:
```bash
python main.py train --dataset data/real_dataset.csv --epochs 3
```
The dataset is split 80/20 into train/test (stratified), and 10% of the training part is held out for validation. Training saves to `./distilbert_survey_model/`:
* the fine-tuned model and tokenizer,
* `survey_config.json` — the decision threshold chosen on the validation split,
* `hybrid_combiner.joblib` — the learned hybrid, fitted on the validation split,
* `data_split.csv` — the exact split, reused by `evaluate`.

The test split is never used for training, early stopping, threshold selection or tuning.

#### Hyperparameter tuning using Optuna:
```bash
python main.py train --dataset data/real_dataset.csv --tune --trials 5
```
Tuning uses the train/validation splits only.

---

### ⚡ 3. Classify and Calculate Metrics
Filter out survey papers and calculate academic indices (h-index & i10-index) before and after filtering:
```bash
python main.py classify --input data/nima.csv
```
This classifies each paper, writes original research papers to `data/Non-Survey-Papers.csv`, writes surveys and non-papers (editorials, errata, ...) to `data/Survey-Papers.csv`, and prints a comparison table showing the change in indices. Only surveys are removed from the metrics: non-papers count in both the original and the filtered values, so the difference comes from the surveys alone. `Prediction` is 0 for a survey and 1 otherwise.

Choose the classifier with `--mode`: `learned` (default), `or` (the paper's DistilBERT OR keyword rule), `model` (DistilBERT only) or `keyword` (title keywords only).

Each paper gets a `Category`:
* `survey` — excluded;
* `non-paper` — books, editorials, errata (from the publication type or title) — never counted as surveys; kept in the metrics but left out of the research-only file;
* `magazine-overview` — a magazine article (e.g. IEEE Communications Magazine, IEEE Network) that the classifier flagged but that does not present itself as a survey (no survey term in the title, no survey phrasing in the abstract, not typed "Review"). Kept by default; add `--exclude-magazine-overviews` to exclude these too;
* `research` — kept.

---

### 📊 4. Evaluate
Regenerate Table II (classification performance, survey = positive class) on the held-out test split and the hand-labeled set, and Table IV (author impact):
```bash
python main.py evaluate --baseline-model ./distilbert_survey_model_synthetic
```
External test on unseen venues: text from the Kaggle arXiv snapshot (`Cornell-University/arxiv`, downloaded with `kagglehub`, no Kaggle account needed), labels from where each paper was published (Foundations and Trends, Annual Reviews, ... vs. JMLR, IEEE TIT, ...; none used in training). Writes `data/kaggle_arxiv_test.csv` and `reports/external_evaluation.md`:
```bash
python src/build_kaggle_test.py
python src/evaluate_external.py
```

Results are written to `reports/evaluation.md` and `reports/evaluation.json`. Table IV reports the main result (surveys excluded) and a sensitivity column that also excludes magazine overviews. Use `--authors "data/proauthor_merged/*.csv"` to run it on other profiles.

To see all option flags:
```bash
python main.py --help
```

---

## 🗂️ Data in this repository

| File | Contents |
|---|---|
| `data/real_dataset.csv` | Training data: 9,624 papers (4,812 surveys / 4,812 research) from OpenAlex, labeled by venue |
| `data/real_dataset_split.csv` | Train / validation / test assignment of each paper (by `OpenAlexId`) used for the reported results |
| `data/eval_to_label.csv` | 328 author-profile papers with survey/research labels (`LabeledBy` says who labeled them) |
| `data/hard_cases_to_label.csv` | Survey-like papers from research journals, excluded from training (unlabeled) |
| `data/proauthor/*.csv` | Original Semantic Scholar profiles of the five authors (source of the evaluation set) |
| `data/proauthor_s2/*.csv` | Semantic Scholar profiles with merged author IDs (`data/author_ids.json`), used for Table IV |
| `data/scholar_reference.json` | Google Scholar metrics and top-20 papers per author, with reference labels |

Metadata comes from [OpenAlex](https://openalex.org) (CC0) and the [Semantic Scholar API](https://www.semanticscholar.org/product/api). Abstracts remain the property of their publishers; they are included for research reproducibility.

---

## 🧪 Tests
Unit tests cover text cleaning, the categorization rules and the metric calculations (no model needed):
```bash
python -m pytest tests
```

---

## 🔒 Security Note
Never commit the `.env` file containing your API keys or any generated `.csv` files inside the `data/` folder to GitHub. The `.gitignore` file has been preconfigured to exclude them.
