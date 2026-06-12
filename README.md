# Academic Paper Classifier & Author Metrics Pipeline

A comprehensive machine learning pipeline designed to automatically classify scientific literature (distinguishing between **Survey/Review papers** and **Original Research papers**) and compute academic metrics (such as **h-index** and **i10-index**) using hybrid ML classification and keyword-matching logic.

---

## 🚀 Key Features

* **Data Extraction**: Seamlessly fetch publication data for any author from the **Semantic Scholar API**.
* **Hybrid Classification**: Employs a fine-tuned **DistilBERT** sequence classifier combined with rule-based keyword matching (e.g., searching for "survey", "systematic review", "taxonomy") to filter out review papers.
* **Hyperparameter Tuning**: Automated optimization using **Optuna** for DistilBERT hyperparameter search.
* **Academic Metric Analytics**: Calculates key metrics like **h-index** and **i10-index** on the filtered set of original research papers.
* **CUDA Optimization**: GPU-accelerated inference with mixed-precision, memory pinning, and compilation (`torch.compile`) support for maximum throughput.
* **Unified CLI**: Run the entire pipeline through a single entrypoint script (`main.py`).

---

## 📁 Repository Structure

```text
├── data/                      # Directory for data (ignored by git, kept via .gitkeep)
│   └── proauthor/             # Subfolder for raw author files
├── src/                       # Main source code
│   ├── extract_semantic.py    # Semantic Scholar API client to fetch author profile & papers
│   ├── classifier.py          # Unified classifier logic & academic metrics calculator
│   └── train.py               # Fine-tuning script with Optuna hyperparameter optimization
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
Install the required packages:

```bash
pip install pandas transformers optuna scikit-learn requests python-dotenv tabulate tqdm
```

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
Fetch publications for an author ID (e.g., `144019071`) from the Semantic Scholar API:
```bash
python main.py extract --author 144019071 --output data/prof3.csv
```
This will fetch all publications and save the metadata to `data/prof3.csv`.

---

### 🏋️ 2. Train or Tune the Classifier

#### Train standard DistilBERT:
```bash
python main.py train --dataset data/dataset.csv --epochs 3
```
This trains the sequence classifier and saves the fine-tuned model and tokenizer under `./distilbert_survey_model/`.

#### Hyperparameter tuning using Optuna:
```bash
python main.py train --dataset data/dataset.csv --tune --trials 5
```

---

### ⚡ 3. Classify and Calculate Metrics
Filter out survey papers and calculate academic indices (h-index & i10-index) before and after filtering:
```bash
python main.py classify --input data/prof3.csv
```
This classifies each paper, writes original research papers to `data/Non-Survey-Papers.csv`, writes excluded surveys to `data/Survey-Papers.csv`, and prints a comprehensive comparison table showing the change in indices.

To see all option flags:
```bash
python main.py --help
```

---

## 🔒 Security Note
Never commit the `.env` file containing your API keys or any generated `.csv` files inside the `data/` folder to GitHub. The `.gitignore` file has been preconfigured to exclude them.
