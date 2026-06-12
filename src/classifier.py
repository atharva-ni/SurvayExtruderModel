import os
import time
import pandas as pd
import torch
from typing import List, Tuple
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tqdm import tqdm

# ---------------- Survey Keyword Logic ---------------- #
survey_keywords = [
    "survey", "review", "overview", "comparative",
    "taxonomy", "state of the art", "systematic"
]

def keyword_is_survey(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in survey_keywords)

# ---------------- Dataset Wrapper ---------------- #
class SurveyDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: DistilBertTokenizerFast, max_length: int = 256):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

    def __len__(self) -> int:
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx: int) -> dict:
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

# ---------------- Hybrid Classifier Function ---------------- #
def classify_with_hybrid_model(
    df: pd.DataFrame,
    model_path: str,
    batch_size: int = 32,
    threshold: float = 0.8
) -> List[int]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_path, ignore_mismatched_sizes=True
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if device.type == "cuda":
        print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
        try:
            model = torch.compile(model)
            print("🚀 PyTorch model compiled successfully for optimized execution.")
        except Exception:
            print("ℹ️ torch.compile not supported or failed, running model normally.")
    else:
        print("⚠️ Running on CPU (slow inference). Consider using a GPU-enabled environment.")

    # Combine Title and Abstract for classification text
    # Support both title/abstract and Title/Abstract casing
    title_col = 'title' if 'title' in df.columns else 'Title'
    abstract_col = 'abstract' if 'abstract' in df.columns else 'Abstract'
    
    texts = (df[title_col].astype(str).fillna('') + " " +
             df[abstract_col].astype(str).fillna('')).str.lower().tolist()

    dataset = SurveyDataset(texts, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Safer default for Windows
        pin_memory=True if device.type == "cuda" else False
    )

    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Classifying Papers", unit="batch"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            survey_probs = probs[:, 0].cpu().numpy()  # Class 0: Survey

            all_probs.extend(survey_probs)

    model_preds = [0 if prob > threshold else 1 for prob in all_probs]
    keyword_preds = [0 if keyword_is_survey(t) else None for t in texts]

    final_preds = [
        kp if kp is not None else mp
        for kp, mp in zip(keyword_preds, model_preds)
    ]
    return final_preds

# ---------------- Index Calculation ---------------- #
def calculate_indices(df: pd.DataFrame, citation_col: str) -> Tuple[int, int]:
    citations = df[citation_col].fillna(0).astype(int).sort_values(ascending=False).values
    h_index = sum(c >= (i + 1) for i, c in enumerate(citations))
    i10_index = sum(c >= 10 for c in citations)
    return h_index, i10_index

# ---------------- Main Filtering Pipeline ---------------- #
def run_classification_pipeline(
    input_csv: str,
    output_csv: str,
    survey_csv: str,
    model_path: str,
    batch_size: int = 32,
    threshold: float = 0.8
) -> None:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    print(f"✅ Loaded CSV '{input_csv}' with columns: {df.columns.tolist()}")

    # Normalize Title/Abstract columns
    title_col = next((c for c in df.columns if c.lower() == 'title'), None)
    abstract_col = next((c for c in df.columns if c.lower() == 'abstract'), None)
    
    # Identify citation column name variations
    citation_col = next((c for c in df.columns if c.lower() in ['citationcount', 'n_citation', 'num_citations', 'citations']), None)

    if not title_col or not abstract_col:
        raise KeyError("CSV must contain title and abstract columns.")

    if not citation_col:
        print("⚠️ No citation count column found. Creating an empty 'citations' column defaulted to 0.")
        citation_col = 'citations'
        df[citation_col] = 0

    df[title_col] = df[title_col].fillna('').astype(str)
    df[abstract_col] = df[abstract_col].fillna('').astype(str)
    df[citation_col] = df[citation_col].fillna(0).astype(int)

    print(f"📊 Total publications to analyze: {len(df)}")

    # Classify
    df['Prediction'] = classify_with_hybrid_model(df, model_path, batch_size, threshold)

    survey_df = df[df['Prediction'] == 0]
    non_survey_df = df[df['Prediction'] == 1].drop(columns=['Prediction'])

    # Ensure output folders exist
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(survey_csv)), exist_ok=True)

    non_survey_df.to_csv(output_csv, index=False)
    survey_df.to_csv(survey_csv, index=False)

    total_papers = len(df)
    excluded_papers = len(survey_df)

    excluded_citations = survey_df[citation_col].sum()
    total_citations = df[citation_col].sum()

    percent_papers_excluded = (excluded_papers / total_papers) * 100 if total_papers else 0
    percent_citations_excluded = (excluded_citations / total_citations) * 100 if total_citations else 0

    total_h_index, total_i10 = calculate_indices(df, citation_col)
    non_survey_h_index, non_survey_i10 = calculate_indices(non_survey_df, citation_col)

    print(f"\n📊 Survey Exclusions Summary:")
    print(f"  • Papers excluded as surveys: {excluded_papers} / {total_papers} ({percent_papers_excluded:.2f}%)")
    print(f"  • Citations excluded: {excluded_citations} / {total_citations} ({percent_citations_excluded:.2f}%)")

    comparison = [
        ["Total Papers", total_papers, len(non_survey_df)],
        ["Total Citations", total_citations, total_citations - excluded_citations],
        ["H-Index", total_h_index, non_survey_h_index],
        ["i10-Index", total_i10, non_survey_i10],
    ]

    print("\n📋 Comparison Table (Before vs. After filtering out surveys):")
    print(tabulate(comparison, headers=["Metric", "With Surveys", "Without Surveys"], tablefmt="grid"))


if __name__ == "__main__":
    # Default local run on prof3.csv (produced by semantic extraction)
    input_file = os.path.join("data", "prof3.csv")
    if not os.path.exists(input_file):
        # Fall back to any csv in data
        import glob
        csvs = glob.glob(os.path.join("data", "*.csv"))
        if csvs:
            input_file = csvs[0]
            
    output_file = os.path.join("data", "Non-Survey-Papers.csv")
    survey_file = os.path.join("data", "Survey-Papers.csv")
    model_dir = './distilbert_survey_model'

    if os.path.exists(input_file):
        run_classification_pipeline(
            input_csv=input_file,
            output_csv=output_file,
            survey_csv=survey_file,
            model_path=model_dir
        )
    else:
        print(f"⚠️ Default input file '{input_file}' not found. Please specify input file or run data extractor first.")
