"""
Final Survey/Non-Research Paper Classifier
============================================
Excludes surveys, guest editorials, and other non-research papers.
Keeps only primary research papers.
"""

import os
import pandas as pd
import torch
from typing import List, Tuple
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertConfig
from tqdm import tqdm
from safetensors.torch import load_file


# ============================================================================
# KEYWORD-BASED CLASSIFICATION (Title-only)
# ============================================================================

def keyword_is_survey(title: str, abstract: str) -> bool:
    """
    Check if paper is survey/editorial based on TITLE ONLY.
    Returns True for surveys, reviews, editorials, and other non-research papers.
    """
    title_lower = title.lower().strip()
    
    # First, exclude guest editorials (they're not surveys but still non-research)
    # We check them separately to avoid false positives with "review" in editorial titles
    if any(kw in title_lower for kw in ["guest editorial", "editorial:"]):
        return True
    
    # Strong survey/review indicators in title
    strong_keywords = [
        "survey",
        "review",  # ✅ Added: Includes "systematic review", "literature review", etc.
        "systematic review", 
        "literature review",
        "comprehensive survey",
        "state-of-the-art",
        "state of the art"
    ]
    
    for keyword in strong_keywords:
        if keyword in title_lower:
            return True
        
    return False


# ============================================================================
# DATASET WRAPPER
# ============================================================================

class SurveyDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: DistilBertTokenizerFast, max_length: int = 256):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

    def __len__(self) -> int:
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx: int) -> dict:
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


# ============================================================================
# HYBRID MODEL CLASSIFICATION
# ============================================================================

def classify_with_hybrid_model(
    df: pd.DataFrame,
    model_path: str = './distilbert_survey_model',
    batch_size: int = 32,
    threshold: float = 0.85,
    use_keywords: bool = True
) -> List[int]:
    """
    Classify papers using hybrid approach:
    1. Keywords in title → non-research paper (if enabled)
    2. Model prediction → research vs non-research
    
    Returns:
        List of predictions: 0 = non-research (exclude), 1 = research (keep)
    """
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    config = DistilBertConfig.from_pretrained(model_path)
    model = DistilBertForSequenceClassification(config=config)
    
    # Reconstruct custom classifier
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(config.dim, 2)
    )
    
    # Load trained weights
    state_dict = load_file(f"{model_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Display GPU info
    if torch.cuda.is_available():
        print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU (slower)")
    
    print(f"🎯 Model threshold: {threshold}")
    print(f"📝 Keyword override: {'Enabled (Title-only)' if use_keywords else 'Disabled'}")

    # Prepare texts (title + abstract)
    texts = (df['title'].astype(str).fillna('') + " " +
             df['abstract'].astype(str).fillna('')).tolist()

    # Create dataset and dataloader
    dataset = SurveyDataset(texts, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Get model predictions
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🤖 Model Processing", unit="batch"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            survey_probs = probs[:, 0].cpu().numpy()
            all_probs.extend(survey_probs)

    # Convert probabilities to predictions
    model_preds = [0 if prob > threshold else 1 for prob in all_probs]
    
    # Apply keyword override if enabled
    if use_keywords:
        final_preds = []
        keyword_overrides = 0
        
        for idx, (title, abstract, model_pred) in enumerate(zip(
            df['title'].fillna(''), 
            df['abstract'].fillna(''), 
            model_preds
        )):
            if keyword_is_survey(title, abstract):
                final_preds.append(0)  # Non-research
                if model_pred == 1:  # Model disagreed
                    keyword_overrides += 1
            else:
                final_preds.append(model_pred)
        
        print(f"📊 Keyword overrides: {keyword_overrides} papers")
        return final_preds
    else:
        return model_preds


# ============================================================================
# INDEX CALCULATION
# ============================================================================

def calculate_indices(df: pd.DataFrame) -> Tuple[int, int]:
    """Calculate h-index and i10-index from citation counts."""
    citations = df['citationCount'].fillna(0).astype(int).sort_values(ascending=False).values
    h_index = sum(c >= (i + 1) for i, c in enumerate(citations))
    i10_index = sum(c >= 10 for c in citations)
    return h_index, i10_index


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def exclude_predicted_surveys(
    input_csv: str,
    output_csv: str,
    survey_csv: str = 'Survey-Papers.csv',
    model_path: str = './distilbert_survey_model',
    threshold: float = 0.85,
    use_keywords: bool = True,
    batch_size: int = 32
) -> None:
    """
    Main pipeline to classify and separate research vs non-research papers.
    
    Args:
        input_csv: Input CSV file path
        output_csv: Output file for research papers
        survey_csv: Output file for non-research papers
        model_path: Path to trained model
        threshold: Classification threshold (0.85 = 90.24% accuracy)
        use_keywords: Whether to use keyword override
        batch_size: Batch size for DataLoader
    """
    
    # Validate input file
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    # Load data
    df = pd.read_csv(input_csv)
    
    required_cols = {'title', 'abstract', 'citationCount'}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"CSV must contain: {required_cols}")

    # Clean data
    df['title'] = df['title'].fillna('').astype(str)
    df['abstract'] = df['abstract'].fillna('').astype(str)
    df['citationCount'] = df['citationCount'].fillna(0).astype(int)

    print(f"✅ Loaded {len(df)} papers from '{input_csv}'\n")

    # Classify papers
    predictions = classify_with_hybrid_model(
        df, 
        model_path=model_path,
        batch_size=batch_size,
        threshold=threshold,
        use_keywords=use_keywords
    )

    # Add predictions to dataframe
    df['Prediction'] = predictions

    # Split into research and non-research
    non_research_df = df[df['Prediction'] == 0]  # Surveys, editorials, etc.
    research_df = df[df['Prediction'] == 1].drop(columns=['Prediction'])  # Keep only research

    # Save files
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(survey_csv)), exist_ok=True)
    research_df.to_csv(output_csv, index=False)
    non_research_df.to_csv(survey_csv, index=False)

    # Calculate statistics
    total_papers = len(df)
    excluded_papers = len(non_research_df)
    excluded_citations = non_research_df['citationCount'].sum()
    total_citations = df['citationCount'].sum()

    percent_papers_excluded = (excluded_papers / total_papers) * 100 if total_papers else 0
    percent_citations_excluded = (excluded_citations / total_citations) * 100 if total_citations else 0

    total_h_index, total_i10 = calculate_indices(df)
    research_h_index, research_i10 = calculate_indices(research_df)

    # Print summary
    print(f"\n📊 Papers excluded as non-research: {excluded_papers} ({percent_papers_excluded:.2f}%)")
    print(f"📉 Citations excluded: {excluded_citations} ({percent_citations_excluded:.2f}%)")

    # Comparison table
    comparison = [
        ["Total Papers", total_papers, len(research_df)],
        ["Total Citations", total_citations, total_citations - excluded_citations],
        ["H-Index", total_h_index, research_h_index],
        ["i10-Index", total_i10, research_i10],
    ]

    print("\n📋 Comparison Table:")
    print(tabulate(comparison, headers=["Metric", "With Non-Research", "Research Only"], tablefmt="grid"))

    # Final output
    print(f"\n✅ Research papers saved to: '{output_csv}'")
    print(f"✅ Non-research papers (surveys/editorials) saved to: '{survey_csv}'")


# Alias/Wrapper to preserve pipeline name called by main.py
def run_classification_pipeline(
    input_csv: str,
    output_csv: str,
    survey_csv: str,
    model_path: str,
    batch_size: int = 32,
    threshold: float = 0.85
) -> None:
    exclude_predicted_surveys(
        input_csv=input_csv,
        output_csv=output_csv,
        survey_csv=survey_csv,
        model_path=model_path,
        threshold=threshold,
        use_keywords=True,
        batch_size=batch_size
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configuration
    input_csv = './data/proauthor/auth1.csv'
    output_csv = 'data/Non-Survey-Papers.csv'
    survey_csv = 'data/Survey-Papers.csv'
    model_dir = './distilbert_survey_model'

    # Run classification
    exclude_predicted_surveys(
        input_csv=input_csv, 
        output_csv=output_csv, 
        survey_csv=survey_csv, 
        model_path=model_dir,
        threshold=0.85,  
        use_keywords=True  
    )
    
    print(f"\n{'='*60}")
    print("✅ PROCESS COMPLETE!")
    print(f"{'='*60}")
    print(f"\nResearch papers: {output_csv}")
    print(f"Non-research papers: {survey_csv}")
