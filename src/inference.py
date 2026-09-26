"""
DistilBERT loading and batched survey-probability inference.
"""

import os
import json
from typing import List, Tuple

import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertConfig

from text_utils import MAX_LENGTH

CONFIG_FILE = "survey_config.json"
SURVEY_LABEL = 0  # label ids: 0 = survey, 1 = non-survey


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str) -> Tuple[DistilBertTokenizerFast, DistilBertForSequenceClassification]:
    """
    Load a fine-tuned classifier. Also supports the legacy checkpoint format,
    whose head was a Sequential(Dropout, Linear) saved as 'classifier.1.*'.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    weights = os.path.join(model_path, "model.safetensors")

    legacy = False
    if os.path.exists(weights):
        from safetensors import safe_open
        with safe_open(weights, "pt") as f:
            legacy = "classifier.1.weight" in f.keys()

    if legacy:
        from safetensors.torch import load_file
        config = DistilBertConfig.from_pretrained(model_path, num_labels=2)
        model = DistilBertForSequenceClassification(config)
        state = load_file(weights)
        state = {k.replace("classifier.1.", "classifier."): v for k, v in state.items()
                 if not k.startswith("loss_fct.")}  # unused loss weights saved by the old train.py
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if "position_ids" not in k]
        if missing or unexpected:
            raise RuntimeError(f"Legacy checkpoint mismatch. Missing: {missing}, unexpected: {unexpected}")
    else:
        model = DistilBertForSequenceClassification.from_pretrained(model_path)

    model.to(get_device()).eval()
    return tokenizer, model


def load_config(model_path: str) -> dict:
    path = os.path.join(model_path, CONFIG_FILE)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"threshold": 0.5, "max_length": MAX_LENGTH}


def save_config(model_path: str, config: dict) -> None:
    with open(os.path.join(model_path, CONFIG_FILE), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


@torch.no_grad()
def predict_survey_proba(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int = 32,
    max_length: int = MAX_LENGTH,
    show_progress: bool = False,
) -> np.ndarray:
    """Return P(survey) for each text."""
    device = next(model.parameters()).device
    # Sort by length so each batch pads to a similar size
    order = np.argsort([len(t) for t in texts])
    probs = np.zeros(len(texts), dtype=np.float32)

    batches = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm import tqdm
        batches = tqdm(batches, desc="🤖 Model Processing", unit="batch")

    for start in batches:
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**enc).logits
        probs[idx] = torch.softmax(logits.float(), dim=-1)[:, SURVEY_LABEL].cpu().numpy()
    return probs
