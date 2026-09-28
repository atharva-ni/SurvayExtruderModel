"""
DistilBERT Survey Classifier Training
=====================================
  1. Stratified split: 20% test, then 10% of the remainder as validation.
     The test split is never used for training, early stopping or tuning.
  2. Fine-tune DistilBERT (survey = label 0 = positive class for all metrics).
     A share of training papers is shown with the title only, because many
     papers in author profiles have no abstract.
  3. Choose the DistilBERT decision threshold on the validation split
     (with and without abstracts), searching 0.01-0.99.
  4. Fit the learned hybrid combiner on the same validation data.
The split is saved with the model (data_split.csv) so evaluation reuses it.
"""

import os
import random
import warnings
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

from text_utils import MAX_LENGTH, paper_text
from inference import predict_survey_proba, save_config, SURVEY_LABEL
from hybrid import LearnedHybrid, best_f1_threshold

warnings.simplefilter("ignore", category=FutureWarning)

BASE_MODEL = "distilbert-base-uncased"
SPLIT_FILE = "data_split.csv"
TITLE_ONLY_RATE = 0.25  # share of training papers shown without their abstract


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------- Data -------------------- #
def load_dataset(dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    df = pd.read_csv(dataset_path)
    df = df[df["Label"].isin([0, 1])].dropna(subset=["Title", "Abstract"]).copy()
    df["Label"] = df["Label"].astype(int)
    df["Text"] = [paper_text(t, a) for t, a in zip(df["Title"], df["Abstract"])]
    df = df[df["Text"].str.len() > 20]
    df = df.drop_duplicates(subset="Text").reset_index(drop=True)
    return df


def split_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """80/20 train/test split, then 10% of train held out for validation (as in the paper)."""
    rest, test = train_test_split(df.index, test_size=0.2, stratify=df["Label"], random_state=seed)
    train, val = train_test_split(rest, test_size=0.1, stratify=df.loc[rest, "Label"], random_state=seed)
    df = df.copy()
    df.loc[train, "Split"] = "train"
    df.loc[val, "Split"] = "val"
    df.loc[test, "Split"] = "test"
    return df


def title_only_augment(df: pd.DataFrame, rate: float = TITLE_ONLY_RATE, seed: int = 42) -> pd.DataFrame:
    """Drop the abstract for a random share of papers (in place of, not in addition to, the full text)."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    drop = rng.random(len(df)) < rate
    df.loc[drop, "Text"] = [paper_text(t, "") for t in df.loc[drop, "Title"]]
    df.loc[drop, "Abstract"] = ""
    return df


def with_title_only_copies(df: pd.DataFrame) -> pd.DataFrame:
    """Each paper twice: with its abstract and with the title only."""
    title_only = df.copy()
    title_only["Abstract"] = ""
    title_only["Text"] = [paper_text(t, "") for t in title_only["Title"]]
    return pd.concat([df, title_only], ignore_index=True)


class SurveyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item


# -------------------- Metrics (survey = positive) -------------------- #
def survey_metrics(is_survey_true: np.ndarray, is_survey_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        is_survey_true, is_survey_pred, average="binary", pos_label=1, zero_division=0
    )
    return {"accuracy": accuracy_score(is_survey_true, is_survey_pred),
            "precision": precision, "recall": recall, "f1": f1}


def compute_metrics(pred):
    logits, labels = pred.predictions, pred.label_ids
    is_survey_pred = (np.argmax(logits, axis=1) == SURVEY_LABEL).astype(int)
    return survey_metrics((labels == SURVEY_LABEL).astype(int), is_survey_pred)


class WeightedLossTrainer(Trainer):
    """Cross-entropy with class weights (the Trainer ignores model.loss_fct)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def build_model(dropout: float = 0.2) -> DistilBertForSequenceClassification:
    return DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "survey", 1: "non-survey"},
        label2id={"survey": 0, "non-survey": 1},
        dropout=dropout,
        attention_dropout=dropout,
        seq_classif_dropout=dropout,
    )


def make_trainer(model, tokenizer, train_df, val_df, output_dir, epochs, batch_size, lr,
                 weight_decay=0.01, seed=42, save_best=True) -> Trainer:
    def encode(frame):
        enc = tokenizer(frame["Text"].tolist(), truncation=True, max_length=MAX_LENGTH)
        return SurveyDataset(enc, frame["Label"].values)

    labels = train_df["Label"].values
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    total_steps = epochs * int(np.ceil(len(train_df) / batch_size))

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_steps=int(0.1 * total_steps),
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch" if save_best else "no",
        load_best_model_at_end=save_best,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )
    return WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=encode(train_df),
        eval_dataset=encode(val_df),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if save_best else None,
        class_weights=torch.tensor(weights, dtype=torch.float),
    )


# -------------------- Training -------------------- #
def run_training(dataset_path, output_dir="./distilbert_survey_model", epochs=3, batch_size=16, lr=2e-5, seed=42):
    set_seed(seed)
    print(f"✅ Using device: {'cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu'}")

    df = split_dataset(load_dataset(dataset_path), seed=seed)
    train_df, val_df, test_df = (df[df["Split"] == s] for s in ("train", "val", "test"))
    print(f"📊 {len(df)} papers → train {len(train_df)}, val {len(val_df)}, test {len(test_df)} "
          f"(surveys: {(df['Label'] == 0).sum()}, non-surveys: {(df['Label'] == 1).sum()})")

    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    trainer = make_trainer(build_model(), tokenizer, title_only_augment(train_df, seed=seed), val_df,
                           output_dir=os.path.join("results", "training"),
                           epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)

    print("\n🚀 Training started...\n")
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    model = trainer.model.eval()

    # ---- Threshold and hybrid combiner, chosen on validation papers with and without abstracts ----
    val_both = with_title_only_copies(val_df)
    val_proba = predict_survey_proba(val_both["Text"].tolist(), tokenizer, model)
    val_is_survey = (val_both["Label"].values == SURVEY_LABEL).astype(int)
    threshold = best_f1_threshold(val_is_survey, val_proba)

    combiner = LearnedHybrid().fit(val_both, val_proba, val_is_survey, seed=seed)
    combiner.save(output_dir)

    save_config(output_dir, {
        "threshold": threshold,
        "max_length": MAX_LENGTH,
        "base_model": BASE_MODEL,
        "dataset": os.path.abspath(dataset_path),
        "labels": {"0": "survey", "1": "non-survey"},
    })
    df.drop(columns="Text").to_csv(os.path.join(output_dir, SPLIT_FILE), index=False)

    # ---- Quick test-split report (full comparison: python main.py evaluate) ----
    test_proba = predict_survey_proba(test_df["Text"].tolist(), tokenizer, model)
    test_is_survey = (test_df["Label"].values == SURVEY_LABEL).astype(int)
    m = survey_metrics(test_is_survey, (test_proba >= threshold).astype(int))
    h = survey_metrics(test_is_survey, combiner.predict(test_df, test_proba))
    title_only = with_title_only_copies(test_df).iloc[len(test_df):]
    t_proba = predict_survey_proba(title_only["Text"].tolist(), tokenizer, model)
    t = survey_metrics(test_is_survey, (t_proba >= threshold).astype(int))

    print(f"\n💾 Model, threshold ({threshold:.2f}) and hybrid combiner saved to {output_dir}")
    print(f"🔧 Hybrid coefficients: {combiner.coefficients()}")
    print("\n📊 Held-out test split (survey = positive class):")
    for name, r in (("DistilBERT", m), ("Learned hybrid", h), ("DistilBERT, title only", t)):
        print(f"   {name:22s} acc {r['accuracy']:.3f}  prec {r['precision']:.3f}  "
              f"rec {r['recall']:.3f}  F1 {r['f1']:.3f}")
    print("✅ Training complete.")


# -------------------- Re-select thresholds without retraining -------------------- #
def retune_thresholds(model_dir="./distilbert_survey_model", seed=42):
    """Re-run threshold selection and the combiner fit on the saved validation split (model weights unchanged)."""
    from inference import load_model, load_config
    split = pd.read_csv(os.path.join(model_dir, SPLIT_FILE))
    val_df = split[split["Split"] == "val"].copy()
    val_df["Abstract"] = val_df["Abstract"].fillna("")
    val_df["Text"] = [paper_text(t, a) for t, a in zip(val_df["Title"], val_df["Abstract"])]
    tokenizer, model = load_model(model_dir)

    val_both = with_title_only_copies(val_df)
    val_proba = predict_survey_proba(val_both["Text"].tolist(), tokenizer, model)
    val_is_survey = (val_both["Label"].values == SURVEY_LABEL).astype(int)

    config = load_config(model_dir)
    old_threshold, old_combiner = config["threshold"], LearnedHybrid.load(model_dir)
    config["threshold"] = best_f1_threshold(val_is_survey, val_proba)
    combiner = LearnedHybrid().fit(val_both, val_proba, val_is_survey, seed=seed)

    print(f"DistilBERT threshold: {old_threshold:.2f} -> {config['threshold']:.2f}")
    print(f"Hybrid threshold:     {old_combiner.threshold:.2f} -> {combiner.threshold:.2f}")
    print(f"Hybrid coefficients:  {old_combiner.coefficients()} -> {combiner.coefficients()}")
    save_config(model_dir, config)
    combiner.save(model_dir)


# -------------------- Hyperparameter tuning -------------------- #
def run_tuning(dataset_path, n_trials=5, seed=42):
    """Optuna search on the train/validation splits only; the test split is untouched."""
    import optuna

    set_seed(seed)
    df = split_dataset(load_dataset(dataset_path), seed=seed)
    train_df, val_df = df[df["Split"] == "train"], df[df["Split"] == "val"]
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
        epochs = trial.suggest_int("num_train_epochs", 2, 4)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])
        dropout = trial.suggest_float("dropout", 0.1, 0.3)

        trainer = make_trainer(build_model(dropout), tokenizer, title_only_augment(train_df, seed=seed), val_df,
                               output_dir=os.path.join("results", "tuning"), epochs=epochs,
                               batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                               seed=seed, save_best=False)
        trainer.train()
        return trainer.evaluate()["eval_f1"]

    print(f"🚀 Starting Optuna tuning with {n_trials} trials (validation F1, survey = positive)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n🏆 Tuning Complete!")
    print(f"Best validation F1: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  • {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistilBERT survey classifier training & tuning")
    parser.add_argument("--dataset", type=str, default=os.path.join("data", "real_dataset.csv"))
    parser.add_argument("--output_dir", type=str, default="./distilbert_survey_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--retune-thresholds", action="store_true",
                        help="Re-select the thresholds of a trained model on its validation split")
    args = parser.parse_args()

    if args.retune_thresholds:
        retune_thresholds(args.output_dir)
    elif args.tune:
        run_tuning(args.dataset, n_trials=args.trials)
    else:
        run_training(args.dataset, args.output_dir, args.epochs, args.batch_size, args.lr)
