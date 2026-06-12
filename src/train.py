import os
import random
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
import optuna

warnings.simplefilter("ignore", category=FutureWarning)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Dataset Class -------------------- #
class SurveyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# -------------------- Metric Computation -------------------- #
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        pred.label_ids, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(pred.label_ids, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def run_training(dataset_path, output_dir, epochs=3, batch_size=16, lr=2e-5):
    """
    Standard training logic from Train.py
    """
    print(f"📊 Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)[['Title', 'Abstract', 'Label']].dropna()
    df = df[~df[['Title', 'Abstract']].apply(lambda x: x.str.strip().eq('').any(), axis=1)]
    df['Text'] = df['Title'] + ' ' + df['Abstract']
    print("Label distribution:\n", df['Label'].value_counts())

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Text'].tolist(),
        df['Label'].tolist(),
        test_size=0.2,
        stratify=df['Label'],
        random_state=42
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    train_dataset = SurveyDataset(train_encodings, train_labels)
    val_dataset = SurveyDataset(val_encodings, val_labels)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    config = DistilBertConfig.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        dropout=0.2,
        attention_dropout=0.2
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", config=config
    ).to(device)

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(config.dim, 2)
    )
    model.loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=lr,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("🚀 Starting training...")
    trainer.train()

    print(f"💾 Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Training complete.")


def run_tuning(dataset_path, n_trials=5):
    """
    Hyperparameter tuning logic from distil.py using Optuna
    """
    print(f"📊 Loading dataset for tuning: {dataset_path}")
    df = pd.read_csv(dataset_path)[['Title', 'Abstract', 'Label']].dropna()
    df = df[df['Label'].isin([0, 1])]
    df['Text'] = df['Title'].astype(str) + " " + df['Abstract'].astype(str)
    texts, labels = df['Text'].tolist(), df['Label'].tolist()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def model_init():
        return DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    def objective(trial):
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        f1_scores = []

        for train_index, val_index in kf.split(texts, labels):
            train_texts = [texts[i] for i in train_index]
            train_labels = [labels[i] for i in train_index]
            val_texts = [texts[i] for i in val_index]
            val_labels = [labels[i] for i in val_index]

            train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
            val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

            train_dataset = SurveyDataset(train_encodings, train_labels)
            val_dataset = SurveyDataset(val_encodings, val_labels)

            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=trial.suggest_int("num_train_epochs", 3, 5),
                per_device_train_batch_size=trial.suggest_categorical("batch_size", [8, 16]),
                per_device_eval_batch_size=32,
                learning_rate=trial.suggest_float("learning_rate", 2e-5, 5e-5, log=True),
                weight_decay=trial.suggest_float("weight_decay", 0.0, 0.2),
                evaluation_strategy="epoch",
                save_strategy="no",
                logging_dir="./logs",
                seed=42,
                report_to="none"
            )

            trainer = Trainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics,
            )

            trainer.train()
            eval_metrics = trainer.evaluate()
            f1_scores.append(eval_metrics.get("eval_f1", 0.0))

        return np.mean(f1_scores)

    print(f"🚀 Starting Optuna tuning with {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n🏆 Tuning Complete!")
    print(f"Best Trial F1 Score: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  • {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistilBERT Classifier Training & Tuning")
    parser.add_argument("--dataset", type=str, default=os.path.join("data", "dataset.csv"), help="Path to input dataset CSV")
    parser.add_argument("--output_dir", type=str, default="./distilbert_survey_model", help="Path to save model weights")
    parser.add_argument("--epochs", type=str, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=str, default=16, help="Batch size")
    parser.add_argument("--lr", type=str, default=2e-5, help="Learning rate")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning instead of training")
    parser.add_argument("--trials", type=int, default=5, help="Number of tuning trials for Optuna")

    args = parser.parse_args()

    if args.tune:
        run_tuning(args.dataset, n_trials=args.trials)
    else:
        run_training(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr)
        )
