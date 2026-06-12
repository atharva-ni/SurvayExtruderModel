import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Survey-related keywords
survey_keywords = [
    "survey", "review", "overview", "comparative",
    "taxonomy", "state of the art", "systematic"
]

def keyword_is_survey(text):
    text = text.lower()
    return any(keyword in text for keyword in survey_keywords)

def load_data(df):
    texts = (df['title'].astype(str) + " " + df['abstract'].astype(str)).tolist()
    return [t.lower() for t in texts]

def classify_with_hybrid_model(texts, model_path, threshold=0.8):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokens = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        survey_probs = probs[:, 0].cpu().numpy()  # probability of class 0 (survey)

    model_preds = [0 if prob > threshold else 1 for prob in survey_probs]
    keyword_preds = [0 if keyword_is_survey(t) else None for t in texts]

    final_preds = [
        kp if kp is not None else mp
        for kp, mp in zip(keyword_preds, model_preds)
    ]

    return final_preds, survey_probs

def calculate_and_print_metrics(df, label_col='label', pred_col='Prediction', prob_col='DistilBERT_Prob', model_name=''):
    y_true = df[label_col].astype(int).values
    y_pred = df[pred_col].values
    y_prob = df[prob_col].values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n📊 Evaluation: {model_name}")
    print(f"Accuracy : {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall   : {rec:.2f}")
    print(f"F1 Score : {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["survey", "not survey"]))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Survey", "Not Survey"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, 1 - y_prob)  # flip prob for "not survey"
    auc = roc_auc_score(y_true, 1 - y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Survey vs Not Survey)")
    plt.legend()
    plt.grid()
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, 1 - y_prob)
    avg_prec = average_precision_score(y_true, 1 - y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {avg_prec:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    input_csv = 'paper1.csv'
    distilbert_path = './distilbert_survey_model'

    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip().str.lower()
    required_cols = ['title', 'abstract', 'label']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")

    texts = load_data(df)

    # DistilBERT + Keyword
    distilbert_preds, probs = classify_with_hybrid_model(texts, distilbert_path)
    df['Prediction'] = distilbert_preds
    df['DistilBERT_Prob'] = probs  # store probability of survey class

    # Print Metrics & Plots
    calculate_and_print_metrics(df, pred_col='Prediction', prob_col='DistilBERT_Prob', model_name='DistilBERT + Keyword')
