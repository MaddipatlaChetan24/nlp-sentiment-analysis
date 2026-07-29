import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from utils.helpers import (
    plot_confusion_matrix, plot_roc_curve,
    get_misclassified, print_classification_report,
    RESULTS_DIR
)


def evaluate_model(y_true, y_pred, y_prob=None, model_name='Model'):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc_auc = None

    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"\n  Classification Report:")
    print(print_classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"    FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")

    safe_name = model_name.lower().replace(' ', '_')
    plot_confusion_matrix(y_true, y_pred, save_path=f'cm_{safe_name}.png')
    if y_prob is not None:
        plot_roc_curve(y_true, y_prob, save_path=f'roc_{safe_name}.png')

    metrics = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
    }
    if roc_auc is not None:
        metrics['roc_auc'] = round(roc_auc, 4)
    else:
        metrics['roc_auc'] = 'N/A'

    return {model_name: metrics}


def analyze_errors(texts, y_true, y_pred, model_name='Model'):
    result = get_misclassified(y_true, y_pred, texts)
    print(f"\n{'='*40}")
    print(f"  Error Analysis - {model_name}")
    print(f"{'='*40}")
    print(f"  False Positives: {result['count_fp']}")
    print(f"  False Negatives: {result['count_fn']}")
    print(f"  Total Errors:    {result['count_fp'] + result['count_fn']}")

    print(f"\n  --- False Positives (predicted Positive, actually Negative) ---")
    for i, fp in enumerate(result['false_positives'][:5]):
        print(f"  {i+1}. {fp['text'][:150]}...")

    print(f"\n  --- False Negatives (predicted Negative, actually Positive) ---")
    for i, fn in enumerate(result['false_negatives'][:5]):
        print(f"  {i+1}. {fn['text'][:150]}...")

    return result
