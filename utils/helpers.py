import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
    }
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['roc_auc'] = None
    return metrics


def print_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(os.path.join(RESULTS_DIR, save_path), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(y_true, y_prob, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    if save_path:
        plt.savefig(os.path.join(RESULTS_DIR, save_path), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_results_table(results_dict):
    df = pd.DataFrame(results_dict).T
    df = df.round(4)
    return df


def save_results_table(df, filename='model_comparison.csv'):
    df.to_csv(os.path.join(RESULTS_DIR, filename))
    print(f"Results saved to {os.path.join(RESULTS_DIR, filename)}")


def get_misclassified(y_true, y_pred, texts):
    fp_indices = []
    fn_indices = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true == 0 and pred == 1:
            fp_indices.append(i)
        elif true == 1 and pred == 0:
            fn_indices.append(i)

    results = {
        'false_positives': [{'text': texts[i], 'true_label': 0, 'pred_label': 1} for i in fp_indices],
        'false_negatives': [{'text': texts[i], 'true_label': 1, 'pred_label': 0} for i in fn_indices],
        'count_fp': len(fp_indices),
        'count_fn': len(fn_indices),
    }
    return results
