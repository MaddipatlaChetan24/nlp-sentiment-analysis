import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

from data.dataset import load_imdb_data
from utils.preprocessing import clean_text, remove_stopwords, lemmatize
from evaluate import evaluate_model, analyze_errors
from models.ml_models import train_ml_model
from config import set_seed

set_seed()

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'error_analysis')
os.makedirs(RESULTS_DIR, exist_ok=True)


def analyze_misclassification_patterns(texts, y_true, y_pred, model_name='Model'):
    errors = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            errors.append({
                'text': texts[i],
                'true_label': 'Positive' if true == 1 else 'Negative',
                'pred_label': 'Positive' if pred == 1 else 'Negative',
                'text_length': len(texts[i].split()),
            })

    df_errors = pd.DataFrame(errors)
    print(f"\n  Misclassification Analysis - {model_name}")
    print(f"  Total misclassified: {len(df_errors)}")

    if len(df_errors) > 0:
        print(f"\n  Error examples:")
        for i in range(min(10, len(df_errors))):
            err = errors[i]
            print(f"\n  Example {i+1}:")
            print(f"    True: {err['true_label']} | Predicted: {err['pred_label']}")
            print(f"    Text: {err['text'][:200]}...")


def analyze_error_causes(texts, y_true, y_pred):
    fp_indices = [i for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1]
    fn_indices = [i for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0]

    print("\n" + "=" * 60)
    print("ROOT CAUSE ANALYSIS OF ERRORS")
    print("=" * 60)

    print("\nCommon failure patterns:")
    print("  1. Ambiguous language - reviews with mixed sentiment")
    print("  2. Sarcasm / irony - not captured by bag-of-words models")
    print("  3. Negation handling - 'not good' vs 'not bad'")
    print("  4. Short reviews - insufficient context")
    print("  5. Domain-specific jargon not in training vocabulary")

    if len(fp_indices) > 0:
        fp_text = ' '.join([texts[i] for i in fp_indices[:100]])
        fp_words = [w for w in fp_text.lower().split() if len(w) > 3]
        from collections import Counter
        common_fp = Counter(fp_words).most_common(10)
        print(f"\n  Top words in False Positives: {common_fp}")

    if len(fn_indices) > 0:
        fn_text = ' '.join([texts[i] for i in fn_indices[:100]])
        fn_words = [w for w in fn_text.lower().split() if len(w) > 3]
        from collections import Counter
        common_fn = Counter(fn_words).most_common(10)
        print(f"\n  Top words in False Negatives: {common_fn}")


def run_error_analysis():
    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    print("\nLoading data...")
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(preprocess=True)

    test_sample_size = min(len(test_texts), 5000)
    np.random.seed(42)
    indices = np.random.choice(len(test_texts), test_sample_size, replace=False)
    test_texts_sample = [test_texts[i] for i in indices]
    test_labels_sample = [test_labels[i] for i in indices]

    train_sample_size = min(len(train_texts), 10000)
    train_indices = np.random.choice(len(train_texts), train_sample_size, replace=False)
    train_texts_sample = [train_texts[i] for i in train_indices]
    train_labels_sample = [train_labels[i] for i in train_indices]

    print(f"\nTraining Logistic Regression for error analysis...")
    model, display_name = train_ml_model(
        'logistic_regression', train_texts_sample, train_labels_sample,
        use_gridsearch=True
    )

    y_pred = model.predict(test_texts_sample)
    y_prob = model.predict_proba(test_texts_sample)[:, 1]

    evaluate_model(test_labels_sample, y_pred, y_prob, model_name=display_name)
    analyze_errors(test_texts_sample, test_labels_sample, y_pred, model_name=display_name)
    analyze_misclassification_patterns(
        test_texts_sample, test_labels_sample, y_pred, model_name=display_name
    )
    analyze_error_causes(test_texts_sample, test_labels_sample, y_pred)

    print(f"\nError analysis complete! See results/error_analysis/ for details.")


if __name__ == '__main__':
    run_error_analysis()
