import argparse
import numpy as np
import os
import torch

from config import CONFIG, set_seed, MODEL_NAMES
from data.dataset import load_imdb_data, split_train_val, create_torch_dataloaders
from models.ml_models import train_ml_model
from models.lstm_model import build_lstm, build_bilstm, PyTorchTrainer
from models.distilbert_model import DistilBERTModel
from evaluate import evaluate_model
from utils.helpers import create_results_table, save_results_table, RESULTS_DIR

set_seed()


def train_lstm_family(vocab_size, train_loader, val_loader, test_texts, test_labels, model_type='lstm'):
    if model_type == 'lstm':
        model, display_name = build_lstm(vocab_size)
    else:
        model, display_name = build_bilstm(vocab_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = PyTorchTrainer(model, device)
    print(f"\nTraining {display_name}...")
    history = trainer.train(train_loader, val_loader)

    test_loader = val_loader
    _, y_pred, y_true, y_prob = trainer.evaluate(test_loader)
    metrics = evaluate_model(y_true, y_pred, y_prob, model_name=display_name)
    return metrics


def train_distilbert(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels):
    print("\nTraining DistilBERT...")
    model = DistilBERTModel()
    model.train(train_texts, train_labels, val_texts, val_labels)

    print("Evaluating DistilBERT on test set...")
    sample_size = min(len(test_texts), 5000)
    indices = np.random.choice(len(test_texts), sample_size, replace=False)
    test_texts_sample = [test_texts[i] for i in indices]
    test_labels_sample = [test_labels[i] for i in indices]

    y_pred, y_prob = model.predict(test_texts_sample)
    metrics = evaluate_model(test_labels_sample, y_pred, y_prob, model_name='DistilBERT')
    return metrics


def train_all(train_ml=True, train_deep=True, train_distilbert_model=True):
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(preprocess=True)
    train_texts_sub, val_texts, train_labels_sub, val_labels = split_train_val(train_texts, train_labels)

    sample_size = 25000
    train_idx = np.random.choice(len(train_texts_sub), min(sample_size, len(train_texts_sub)), replace=False)
    train_texts_sample = [train_texts_sub[i] for i in train_idx]
    train_labels_sample = [train_labels_sub[i] for i in train_idx]

    test_idx = np.random.choice(len(test_texts), 5000, replace=False)
    test_texts_sample = [test_texts[i] for i in test_idx]
    test_labels_sample = [test_labels[i] for i in test_idx]

    all_metrics = {}

    if train_ml:
        print("=" * 60)
        print("TRAINING ML MODELS")
        print("=" * 60)
        for model_name in ['logistic_regression', 'naive_bayes', 'svm']:
            print(f"\nTraining {model_name.replace('_', ' ').title()}...")
            model, display_name = train_ml_model(
                model_name, train_texts_sample, train_labels_sample,
                use_gridsearch=True
            )
            y_pred = model.predict(test_texts_sample)
            y_prob = model.predict_proba(test_texts_sample)[:, 1] if hasattr(model, 'predict_proba') else None
            if y_prob is None:
                y_prob = model.decision_function(test_texts_sample)
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)
            metrics = evaluate_model(test_labels_sample, y_pred, y_prob, model_name=display_name)
            all_metrics[display_name] = metrics

    if train_deep:
        print("\n" + "=" * 60)
        print("TRAINING DEEP LEARNING MODELS (LSTM/BiLSTM)")
        print("=" * 60)
        train_loader, val_loader, tokenizer = create_torch_dataloaders(
            train_texts_sub, train_labels_sub,
            val_texts, val_labels
        )
        vocab_size = tokenizer.vocab_size
        for model_type in ['lstm', 'bilstm']:
            metrics = train_lstm_family(
                vocab_size, train_loader, val_loader,
                test_texts_sample, test_labels_sample,
                model_type=model_type
            )
            all_metrics.update(metrics)

    if train_distilbert_model:
        print("\n" + "=" * 60)
        print("TRAINING DISTILBERT")
        print("=" * 60)
        try:
            metrics = train_distilbert(
                train_texts_sample, train_labels_sample,
                val_texts, val_labels,
                test_texts_sample, test_labels_sample
            )
            all_metrics.update(metrics)
        except Exception as e:
            print(f"  DistilBERT training skipped (error: {e})")

    results_df = create_results_table(all_metrics)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(results_df.to_string())
    save_results_table(results_df, 'model_comparison.csv')
    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    parser.add_argument('--models', nargs='+', default=MODEL_NAMES,
                        help=f'Models to train: {MODEL_NAMES}')
    parser.add_argument('--no-ml', action='store_true', help='Skip ML models')
    parser.add_argument('--no-deep', action='store_true', help='Skip LSTM/BiLSTM')
    parser.add_argument('--no-distilbert', action='store_true', help='Skip DistilBERT')
    args = parser.parse_args()

    train_all(
        train_ml=not args.no_ml,
        train_deep=not args.no_deep,
        train_distilbert_model=not args.no_distilbert
    )
