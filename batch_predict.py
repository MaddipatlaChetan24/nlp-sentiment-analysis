import argparse
import pandas as pd
import pickle
import torch
from tqdm import tqdm
from config import CONFIG
from utils.preprocessing import clean_text


def load_model(model_type, model_path):
    if model_type in ['logistic_regression', 'naive_bayes', 'svm']:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    elif model_type in ['lstm', 'bilstm']:
        from models.lstm_model import build_lstm, build_bilstm
        from data.dataset import get_tokenizer
        tokenizer = get_tokenizer()
        if model_type == 'lstm':
            model, _ = build_lstm(tokenizer.vocab_size)
        else:
            model, _ = build_bilstm(tokenizer.vocab_size)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model, tokenizer
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def batch_predict(input_csv, output_csv, model, model_type, text_column='review'):
    print(f"Loading reviews from {input_csv}...")
    df = pd.read_csv(input_csv)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Columns: {list(df.columns)}")

    texts = df[text_column].tolist()
    predictions = []
    probabilities = []

    print(f"Processing {len(texts)} reviews...")
    for text in tqdm(texts):
        cleaned = clean_text(str(text))

        if model_type in ['logistic_regression', 'naive_bayes', 'svm']:
            pred = model.predict([cleaned])[0]
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba([cleaned])[0][1]
            else:
                prob = None
        elif model_type in ['lstm', 'bilstm']:
            model_obj, tokenizer = model
            encodings = tokenizer([cleaned], truncation=True, padding=True, return_tensors='pt')
            with torch.no_grad():
                output = model_obj(encodings['input_ids'], encodings['attention_mask']).squeeze(1)
                prob = torch.sigmoid(output).item()
                pred = 1 if prob >= 0.5 else 0
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        predictions.append('Positive' if pred == 1 else 'Negative')
        probabilities.append(round(prob, 4) if prob is not None else None)

    df['sentiment'] = predictions
    df['confidence'] = probabilities
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    pos_count = sum(1 for p in predictions if p == 'Positive')
    neg_count = len(predictions) - pos_count
    print(f"Summary: {pos_count} positive, {neg_count} negative")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch predict sentiment for CSV reviews')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file path')
    parser.add_argument('--text-column', type=str, default='review', help='Name of text column in CSV')
    parser.add_argument('--model-type', type=str, default='logistic_regression',
                        choices=['logistic_regression', 'naive_bayes', 'svm', 'lstm', 'bilstm'])
    parser.add_argument('--model-path', type=str, default='models/saved/best_model.pkl')
    args = parser.parse_args()

    model = load_model(args.model_type, args.model_path)
    batch_predict(args.input, args.output, model, args.model_type, args.text_column)
