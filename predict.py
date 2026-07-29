import argparse
import pickle
import torch
import numpy as np
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


def predict_single(text, model, model_type):
    cleaned = clean_text(text)
    if model_type in ['logistic_regression', 'naive_bayes', 'svm']:
        pred = model.predict([cleaned])[0]
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba([cleaned])[0][1]
        else:
            prob = None
        return pred, prob, cleaned
    elif model_type in ['lstm', 'bilstm']:
        model_obj, tokenizer = model
        encodings = tokenizer([cleaned], truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            output = model_obj(encodings['input_ids'], encodings['attention_mask']).squeeze(1)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob >= 0.5 else 0
        return pred, prob, cleaned
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict sentiment for a text')
    parser.add_argument('--text', type=str, required=True, help='Text to analyze')
    parser.add_argument('--model-type', type=str, default='logistic_regression',
                        choices=['logistic_regression', 'naive_bayes', 'svm', 'lstm', 'bilstm'])
    parser.add_argument('--model-path', type=str, default='models/saved/best_model.pkl')
    args = parser.parse_args()

    model = load_model(args.model_type, args.model_path)
    pred, prob, cleaned = predict_single(args.text, model, args.model_type)

    sentiment = 'Positive' if pred == 1 else 'Negative'
    print(f"Text: {args.text[:100]}...")
    print(f"Cleaned: {cleaned[:100]}...")
    print(f"Sentiment: {sentiment} (confidence: {prob:.4f})")
