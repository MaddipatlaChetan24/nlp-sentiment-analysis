from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pickle
import os

from config import CONFIG
from utils.preprocessing import clean_text

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'saved', 'best_model.pkl')
MODEL_TYPE = 'logistic_regression'
_model = None


def load_model():
    global _model
    if _model is None:
        if MODEL_TYPE in ['logistic_regression', 'naive_bayes', 'svm']:
            with open(MODEL_PATH, 'rb') as f:
                _model = pickle.load(f)
        else:
            from models.lstm_model import build_lstm, build_bilstm
            from data.dataset import get_tokenizer
            tokenizer = get_tokenizer()
            if MODEL_TYPE == 'lstm':
                _model, _ = build_lstm(tokenizer.vocab_size)
            else:
                _model, _ = build_bilstm(tokenizer.vocab_size)
            _model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            _model.eval()
            _model._tokenizer = tokenizer
    return _model


@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    cleaned = clean_text(text)
    model = load_model()

    if MODEL_TYPE in ['logistic_regression', 'naive_bayes', 'svm']:
        pred = model.predict([cleaned])[0]
        prob = model.predict_proba([cleaned])[0][1] if hasattr(model, 'predict_proba') else None
    else:
        tokenizer = model._tokenizer
        encodings = tokenizer([cleaned], truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            output = model(encodings['input_ids'], encodings['attention_mask']).squeeze(1)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob >= 0.5 else 0

    sentiment = 'Positive' if pred == 1 else 'Negative'

    return jsonify({
        'sentiment': sentiment,
        'confidence': round(prob, 4) if prob is not None else None,
        'text': text,
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)
