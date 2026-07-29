# API Reference

## Start the server

```bash
python app.py
```

Runs on `http://localhost:5000` by default.

## Endpoints

### `GET /health`

Health check.

**Response:**
```json
{
  "status": "ok"
}
```

### `POST /predict`

Predict sentiment for a text.

**Request:**
```json
{
  "text": "This movie was absolutely fantastic!"
}
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.9834,
  "text": "This movie was absolutely fantastic!"
}
```

### Example usage

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible acting and boring plot."}'
```

```json
{
  "sentiment": "Negative",
  "confidence": 0.0023,
  "text": "Terrible acting and boring plot."
}
```
