# Experiments

## Hyperparameter Grids

### Logistic Regression

| Parameter | Values Tested | Best |
|-----------|---------------|------|
| `C` | 0.1, 1.0, 10.0 | 1.0 |
| `max_features` | 3000, 5000, 10000 | 5000 |
| `solver` | liblinear, lbfgs | liblinear |

### Naive Bayes

| Parameter | Values Tested | Best |
|-----------|---------------|------|
| `alpha` | 0.01, 0.1, 0.5, 1.0 | 0.1 |
| `max_features` | 3000, 5000, 10000 | 5000 |
| `ngram_range` | (1,1), (1,2) | (1,2) |

### SVM (Linear)

| Parameter | Values Tested | Best |
|-----------|---------------|------|
| `C` | 0.01, 0.1, 1.0, 10.0 | 1.0 |
| `max_features` | 3000, 5000, 10000 | 5000 |
| `loss` | squared_hinge, hinge | squared_hinge |

### LSTM / BiLSTM

| Parameter | Value |
|-----------|-------|
| Embedding dim | 100 |
| Hidden dim | 256 |
| Layers | 2 |
| Dropout | 0.3 |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 5 |

### DistilBERT

| Parameter | Value |
|-----------|-------|
| Model | distilbert-base-uncased |
| Epochs | 3 |
| Batch size | 16 |
| Optimizer | Adam (default) |
| Eval strategy | per epoch |
| Metric for best model | F1 |
