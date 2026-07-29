# Architecture

## Training Pipeline

```mermaid
flowchart TD
    A["IMDB Dataset<br/>(50k reviews)"] --> B["Text Preprocessing"]
    B --> C["Train / Val / Test Split"]
    
    C --> D["ML Models"]
    C --> E["Deep Learning"]
    C --> F["Transformers"]
    
    D --> G["TF-IDF Vectorization"]
    G --> H["GridSearchCV"]
    H --> I["Best Estimator"]
    
    E --> J["DistilBERT Tokenizer"]
    J --> K["PyTorch DataLoader"]
    K --> L["LSTM / BiLSTM Training"]
    
    F --> M["DistilBERT Tokenizer"]
    M --> N["HuggingFace Trainer"]
    N --> O["Fine-tuned DistilBERT"]
    
    I & L & O --> P["Evaluation"]
    P --> Q["Results Table + Plots"]
```

## Text Preprocessing

Each review passes through 9 stages before reaching any model:

```mermaid
flowchart LR
    A["Raw Text"] --> B["Lowercase"]
    B --> C["Remove URLs"]
    C --> D["Strip HTML Tags"]
    D --> E["Handle Emojis"]
    E --> F["Expand Contractions"]
    F --> G["Normalize Repeated Chars"]
    G --> H["Remove Stopwords"]
    H --> I["Lemmatization"]
    I --> J["Clean Text"]
```

## Model Architectures

### Logistic Regression
- TF-IDF vectorization (unigrams + bigrams, max 5000 features)
- L2 regularization with C tuned via GridSearch

### Naive Bayes (Multinomial)
- TF-IDF vectorization
- Additive smoothing (alpha) tuned via GridSearch

### SVM (Linear)
- TF-IDF vectorization
- Hinge / squared hinge loss with C tuned via GridSearch

### LSTM
- Embedding layer (100 dim) → 2-layer LSTM (256 hidden) → Dropout (0.3) → FC (1)
- BCEWithLogitsLoss, Adam optimizer

### BiLSTM
- Same as LSTM but bidirectional (concatenates forward + backward hidden states)

### DistilBERT
- `distilbert-base-uncased` with classification head
- Fine-tuned for 3 epochs with HuggingFace Trainer
- Early stopping based on validation F1
