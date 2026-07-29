import random
import numpy as np
import torch
import os

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

CONFIG = {
    'seed': SEED,
    'embedding_dim': 100,
    'hidden_dim': 256,
    'output_dim': 1,
    'num_layers': 2,
    'dropout': 0.3,
    'batch_size': 64,
    'num_epochs': 5,
    'learning_rate': 1e-3,
    'max_len': 512,
    'test_size': 0.2,
    'val_size': 0.1,
    'max_features': 5000,
    'n_gram_range': (1, 2),
}

MODEL_NAMES = ['logistic_regression', 'naive_bayes', 'svm', 'lstm', 'bilstm', 'distilbert']
