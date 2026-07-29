import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from config import CONFIG
from utils.preprocessing import clean_text


class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def load_imdb_data(preprocess=True):
    print("Loading IMDB dataset...")
    dataset = load_dataset('imdb')
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    if preprocess:
        print("Preprocessing texts...")
        train_texts = [clean_text(t) for t in train_texts]
        test_texts = [clean_text(t) for t in test_texts]

    return train_texts, train_labels, test_texts, test_labels


def get_tokenizer():
    return DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


def create_torch_dataloaders(train_texts, train_labels, test_texts, test_labels):
    tokenizer = get_tokenizer()

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True,
        max_length=CONFIG['max_len'], return_tensors='pt'
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True,
        max_length=CONFIG['max_len'], return_tensors='pt'
    )

    train_dataset = SentimentDataset(
        {k: v.numpy() for k, v in train_encodings.items()},
        train_labels
    )
    test_dataset = SentimentDataset(
        {k: v.numpy() for k, v in test_encodings.items()},
        test_labels
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG['batch_size']
    )

    return train_loader, test_loader, tokenizer


def split_train_val(train_texts, train_labels):
    return train_test_split(
        train_texts, train_labels,
        test_size=CONFIG['val_size'],
        random_state=CONFIG['seed'],
        stratify=train_labels
    )
