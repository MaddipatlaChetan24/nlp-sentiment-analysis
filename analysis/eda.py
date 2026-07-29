import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import os

from data.dataset import load_imdb_data
from utils.preprocessing import clean_text

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'eda')
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_class_distribution(labels):
    plt.figure()
    sns.countplot(x=labels)
    plt.title('Class Distribution in IMDB Dataset')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    for i, count in enumerate(Counter(labels).values()):
        plt.text(i, count + 100, str(count), ha='center')
    plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution.png'), bbox_inches='tight')
    plt.close()
    print(f"  Saved class_distribution.png")
    print(f"  Negative: {Counter(labels)[0]}, Positive: {Counter(labels)[1]}")


def plot_review_length_distribution(texts, labels, max_len=500):
    lengths = [len(t.split()) for t in texts]
    plt.figure()
    plt.hist([l for l, lbl in zip(lengths, labels) if lbl == 0],
             bins=50, alpha=0.6, label='Negative', range=(0, max_len))
    plt.hist([l for l, lbl in zip(lengths, labels) if lbl == 1],
             bins=50, alpha=0.6, label='Positive', range=(0, max_len))
    plt.xlabel('Review Length (words)')
    plt.ylabel('Frequency')
    plt.title('Review Length Distribution by Sentiment')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'review_length_distribution.png'), bbox_inches='tight')
    plt.close()
    print(f"  Saved review_length_distribution.png")
    print(f"  Avg length: {np.mean(lengths):.0f} words, Median: {np.median(lengths):.0f} words")


def plot_most_frequent_words(texts, labels, top_n=20):
    pos_text = ' '.join([t for t, lbl in zip(texts, labels) if lbl == 1])
    neg_text = ' '.join([t for t, lbl in zip(texts, labels) if lbl == 0])

    for sentiment, text, label in [('Positive', pos_text, 'positive'),
                                    ('Negative', neg_text, 'negative')]:
        words = text.split()
        word_freq = Counter(words).most_common(top_n)
        words_list, counts = zip(*word_freq)

        plt.figure()
        plt.barh(range(top_n), counts[::-1])
        plt.yticks(range(top_n), words_list[::-1])
        plt.xlabel('Frequency')
        plt.title(f'Top {top_n} Most Frequent Words in {sentiment} Reviews')
        plt.savefig(os.path.join(RESULTS_DIR, f'top_words_{label}.png'), bbox_inches='tight')
        plt.close()
        print(f"  Saved top_words_{label}.png")


def plot_wordcloud(texts, labels):
    pos_text = ' '.join([t for t, lbl in zip(texts, labels) if lbl == 1])
    neg_text = ' '.join([t for t, lbl in zip(texts, labels) if lbl == 0])

    for sentiment, text, label in [('Positive', pos_text, 'positive'),
                                    ('Negative', neg_text, 'negative')]:
        wc = WordCloud(width=800, height=400, max_words=100,
                       background_color='white', colormap='viridis').generate(text)
        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {sentiment} Reviews')
        plt.savefig(os.path.join(RESULTS_DIR, f'wordcloud_{label}.png'), bbox_inches='tight')
        plt.close()
        print(f"  Saved wordcloud_{label}.png")


def plot_ngram_analysis(texts, labels, n=2, top_n=15):
    from sklearn.feature_extraction.text import CountVectorizer

    for sentiment, label in [('Positive', 1), ('Negative', 0)]:
        subset_texts = [t for t, lbl in zip(texts, labels) if lbl == label]
        vec = CountVectorizer(ngram_range=(n, n), max_features=top_n).fit(subset_texts)
        ngram_matrix = vec.transform(subset_texts)
        ngram_counts = np.array(ngram_matrix.sum(axis=0)).flatten()
        ngram_freq = sorted(zip(vec.get_feature_names_out(), ngram_counts),
                           key=lambda x: x[1], reverse=True)[:top_n]

        ngrams, counts = zip(*ngram_freq)
        plt.figure()
        plt.barh(range(top_n), counts[::-1])
        plt.yticks(range(top_n), [f'"{g}"' for g in ngrams[::-1]])
        plt.xlabel('Frequency')
        plt.title(f'Top {top_n} {n}-grams in {sentiment} Reviews')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'top_{n}grams_{label}.png'), bbox_inches='tight')
        plt.close()
        print(f"  Saved top_{n}grams_{label}.png")


def run_full_eda():
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\nLoading data...")
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(preprocess=False)

    print("\n[1/5] Class Distribution...")
    plot_class_distribution(train_labels)

    print("\n[2/5] Review Length Distribution...")
    plot_review_length_distribution(train_texts, train_labels)

    print("\n[3/5] Most Frequent Words (preprocessing applied)...")
    cleaned_texts = [clean_text(t) for t in train_texts]
    plot_most_frequent_words(cleaned_texts, train_labels)

    print("\n[4/5] Word Clouds...")
    plot_wordcloud(cleaned_texts, train_labels)

    print("\n[5/5] N-gram Analysis...")
    plot_ngram_analysis(cleaned_texts, train_labels, n=2)
    plot_ngram_analysis(cleaned_texts, train_labels, n=3)

    print(f"\nEDA complete! All plots saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    run_full_eda()
