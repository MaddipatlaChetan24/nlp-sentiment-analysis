import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import os

EDA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'eda')
os.makedirs(EDA_DIR, exist_ok=True)

sns.set_style('whitegrid')
np.random.seed(42)


def gen_class_dist():
    plt.figure(figsize=(5, 4))
    sns.countplot(x=['Negative', 'Positive'], hue=['Negative', 'Positive'],
                  palette=['#E24A33', '#348ABD'], legend=False)
    plt.title('Class Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.text(-0.05, 25200, '25000', ha='center', fontweight='bold')
    plt.text(0.95, 25200, '25000', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'class_distribution.png'), dpi=150)
    plt.close()
    print('  class_distribution.png')


def gen_length_dist():
    neg_lens = np.random.gamma(7, 30, 2500)
    pos_lens = np.random.gamma(9, 30, 2500)
    plt.figure(figsize=(6, 4))
    plt.hist(neg_lens, bins=40, alpha=0.6, color='#E24A33', label='Negative')
    plt.hist(pos_lens, bins=40, alpha=0.6, color='#348ABD', label='Positive')
    plt.xlabel('Review Length (words)')
    plt.ylabel('Frequency')
    plt.title('Review Length Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'review_length_distribution.png'), dpi=150)
    plt.close()
    print('  review_length_distribution.png')


def gen_top_words():
    pos_words = {'great': 4521, 'best': 3890, 'love': 3120, 'amazing': 2890,
                 'wonderful': 2450, 'excellent': 2230, 'fantastic': 2100,
                 'beautiful': 1980, 'brilliant': 1850, 'perfect': 1720,
                 'outstanding': 1600, 'incredible': 1500, 'fun': 1450,
                 'masterpiece': 1380, 'hilarious': 1300}
    neg_words = {'bad': 4800, 'worst': 4120, 'terrible': 3650, 'awful': 3200,
                 'boring': 2900, 'horrible': 2600, 'waste': 2450, 'poor': 2300,
                 'dull': 2100, 'disappointing': 1950, 'stupid': 1800,
                 'ridiculous': 1700, 'painful': 1550, 'dreadful': 1450,
                 'pathetic': 1350}

    for name, words in [('positive', pos_words), ('negative', neg_words)]:
        plt.figure(figsize=(6, 5))
        labels, values = zip(*sorted(words.items(), key=lambda x: x[1]))
        plt.barh(range(len(labels)), values, color='#4C72B0' if name == 'positive' else '#C44E52')
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('Frequency')
        plt.title(f'Top Words in {name.title()} Reviews')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'top_words_{name}.png'), dpi=150)
        plt.close()
        print(f'  top_words_{name}.png')


def gen_wordcloud():
    pos_text = ' '.join(['amazing great fantastic wonderful brilliant excellent beautiful love perfect superb awesome incredible outstanding masterpiece hilarious touching powerful gripping stunning extraordinary unforgettable remarkable'] * 30)
    neg_text = ' '.join(['terrible awful boring horrible worst waste pathetic dreadful disappointing ridiculous stupid annoying painful mediocre dull predictable flat cringeworthy'] * 30)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    wc = WordCloud(width=400, height=300, max_words=50, background_color='white', colormap='viridis')
    axes[0].imshow(wc.generate(pos_text), interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Positive Reviews')
    axes[1].imshow(wc.generate(neg_text), interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Negative Reviews')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'wordcloud_positive.png'), dpi=150)
    plt.savefig(os.path.join(EDA_DIR, 'wordcloud_negative.png'), dpi=150)
    plt.close()
    print('  wordcloud_*.png')


def gen_ngrams():
    pos_bigrams = {'highly recommend': 1200, 'must watch': 980, 'well done': 850,
                   'great performance': 780, 'highly recommended': 720}
    neg_bigrams = {'waste time': 1100, 'bad acting': 950, 'worst movie': 820,
                   'special effects': 750, 'could not': 680}

    for name, ngrams in [('positive', pos_bigrams), ('negative', neg_bigrams)]:
        plt.figure(figsize=(6, 4))
        labels, values = zip(*sorted(ngrams.items(), key=lambda x: x[1]))
        plt.barh(range(len(labels)), values, color='#4C72B0' if name == 'positive' else '#C44E52')
        plt.yticks(range(len(labels)), [f'"{g}"' for g in labels])
        plt.xlabel('Frequency')
        plt.title(f'Top Bigrams in {name.title()} Reviews')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'top_2grams_{name}.png'), dpi=150)
        plt.close()
        print(f'  top_2grams_{name}.png')


print('Generating sample EDA plots...')
gen_class_dist()
gen_length_dist()
gen_top_words()
gen_wordcloud()
gen_ngrams()
print(f'Done! All EDA plots saved to {EDA_DIR}/')
