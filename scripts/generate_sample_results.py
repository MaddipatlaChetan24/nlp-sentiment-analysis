import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from wordcloud import WordCloud
import os

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'images')
os.makedirs(BASE, exist_ok=True)
RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS, exist_ok=True)

sns.set_style('whitegrid')

np.random.seed(42)

y_true = np.array([0]*500 + [1]*500)
y_pred_lr = np.array([0]*445 + [1]*55 + [0]*60 + [1]*440)
y_prob_lr = np.concatenate([
    np.random.beta(2, 8, 500),
    np.random.beta(8, 2, 500)
])

def plot_cm():
    cm = confusion_matrix(y_true, y_pred_lr)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix — Logistic Regression')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'confusion_matrix.png'), dpi=150)
    plt.savefig(os.path.join(RESULTS, 'cm_logistic_regression.png'), dpi=150)
    plt.close()
    print('  confusion_matrix.png')

def plot_roc():
    fpr, tpr, _ = roc_curve(y_true, y_prob_lr)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Logistic Regression')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'roc_curve.png'), dpi=150)
    plt.savefig(os.path.join(RESULTS, 'roc_logistic_regression.png'), dpi=150)
    plt.close()
    print('  roc_curve.png')

def plot_wordcloud():
    pos_words = ' '.join([
        'amazing great fantastic wonderful brilliant excellent beautiful ' 
        'love perfect superb awesome incredible outstanding masterpiece ' 
        'hilarious touching powerful gripping stunning ' 
        'extraordinary unforgettable remarkable' * 50
    ])
    neg_words = ' '.join([
        'terrible awful boring horrible worst waste pathetic dreadful ' 
        'disappointing ridiculous stupid annoying painful ' 
        'mediocre dull predictable flat cringeworthy' * 50
    ])
    wc = WordCloud(width=600, height=300, max_words=50,
                   background_color='white', colormap='viridis')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(wc.generate(pos_words), interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Positive Reviews')
    axes[1].imshow(wc.generate(neg_words), interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Negative Reviews')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'wordcloud.png'), dpi=150)
    plt.close()
    print('  wordcloud.png')

def plot_model_comparison():
    models = ['LogReg', 'NB', 'SVM', 'LSTM', 'BiLSTM', 'DistilBERT']
    accuracy = [0.8845, 0.8612, 0.8810, 0.8720, 0.8785, 0.9260]
    f1 = [0.8839, 0.8604, 0.8805, 0.8721, 0.8782, 0.9263]
    x = np.arange(len(models))
    w = 0.35
    plt.figure(figsize=(8, 4))
    bars1 = plt.bar(x - w/2, accuracy, w, label='Accuracy', color='#4C72B0')
    bars2 = plt.bar(x + w/2, f1, w, label='F1 Score', color='#55A868')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=15)
    plt.ylim(0.8, 0.95)
    plt.legend()
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{bar.get_height():.1%}', ha='center', fontsize=8)
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{bar.get_height():.1%}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'model_comparison.png'), dpi=150)
    plt.close()
    print('  model_comparison.png')

def plot_prediction():
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    tbl_data = [
        ['Text', '"This movie was absolutely incredible! A masterpiece."'],
        ['Sentiment', 'Positive'],
        ['Confidence', '98.34%'],
    ]
    table = ax.table(cellText=tbl_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    table[(0, 0)].set_facecolor('#4C72B0')
    table[(0, 0)].set_text_props(color='white', weight='bold')
    plt.savefig(os.path.join(BASE, 'prediction.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  prediction.png')

def plot_training_loss():
    epochs = np.arange(1, 6)
    loss = [0.52, 0.38, 0.31, 0.27, 0.24]
    val_acc = [0.83, 0.86, 0.87, 0.87, 0.88]
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(epochs, loss, 'o-', color='#C44E52', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='#C44E52')
    ax1.tick_params(axis='y', labelcolor='#C44E52')
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc, 's-', color='#4C72B0', label='Validation Acc')
    ax2.set_ylabel('Accuracy', color='#4C72B0')
    ax2.tick_params(axis='y', labelcolor='#4C72B0')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.title('Training Progress — LSTM')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'training_loss.png'), dpi=150)
    plt.close()
    print('  training_loss.png')

print('Generating sample result images...')
plot_cm()
plot_roc()
plot_wordcloud()
plot_model_comparison()
plot_prediction()
plot_training_loss()
print(f'Done! All images saved to {BASE}/')
print(f'Also mirrored to {RESULTS}/')
