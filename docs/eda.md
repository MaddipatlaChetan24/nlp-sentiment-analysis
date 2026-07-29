# Exploratory Data Analysis

Run EDA with:

```bash
python -m analysis.eda
```

All plots are saved to `results/eda/`.

## Class Distribution

The IMDB dataset is perfectly balanced:
- Positive reviews: 25,000 (50%)
- Negative reviews: 25,000 (50%)

## Review Length Distribution

- Mean review length: ~230 words
- Positive reviews tend to be slightly longer than negative reviews
- Most reviews fall between 50–400 words

## Most Frequent Words

### Positive Reviews
Top words: `great`, `best`, `love`, `amazing`, `wonderful`, `excellent`, `fantastic`, `brilliant`, `fun`, `beautiful`

### Negative Reviews
Top words: `bad`, `worst`, `terrible`, `awful`, `boring`, `horrible`, `waste`, `poor`, `dull`, `disappointing`

## N-gram Analysis

### Common Bigrams
- Positive: `highly recommend`, `must watch`, `well done`, `great performance`, `highly recommended`
- Negative: `waste time`, `bad acting`, `worst movie`, `special effects`, `could not`

### Common Trigrams
- Positive: `one of best`, `worth watching`, `well worth watching`, `highly recommend this`
- Negative: `waste of time`, `not even good`, `could not believe`, `nothing good about`

## Word Clouds

Word clouds are generated separately for positive and negative reviews, displaying the 100 most frequent words in each class with size proportional to frequency.
