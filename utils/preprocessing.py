import re
import nltk
import contractions
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001F900-\U0001F9FF"
    u"\U0001FA00-\U0001FA6F"
    u"\U0001FA70-\U0001FAFF"
    u"\U00002600-\U000026FF"
    u"\U0000FE00-\U0000FE0F"
    "]+", flags=re.UNICODE
)

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
REPEATED_CHAR_PATTERN = re.compile(r'(.)\1{2,}')
NON_ALPHA_PATTERN = re.compile(r'[^a-zA-Z\s]')
MULTI_SPACE_PATTERN = re.compile(r'\s+')


def lowercase(text):
    return text.lower()


def remove_urls(text):
    return URL_PATTERN.sub('', text)


def remove_html_tags(text):
    try:
        return BeautifulSoup(text, 'html.parser').get_text(separator=' ')
    except Exception:
        return HTML_TAG_PATTERN.sub('', text)


def handle_emojis(text):
    return EMOJI_PATTERN.sub(' ', text)


def expand_contractions(text):
    try:
        return contractions.fix(text)
    except Exception:
        return text


def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in STOPWORDS])


def lemmatize(text):
    return ' '.join([LEMMATIZER.lemmatize(word) for word in text.split()])


def normalize_repeated_chars(text):
    return REPEATED_CHAR_PATTERN.sub(r'\1\1', text)


def clean_text(text, remove_stopwords_flag=True, lemmatize_flag=True):
    text = lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = handle_emojis(text)
    text = expand_contractions(text)
    text = normalize_repeated_chars(text)
    text = NON_ALPHA_PATTERN.sub('', text)
    text = MULTI_SPACE_PATTERN.sub(' ', text).strip()
    if remove_stopwords_flag:
        text = remove_stopwords(text)
    if lemmatize_flag:
        text = lemmatize(text)
    return text
