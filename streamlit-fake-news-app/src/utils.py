import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer

# Baixe apenas uma vez
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()
tb = TreebankWordTokenizer()

def preprocess(text):
    if not isinstance(text, str):
        text = '' if text is None else str(text)

    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)   # URLs
    text = re.sub(r'<.*?>', ' ', text)              # HTML
    tokens = tb.tokenize(text)

    # Limpa tokens: tira pontuação e stopwords, mantém só letras com len>2
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # Stemming
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def tokens_to_features(tokens):
    return {word: True for word in tokens}