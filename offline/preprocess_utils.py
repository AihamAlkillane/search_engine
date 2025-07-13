from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import regex as re
from db import db_service

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer


class Preprocessor:
    def __init__(self, **options):

        self.lowercase = options.get("lowercase", True)
        self.remove_stopwords = options.get("remove_stopwords", False)
        self.lemmatize = options.get("lemmatize", False)
        self.stemming = options.get("stemming", False)
        self.remove_punctuation = options.get("remove_punctuation", False)
        self.remove_digits = options.get("remove_digits", False)
        self.digits_as_year = options.get("digits_as_year", False)
        self.remove_short_words = options.get("remove_short_words", False)
        self.custom_stopwords = set(options.get("custom_stopwords", []))

        self.stop_words = stopwords.words('english')
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        if self.lowercase:
            text = text.lower()

        tokens = word_tokenize(text)

        if self.remove_stopwords:
            full_stopwords = set(self.stop_words).union(self.custom_stopwords)
            tokens = [t for t in tokens if t not in full_stopwords]

        if self.remove_punctuation:
            tokens = [t for t in tokens if any(c.isalnum() for c in t)]

        if self.remove_digits:
            tokens = [t for t in tokens if not t.isdigit()]
        elif self.digits_as_year:
            tokens = [t for t in tokens if t.isalpha() or (t.isdigit() and len(t) == 4)]
        else:
            tokens = [t for t in tokens if t.isalpha()]

        if self.remove_short_words:
            tokens = [t for t in tokens if len(t) >= 3]

        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        if self.stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return ' '.join(tokens)


def preprocess_row(row, preprocessor, table_name):
    row_id = row['_id'] if isinstance(row, dict) else row[0]
    text = row['text'] if isinstance(row, dict) else row[1]

    preprocessed_text = preprocessor.preprocess(text)
    db_service.update_preprocessed_data(table_name, row_id, preprocessed_text)
    return row_id
