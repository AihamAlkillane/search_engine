import tornado.ioloop
import tornado.web
import json
import string
from db import db_service
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(
        text,
        remove_stopwords=False,
        lemmatize=False,
        stemming=False,
        lowercase=True,
        digits_as_year=False,
        remove_digits=False,
        remove_punctuation=False,
        remove_short_words=False,
        custom_stopwords=None
):
    if lowercase:
        text = text.lower()

    tokens = word_tokenize(text)

    if remove_punctuation:
        tokens = [t for t in tokens if t not in string.punctuation]

    if remove_digits:
        tokens = [t for t in tokens if not t.isdigit()]
    elif digits_as_year:
        tokens = [t for t in tokens if t.isalpha() or (t.isdigit() and len(t) == 4)]
    else:
        tokens = [t for t in tokens if t.isalpha()]

    if remove_stopwords:
        full_stopwords = stop_words.union(set(custom_stopwords or []))
        tokens = [t for t in tokens if t not in full_stopwords]

    if remove_short_words:
        tokens = [t for t in tokens if len(t) >= 3]

    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    if stemming:
        tokens = [stemmer.stem(t) for t in tokens]
    processed_text = ' '.join(tokens)
    return processed_text


class PreprocessHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            data = json.loads(self.request.body)
            dataset_name = data.get("dataset_name")
            options = data.get("options", {})

            table_name = dataset_name + "_corpus"

            if not dataset_name:
                raise ValueError("dataset_name")

            rows = db_service.fetch_all_rows(table_name)

            for row in rows:
                row_id = row["_id"]
                excluded_fields = {"_id", "preprocessed_data"}
                row_text_data = {k: v for k, v in row.items() if k not in excluded_fields}
                combined_text = " ".join(str(value) for value in row_text_data.values())

                preprocessed_text = preprocess(
                    combined_text,
                    options.get("remove_stopwords", False),
                    options.get("lemmatize", False),
                    options.get("stemming", False),
                    options.get("lowercase", True),
                    options.get("digits_as_year", False),
                    options.get("remove_digits", False),
                    options.get("remove_punctuation", False),
                    options.get("remove_short_words", False),
                    options.get("custom_stopwords", None),
                )
                db_service.update_preprocessed_data(table_name, row_id, preprocessed_text)

            self.set_header("Content-Type", "application/json")
            self.write({
                "message": f"Preprocessing completed for dataset {dataset_name}"
            })

        except Exception as e:
            self.set_status(400)
            self.write({"error": str(e)})
