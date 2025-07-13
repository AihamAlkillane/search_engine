# server.py
import tornado.ioloop
import tornado.web
import joblib
import faiss
import numpy as np
import os

from search import SearchHandler
from db import db_service
from tf_idf_pipeline import TfIdfPipeline
from bert_sentence_pipeline import BertSentencePipeline

MODEL_PATH = "C:/Users/USER/PycharmProjects/SearchEngine/models/"


def load_pipeline(dataset_name):
    vectorizer = joblib.load(f"{MODEL_PATH}{dataset_name}/tf-idf/vectorizer.joblib")
    vectors = joblib.load(f"{MODEL_PATH}{dataset_name}/tf-idf/vectors.joblib")
    inverted_index = joblib.load(f"{MODEL_PATH}{dataset_name}/tf-idf/inverted_index.joblib")
    flat_ip_index_path = f"{MODEL_PATH}{dataset_name}/tf-idf/flat_ip_index.index"
    if os.path.exists(flat_ip_index_path):
        flat_ip_index = faiss.read_index(flat_ip_index_path)
    else:
        flat_ip_index = None


    ids = db_service.fetch_all_ids(dataset_name + "_corpus")

    flat_ip_index_bert = faiss.read_index(f"{MODEL_PATH}{dataset_name}/bert/flat_ip_index.index")
    bert_vectors = np.load(f"{MODEL_PATH}{dataset_name}/bert/vectors.npy")

    return (TfIdfPipeline(vectorizer, vectors, inverted_index, ids, flat_ip_index),
            BertSentencePipeline(flat_ip_index_bert, bert_vectors, ids))


def make_app():
    DATASET_NAME = "lifestyle"
    tfidf_pipeline, bert_sentence_pipline = load_pipeline(DATASET_NAME)

    return tornado.web.Application([
        (r"/search", SearchHandler,
         dict(tfidf_pipeline=tfidf_pipeline, bert_sentence_pipline=bert_sentence_pipline, dataset_name=DATASET_NAME)),
        # (r"/query_handler", QueryHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8080)
    print("Server running at http://localhost:8080")
    tornado.ioloop.IOLoop.current().start()
