from collections import defaultdict
from db import db_service
from online.bert_sentence_pipeline import BertSentencePipeline
from online.tf_idf_pipeline import TfIdfPipeline
import numpy as np
import csv
import json
import joblib
import faiss

MODEL_PATH = "C:/Users/USER/PycharmProjects/SearchEngine/models/"

def load_qrels(qrels_path):
    qrels = defaultdict(list)
    with open(qrels_path, encoding="utf-8") as f:
        tsv_reader = csv.reader(f, delimiter="\t")
        for row in tsv_reader:
            if len(row) < 4:
                continue  # تجاهل الصفوف غير المكتملة
            query_id, _, doc_id, score = row
            if float(score) > 0:
                qrels[query_id].append(doc_id)
    return qrels


def load_queries(queries_path):
    queries = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qid, query_text = parts[0], parts[1]
                queries[qid] = query_text
    return queries


def tf_idf_pipeline(dataset_name):
    vectorizer = joblib.load(f"{MODEL_PATH}{dataset_name}/tf-idf/vectorizer.joblib")
    vectors = joblib.load(f"{MODEL_PATH}{dataset_name}/tf-idf/vectors.joblib")
    inverted_index = joblib.load(f"{MODEL_PATH}{dataset_name}/tf-idf/inverted_index.joblib")
    #flat_ip_index = faiss.read_index(f"{MODEL_PATH}{dataset_name}/tf-idf/flat_ip_index.index")
    ids = db_service.fetch_all_ids(dataset_name + "_corpus")
    return TfIdfPipeline(vectorizer, vectors, inverted_index, ids, None)


def bert_sentence_pipeline(dataset_name):
    flat_ip_index_bert = faiss.read_index(f"{MODEL_PATH}{dataset_name}/bert/flat_ip_index.index")
    bert_vectors = np.load(f"{MODEL_PATH}{dataset_name}/bert/vectors.npy")
    ids = db_service.fetch_all_ids(dataset_name + "_corpus")
    return BertSentencePipeline(flat_ip_index_bert, bert_vectors, ids)
