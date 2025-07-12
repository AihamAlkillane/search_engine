import joblib
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from db import db_service
from online.piplines.tfidf.query_processing import preprocess_query
from online.piplines.tfidf.index_matching import match_inverted_index
from online.piplines.tfidf.vectorization import vectorize_query, get_candidate_vectors
from online.piplines.tfidf.ranking import rank_results

vectorizer = joblib.load("/models/arguana/tf-idf/vectorizer.joblib")
tfidf_matrix = joblib.load("/models/arguana/vectors.joblib")
doc_ids = db_service.fetch_all_ids("arguana_corpus")
with open("/models/arguana/inverted_index.joblib", "rb") as f:
    inverted_index = pickle.load(f)
import json

queries = {}
with open("C:/Users/USER/DataSets/arguana/queries.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        queries[item["_id"]] = item["text"]

from collections import defaultdict
import csv

qrels = defaultdict(list)
with open("C:/Users/USER/DataSets/arguana/qrels/test.tsv", encoding="utf-8") as f:
    tsv_reader = csv.reader(f, delimiter="\t")
    next(tsv_reader)  # لتجاوز رأس الجدول (header)
    for row in tsv_reader:
        query_id, doc_id, score = row
        if float(score) > 0:
            qrels[query_id].append(doc_id)


def retrieve_top_k_docs(query_text, inverted_index, vectorizer, tfidf_matrix, doc_ids, top_k=10):
    # Step 1: Query Preprocessing
    cleaned_query, query_tokens = preprocess_query(query_text)

    # Step 2: Inverted Index Matching
    candidate_doc_ids = match_inverted_index(query_tokens, inverted_index)
    if not candidate_doc_ids:
        return [], []

    # Step 3: Vectorization
    query_vector = vectorize_query(cleaned_query, vectorizer)
    candidate_ids, candidate_vectors = get_candidate_vectors(candidate_doc_ids, doc_ids, tfidf_matrix)
    if len(candidate_ids) == 0:
        return [], []

    # Step 4: Ranking
    similarities = cosine_similarity(query_vector, candidate_vectors).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_doc_ids = [candidate_ids[i] for i in top_indices]
    top_scores = similarities[top_indices]

    return top_doc_ids, top_scores


def precision_at_k(relevant, retrieved, k=10):
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    return sum(1 for doc in retrieved_k if doc in relevant_set) / k


def recall_at_k(relevant, retrieved, k=10):
    relevant_set = set(relevant)
    retrieved_k = retrieved[:k]
    return sum(1 for doc in retrieved_k if doc in relevant_set) / len(relevant)


def average_precision(relevant, retrieved):
    relevant_set = set(relevant)
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            hits += 1
            sum_precisions += hits / (i + 1)
    if hits == 0:
        return 0.0
    return sum_precisions / hits


def reciprocal_rank(relevant, retrieved):
    for i, doc_id in enumerate(retrieved):
        if doc_id in set(relevant):
            return 1 / (i + 1)
    return 0.0

precisions, recalls, average_precisions, reciprocal_ranks = [], [], [], []

for qid, query_text in queries.items():
    relevant_docs = qrels.get(qid, [])
    retrieved_docs, _ = retrieve_top_k_docs(
        query_text=query_text,
        inverted_index=inverted_index,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        doc_ids=doc_ids,
        top_k=100
    ) # تقييم على top-100 مثلاً

    precisions.append(precision_at_k(relevant_docs, retrieved_docs, k=10))
    recalls.append(recall_at_k(relevant_docs, retrieved_docs, k=10))
    average_precisions.append(average_precision(relevant_docs, retrieved_docs))
    reciprocal_ranks.append(reciprocal_rank(relevant_docs, retrieved_docs))

print(f"Precision@10: {np.mean(precisions):.4f}")
print(f"Recall@10: {np.mean(recalls):.4f}")
print(f"MAP: {np.mean(average_precisions):.4f}")
print(f"MRR: {np.mean(reciprocal_ranks):.4f}")
