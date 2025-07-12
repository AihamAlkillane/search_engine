import numpy as np
from collections import defaultdict


def fuse_parallel_results(tfidf_ids, tfidf_scores, bert_ids, bert_scores, alpha=0.5, top_k=10):
    # Combine scores
    score_dict = defaultdict(lambda: {"tfidf": 0.0, "bert": 0.0})

    for doc_id, score in zip(tfidf_ids, tfidf_scores):
        score_dict[doc_id]["tfidf"] = score
    for doc_id, score in zip(bert_ids, bert_scores):
        score_dict[doc_id]["bert"] = score

    # Weighted fusion
    fused_results = []
    for doc_id, scores in score_dict.items():
        fused_score = alpha * scores["tfidf"] + (1 - alpha) * scores["bert"]
        fused_results.append((doc_id, fused_score))

    # Sort and take top_k
    fused_results.sort(key=lambda x: x[1], reverse=True)
    top_ids = [doc_id for doc_id, _ in fused_results[:top_k]]
    top_scores = [score for _, score in fused_results[:top_k]]

    return top_ids, top_scores
