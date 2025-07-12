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
