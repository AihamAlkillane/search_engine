from collections import defaultdict

def extract_valid_terms(rows, max_df_threshold=0.8):
    doc_freq = defaultdict(int)
    total_docs = len(rows)

    for row in rows:
        text = row['preprocessed_data']
        tokens = set(text.split())
        for term in tokens:
            doc_freq[term] += 1

    max_df = int(total_docs * max_df_threshold)
    valid_terms = {
        term for term, df in doc_freq.items()
        if df <= max_df
    }

    return valid_terms


def build_inverted_index(rows, valid_terms):
    inverted_index = defaultdict(set)

    for row in rows:
        doc_id = row['_id']
        text = row['preprocessed_data']
        tokens = set(text.split())
        for term in tokens:
            if term in valid_terms:
                inverted_index[term].add(doc_id)

    return inverted_index
