import re
from sentence_transformers import SentenceTransformer
import numpy as np


class BertSentencePipeline:
    def __init__(self, flat_ip_index, vectors, ids):
        self.vectors = vectors
        self.flat_ip_index = flat_ip_index
        self.ids = ids
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.id_to_index = {doc_id: i for i, doc_id in enumerate(self.ids)}

    def processing_query(self, query):
        query = query.lower()
        query = re.sub(r'[^a-z0-9\s,.?!]', '', query)
        query = query.strip()
        return query

    def encode_query(self, cleaned_query):
        embedding = self.model.encode(cleaned_query, normalize_embeddings=True)
        return embedding.reshape(1, -1)

    def flat_ip_index_matching(self, encoded_query, top_k):
        scores, indices = self.flat_ip_index.search(encoded_query, top_k)
        top_ids = [self.ids[idx] for idx in indices[0]]
        return top_ids, scores

    def run_with_flat_ip_index(self, query, top_k=10):
        cleaned_query = self.processing_query(query)
        encoded_query = self.encode_query(cleaned_query)
        top_ids, scores = self.flat_ip_index_matching(encoded_query, top_k)

        return top_ids, scores[0].tolist()

    def run_with_filtered_candidates(self, query, candidate_ids, top_k=10):
        cleaned_query = self.processing_query(query)
        query_vector = self.encode_query(cleaned_query)

        candidate_vectors = []
        valid_ids = []

        for doc_id in candidate_ids:
            index = self.id_to_index.get(doc_id)
            if index is not None:
                vec = self.flat_ip_index.reconstruct(index)  # أو self.doc_vectors[doc_id]
                candidate_vectors.append(vec)
                valid_ids.append(doc_id)

        if not candidate_vectors:
            return [], []

        candidate_vectors = np.array(candidate_vectors)
        scores = np.dot(candidate_vectors, query_vector.T).squeeze()

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_ids = [valid_ids[i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]

        return top_ids, top_scores
