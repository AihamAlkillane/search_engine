import re
from sklearn.metrics.pairwise import cosine_similarity
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords


class TfIdfPipeline:
    def __init__(self, vectorizer, tf_idf_matrix, inverted_index, ids, flat_ip_index):
        self.vectorizer = vectorizer
        self.tf_idf_matrix = tf_idf_matrix
        self.inverted_index = inverted_index
        self.ids = ids
        self.flat_ip_index = flat_ip_index
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words("english"))

    def processing_query(self, query):
        text = query.lower()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stopwords]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens), tokens

    def index_matching(self, query_tokens):
        matched_ids = set()
        for token in query_tokens:
            if token in self.inverted_index:
                matched_ids.update(self.inverted_index[token])
        return matched_ids  # candidate_ids

    def vectorize_query(self, cleaned_query):
        return self.vectorizer.transform([cleaned_query])

    def get_candidate_vectors(self, candidate_ids):
        id_to_index = {doc_id: i for i, doc_id in enumerate(self.ids)}
        indices = [id_to_index[_id] for _id in candidate_ids if _id in id_to_index]

        matched_ids = [self.ids[i] for i in indices]
        matched_vecs = self.tf_idf_matrix[indices, :]
        return matched_ids, matched_vecs

    def rank_results(self, query_vector, candidate_vectors, candidate_ids, top_k):
        similarities = cosine_similarity(query_vector, candidate_vectors)[0]
        ranked = sorted(zip(candidate_ids, similarities), key=lambda x: -x[1])
        top_ranked = ranked[:top_k]

        top_ids = [doc_id for doc_id, _ in top_ranked]
        top_scores = [score for _, score in top_ranked]

        return top_ids, top_scores

    def run_with_inverted_index(self, query, top_k=10):
        cleaned_query, tokens = self.processing_query(query)
        candidate_ids = self.index_matching(tokens)
        vectorized_query = self.vectorize_query(cleaned_query)
        matched_ids, matched_vectors = self.get_candidate_vectors(candidate_ids)
        return self.rank_results(vectorized_query, matched_vectors, matched_ids, top_k)

    def run_with_flat_ip_index(self, query, top_k=10):
        cleaned_query, _ = self.processing_query(query)
        query_vector = self.vectorize_query(cleaned_query)
        query_vector_np = query_vector.astype("float32").toarray()
        scores, indices = self.flat_ip_index.search(query_vector_np, top_k)
        top_ids = [self.ids[i] for i in indices[0]]
        top_scores = scores[0].tolist()
        return top_ids, top_scores
