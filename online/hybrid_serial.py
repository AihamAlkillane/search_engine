def run_hybrid_serial(self, query, tf_idf_pipeline, bert_sentence_pipeline, top_k=10, candidates_k=1000):
    tf_idf_pipeline.run_with_inverted_index(query, co)
    # Optional: Limit number of candidates
    if len(candidate_ids) > candidates_k:
        candidate_ids = candidate_ids[:candidates_k]

    # Step 3: Vectorize query using BERT
    vectorized_query = self.encode_query(cleaned_query)

    # Step 4: Get vectors of candidate docs
    matched_ids, matched_vectors = self.get_candidate_vectors(candidate_ids)

    # Step 5: Rank candidates based on similarity
    return self.rank_results(vectorized_query, matched_vectors, matched_ids, top_k)
