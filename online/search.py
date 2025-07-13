import joblib
import tornado.web
from fuse_parallel_results import fuse_parallel_results
from db import db_service


class SearchHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def options(self):
        self.set_status(204)
        self.finish()

    def initialize(self, tfidf_pipeline, bert_sentence_pipline, dataset_name):
        self.tfidf_pipeline = tfidf_pipeline
        self.bert_sentence_pipline = bert_sentence_pipline
        self.dataset_name = dataset_name

    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        query = data.get("query", "")
        model_type = data.get("model_type", "")
        index_type = data.get("index_type", "")

        top_ids, top_scores = [], []

        if model_type == "tf-idf":
            if index_type == "inverted_index":
                top_ids, top_scores = self.tfidf_pipeline.run_with_inverted_index(query)
            elif index_type == "flat_ip_index":
                top_ids, top_scores = self.tfidf_pipeline.run_with_flat_ip_index(query)

        elif model_type == "bert-sentence":
            if index_type == "flat_ip_index":
                top_ids, top_scores = self.bert_sentence_pipline.run_with_flat_ip_index(query)
                print("Scores:", top_scores)
                print("Indices:", top_ids)
        elif model_type == "hybrid-serial":
            candidates_ids, top_scores = self.tfidf_pipeline.run_with_inverted_index(query, 1000)
            top_ids, top_scores = self.bert_sentence_pipline.run_with_filtered_candidates(query, candidates_ids, 10)

        elif model_type == "hybrid-parallel":
            tfidf_ids, tfidf_scores  = self.tfidf_pipeline.run_with_inverted_index(query,1000)
            bert_ids,bert_scores = self.bert_sentence_pipline.run_with_flat_ip_index(query,1000)
            top_ids, top_scores = fuse_parallel_results(tfidf_ids, tfidf_scores, bert_ids, bert_scores)

        results_rows = db_service.fetch_documents_by_ids(self.dataset_name + "_corpus", top_ids)
        id_to_score = {doc_id: float(score) for doc_id, score in zip(top_ids, top_scores)}

        results = []
        for row in results_rows:
            doc_id = row["_id"]
            result = {
                "id": doc_id,
                "score": id_to_score.get(doc_id, 0.0),
                "text": row.get("text", "")
            }
            results.append(result)

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        self.write({"results": results})
