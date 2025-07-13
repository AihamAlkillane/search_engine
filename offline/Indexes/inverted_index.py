
import json
import tornado.web
import joblib
from db.db_service import fetch_all_rows
from offline.services.indexing_service import extract_valid_terms, build_inverted_index

MODEL_PATH = "C:/Users/USER/PycharmProjects/SearchEngine/models/"

class InvertedIndex(tornado.web.RequestHandler):
    async def post(self):
        try:
            data = json.loads(self.request.body)
            dataset_name = data.get("dataset_name")
            df_max = float(data.get("df_max", 0.8))

            table_name = dataset_name + "_corpus"

            rows = fetch_all_rows(table_name)
            valid_terms = extract_valid_terms(rows , df_max)
            inverted_index = build_inverted_index(rows, valid_terms)

            file_path = f"{MODEL_PATH}{dataset_name}/tf-idf/inverted_index.joblib"
            joblib.dump(inverted_index, file_path)

            self.write({"message" : "inverted_index created"})
        except Exception as e:
            self.set_status(500)
            self.write({"error": str(e)})
