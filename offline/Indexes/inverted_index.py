
import json
import tornado.web
import pickle
from db.db_service import fetch_all_rows
from offline.services.indexing_service import extract_valid_terms, build_inverted_index


class InvertedIndex(tornado.web.RequestHandler):
    async def post(self):
        try:
            data = json.loads(self.request.body)
            table_name = data.get("table_name") + "_corpus"
            df_max = float(data.get("df_max", 0.8))

            if not table_name:
                self.set_status(400)
                self.write({"error": "Missing 'table_name'"})
                return

            rows = fetch_all_rows(table_name)
            valid_terms = extract_valid_terms(rows , df_max)
            inverted_index = build_inverted_index(rows, valid_terms)

            file_path = "C:/Users/USER/PycharmProjects/SearchEngine/models/" + table_name + "_inverted_index.pkl"

            with open(file_path, "wb") as f:
                pickle.dump(inverted_index, f)

            self.write({"message" : "inverted_index created"})
        except Exception as e:
            self.set_status(500)
            self.write({"error": str(e)})
