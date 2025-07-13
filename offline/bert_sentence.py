import numpy as np
import tornado
import json
import joblib
import faiss
import os
from sentence_transformers import SentenceTransformer
import db.db_service

MODEL_PATH = "C:/Users/USER/PycharmProjects/SearchEngine/models/"
model = SentenceTransformer("all-MiniLM-L6-v2")


class BertSentenceHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            data = json.loads(self.request.body)
            dataset_name = data["dataset_name"]

            folder_path = os.path.join(MODEL_PATH, dataset_name, "bert")
            os.makedirs(folder_path, exist_ok=True)

            table_name = dataset_name + "_corpus"
            rows = db.db_service.fetch_all_rows(table_name)
            documents = []
            id_map = {}
            for idx, row in enumerate(rows):
                content = row['text']
                doc_id = row['_id']
                documents.append(content)
                id_map[idx] = {
                    "_id": doc_id,
                }

            embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
            faiss.normalize_L2(embeddings)

            np.save(f"{MODEL_PATH}{dataset_name}/bert/vectors.npy", embeddings)
            joblib.dump(id_map, f"{MODEL_PATH}{dataset_name}/bert/id_map.joblib", compress=3)

            self.set_header("Content-Type", "application/json")
            self.write({
                "message": f"BERT embeddings for dataset '{dataset_name}'"
            })

        except Exception as e:
            self.set_status(400)
            self.write({"error": str(e)})
