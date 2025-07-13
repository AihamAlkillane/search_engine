import json

import tornado
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

import db.db_service

MODEL_PATH = "C:/Users/USER/PycharmProjects/SearchEngine/models/"


class TfidfHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            data = json.loads(self.request.body)
            dataset_name = data["dataset_name"]
            df_min = data["df_min"]
            df_max = data["df_max"]

            folder_path = os.path.join(MODEL_PATH, dataset_name, "tf-idf")
            os.makedirs(folder_path, exist_ok=True)

            vectorizer = TfidfVectorizer(
                lowercase=False,
                preprocessor=None,
                tokenizer=str.split,
                token_pattern=None,
                min_df=df_min,
                max_df=df_max,
                norm="l2"
            )

            preprocessed_data = db.db_service.fetch_preprocessed_data(dataset_name + "_corpus")
            tf_idf_vectors = vectorizer.fit_transform(preprocessed_data)

            joblib.dump(vectorizer, f"{MODEL_PATH}{dataset_name}/tf-idf/vectorizer.joblib", compress=0)
            joblib.dump(tf_idf_vectors, f"{MODEL_PATH}{dataset_name}/tf-idf/vectors.joblib", compress=0)

            self.set_header("Content-Type", "application/json")
            self.write({
                "message": f"tf-idf_vectorizer training completed for dataset {dataset_name}"
            })

        except Exception as e:
            self.set_status(400)
            self.write({"error": str(e)})
