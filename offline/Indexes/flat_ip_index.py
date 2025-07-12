import tornado
import json
import faiss
import joblib
import numpy as np

MODEL_PATH = "C:/Users/USER/PycharmProjects/SearchEngine/models"


class FlatIPIndexHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            data = json.loads(self.request.body)
            dataset_name = data["dataset_name"]
            vectors_type = data["vectors_type"]

            if vectors_type == "embeddings_bert":
                embeddings = np.load(f"{MODEL_PATH}/{dataset_name}/{vectors_type}/vectors.npy")
                d = embeddings.shape[1]

                index = faiss.IndexFlatIP(d)
                index.add(embeddings)

                faiss.write_index(index, f"{MODEL_PATH}/{dataset_name}/{vectors_type}/falt_ip_index.index")

            if vectors_type == "tf-idf":
                tf_idf_vectors = joblib.load(f"{MODEL_PATH}/{dataset_name}/{vectors_type}/vectors.joblib")
                dense_vectors = tf_idf_vectors.toarray().astype('float32')

                index = faiss.IndexFlatIP(dense_vectors.shape[1])
                index.add(dense_vectors)

                faiss.write_index(index, f"{MODEL_PATH}/{dataset_name}/{vectors_type}/falt_ip_index.index")

            self.set_header("Content-Type", "application/json")
            self.write({
                "message": f"IndexFlatIP built and saved for dataset '{dataset_name}'"
            })

        except Exception as e:
            self.set_status(400)
            self.write({"error": str(e)})
