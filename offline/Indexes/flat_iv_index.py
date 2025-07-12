import tornado
import json
import joblib
import faiss
import os

MODEL_PATH = "C:/Users/USER/PycharmProjects/SearchEngine/models/"
INDEX_PATH = os.path.join(MODEL_PATH, "indexes")
os.makedirs(INDEX_PATH, exist_ok=True)

class BuildIVFFlatIndexHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            # قراءة بيانات الطلب
            data = json.loads(self.request.body)
            dataset_name = data["dataset_name"]
            nlist = data.get("nlist", 100)  # عدد المناطق (centroids)

            embeddings = joblib.load(f"{MODEL_PATH}/{dataset_name}_embeddings.joblib")
            d = embeddings.shape[1]

            quantizer = faiss.IndexFlatIP(d)

            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

            index.train(embeddings)

            index.add(embeddings)

            index_path = os.path.join(INDEX_PATH, f"{dataset_name}_ivf.index")
            faiss.write_index(index, index_path)

            self.set_header("Content-Type", "application/json")
            self.write({
                "message": f"IndexIVFFlat built and saved at '{index_path}'",
                "n_vectors": len(embeddings),
                "nlist": nlist
            })

        except Exception as e:
            self.set_status(400)
            self.write({"error": str(e)})
