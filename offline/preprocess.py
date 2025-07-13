import tornado.ioloop
import tornado.web
import json
from db import db_service
from concurrent.futures import ProcessPoolExecutor
from preprocess_utils import preprocess_row,Preprocessor
import functools

class PreprocessHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            data = json.loads(self.request.body)
            dataset_name = data.get("dataset_name")
            options = data.get("options", {})

            if not dataset_name:
                raise ValueError("dataset_name is required")

            table_name = dataset_name + "_corpus"
            rows = db_service.fetch_all_rows(table_name)
            preprocessor = Preprocessor(**options)

            results = []
            count = 0
            for row in rows:
                row_id = row['_id']
                text = row['text']
                preprocessed_text = preprocessor.preprocess(text)
                results.append((row_id, preprocessed_text))
                count += 1
                if count % 10000 == 0:
                    print(f"Processed {count} rows", flush=True)


            db_service.bulk_update_preprocessed_data(table_name, results)

            self.set_header("Content-Type", "application/json")
            self.write({
                "message": f"Preprocessing finished for dataset {dataset_name}. Total rows processed: {count}"
            })

        except Exception as e:
            self.set_status(400)
            self.write({"error": str(e)})
