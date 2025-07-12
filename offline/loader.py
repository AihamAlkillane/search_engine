import os
import tornado.ioloop
import tornado.web
import json
from db import db_service

BASE_DIR = "C:/Users/USER/PycharmProjects/SearchEngine/offline/storage"
TYPES = ["corpus", "queries", "qrels"]


class LoaderHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            dataset_name = self.get_argument("dataset_name")
            file_type = self.get_argument("file_type")
            columns = json.loads(self.get_argument("columns"))

            folder_path = os.path.join(BASE_DIR, dataset_name)
            os.makedirs(folder_path, exist_ok=True)

            save_path = os.path.join(folder_path, f"{file_type}.jsonl")
            fileinfo = self.request.files["file"][0]
            with open(save_path, 'wb') as f:
                f.write(fileinfo['body'])

            table_name = dataset_name + "_" + file_type
            db_service.create_table(table_name, columns)
            db_service.insert_from_jsonl(table_name, columns, save_path)
            self.write({
                "message": "File loaded successfully",
            })

        except Exception as e:
            self.set_status(400)
            self.write({"error": str(e)})
