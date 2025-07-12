### main.py
import tornado.ioloop
import tornado.web
import loader
import preprocess
import tfidf_vectorizer
import bert_sentence
from offline.Indexes import inverted_index
from offline.Indexes import flat_ip_index


def make_app():
    return tornado.web.Application([
        (r"/loader", loader.LoaderHandler),
        (r"/preprocess", preprocess.PreprocessHandler),
        (r"/vectorizer", tfidf_vectorizer.TfidfHandler),
        (r"/inverted_index", inverted_index.InvertedIndex),
        (r"/bert_sentence", bert_sentence.BertSentenceHandler),
        (r"/flat_ip_index", flat_ip_index.FlatIPIndexHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8000)
    print("Server running at http://localhost:8000")
    tornado.ioloop.IOLoop.current().start()
