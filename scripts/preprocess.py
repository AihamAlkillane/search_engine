import psycopg2
from app.preprocess import preprocess

conn = psycopg2.connect(host="localhost", database="SearchEngine_db", user="postgres", password="1234")
cur = conn.cursor()

cur.execute("SELECT id, title, text FROM arguana_corpus")
rows = cur.fetchall()

for doc_id, title, text in rows:
    fulltext = (title or "") + " " + (text or "")
    preprocessed = preprocess(fulltext)
    cur.execute("UPDATE arguana_corpus SET preprocessed_text = %s WHERE id = %s", (preprocessed, doc_id))

conn.commit()
cur.close()
conn.close()
