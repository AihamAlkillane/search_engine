from psycopg2 import sql
import psycopg2
import json
from psycopg2.extras import RealDictCursor

import db.config

DB_CONFIG = db.config.DB_CONFIG


def create_table(table_name, columns):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        columns_sql_parts = []
        for col in columns:
            if col.lower() == "_id":
                columns_sql_parts.append(f"{col} INT PRIMARY KEY")
            else:
                columns_sql_parts.append(f"{col} TEXT")

        columns_sql_parts.append("preprocessed_data TEXT")

        columns_sql = ", ".join(columns_sql_parts)
        create_query = sql.SQL(
            f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql});"
        )

        cur.execute(create_query)
        conn.commit()

        print(f"Table '{table_name}' created with columns: {columns}")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error creating table: {e}")


def insert_from_tsv(table_name, columns, file_path):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split('\t')

                if len(values) != len(columns):
                    print(f"Skipping line due to column mismatch: {line}")
                    continue

                placeholders = sql.SQL(', ').join(sql.Placeholder() * len(values))

                insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    sql.Identifier(table_name),
                    sql.SQL(', ').join(map(sql.Identifier, columns)),
                    placeholders
                )
                cur.execute(insert_query, values)

        conn.commit()
        cur.close()
        conn.close()

        print(f"Inserted data from {file_path} into {table_name}")

    except Exception as e:
        print(f"Error inserting data: {e}")


def insert_from_jsonl(table_name, columns, file_path):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())

                values = [str(record.get(col, "")) for col in columns]
                placeholders = ', '.join(['%s'] * len(values))

                insert_query = sql.SQL(
                    f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                )

                cur.execute(insert_query, values)

        conn.commit()
        cur.close()
        conn.close()

        print(f"Inserted data from {file_path} into {table_name}")

    except Exception as e:
        print(f"Error inserting data: {e}")


def fetch_all_rows(table_name):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {table_name} ORDER BY _id ASC")
            return cur.fetchall()
    finally:
        conn.close()


def fetch_preprocessed_data(table_name):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            query = f"SELECT preprocessed_data FROM {table_name} ORDER BY _id ASC"
            cur.execute(query)
            rows = cur.fetchall()
            return [row[0] if row[0] is not None else "" for row in rows]
    finally:
        conn.close()


def fetch_all_text(table_name):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            query = f"SELECT text FROM {table_name} ORDER BY _id ASC"
            cur.execute(query)
            rows = cur.fetchall()
            return [row[0] if row[0] is not None else "" for row in rows]
    finally:
        conn.close()


def update_preprocessed_data(table_name, row_id, preprocessed_text):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            query = f"""
                  UPDATE {table_name}
                  SET preprocessed_data = %s
                  WHERE _id = %s
              """
            cur.execute(query, (preprocessed_text, row_id))
            conn.commit()
    finally:
        conn.close()


def fetch_documents_by_ids(table_name, ids):
    placeholders = ', '.join(['%s'] * len(ids))  # For safe query
    query = f"SELECT * FROM {table_name} WHERE _id IN ({placeholders}) "
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, ids)
            return cur.fetchall()
    finally:
        conn.close()


def bulk_update_preprocessed_data(table_name, updates):
    """
    تحديث عدة صفوف دفعة واحدة.

    :param table_name: اسم الجدول
    :param updates: قائمة tuples بالشكل (row_id, preprocessed_text)
    """

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn:
            with conn.cursor() as cur:
                for row_id, preprocessed_text in updates:
                    query = f"""
                        UPDATE {table_name}
                        SET preprocessed_data = %s
                        WHERE _id = %s
                    """
                    cur.execute(query, (preprocessed_text, row_id))
    finally:
        conn.close()


def fetch_all_ids(table_name):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute(f"SELECT _id FROM {table_name} ORDER BY _id ASC")
            rows = cur.fetchall()
            return [row[0] for row in rows]
    finally:
        conn.close()
