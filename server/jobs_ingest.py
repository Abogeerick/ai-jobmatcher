import csv
import json
import sqlite3
import os
from utils import get_embedding

ROOT = os.path.dirname(__file__)
CSV_PATH = os.path.join(ROOT, "../sample_data/sample_jobs.csv")
DB_PATH = os.path.join(ROOT, "jobs.db")

def create_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            title TEXT,
            company TEXT,
            description TEXT,
            embedding TEXT
        )
    """)
    conn.commit()
    conn.close()

def ingest(csv_path=CSV_PATH, db_path=DB_PATH):
    create_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jid = int(row["id"])
            title = row["title"]
            company = row["company"]
            description = row["description"]
            emb = get_embedding(description).tolist()
            cur.execute(
                "INSERT OR REPLACE INTO jobs (id, title, company, description, embedding) VALUES (?, ?, ?, ?, ?)",
                (jid, title, company, description, json.dumps(emb))
            )
    conn.commit()
    conn.close()
    print(f"Ingested jobs from {csv_path} into {db_path}")
            

if __name__ == "__main__":
    ingest()