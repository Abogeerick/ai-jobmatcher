from sentence_transformers import SentenceTransformer
import spacy
import numpy as p
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3, json, os

# Load models once when the file is imported
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print("Loading SentenceTransformer model in utils...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model ready, dim =", embedding_model.get_sentence_embedding_dimension())

print("Loading spaCy English model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy model ready.")

def get_embedding(text: str):
    """
    Convert text into embedding vector and return numpy
    """
    return embedding_model.encode(text)

def extract_skills(text: str):
    """
    Very simple skill extraction: looks for nouns/proper nouns.
    """
    doc = nlp(text)
    skills = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        if token.ent_type_ == "PERSON":
            continue
        if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
            skills.append(token.lemma_.lower())
    return sorted(set(skills))

def top_k_matches(query_emb, job_embs, k=5):
    """
    job_embs: list of dicts: {'id', 'title', 'company', 'description', 'embedding'(np.array)}
    Returns list of top-k matches with scores.
    """
    if not job_embs:
        return []
    X = np.array ([j["embeddings"] for j in job_embs])
    sims = cosine_similarity([query_emb], X)[0]
    idxs = sims.argsort()[::-1][:k]
    matches = []
    for i in idxs:
        j = job_embs[i]
        matches.append({
            "id": j['id'],
            "title": j['title'],
            "company": j['company'],
            "description": j['description'],
            "score": float(sims[i])
        })
    return matches

def load_jobs_from_sqlite(db_path = None):
     """Load jobs and embeddings from SQLite into memory as list of dicts."""
    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), "jobs.db")
        if not os.path.exists(db_path):
        print("Warning: jobs DB not found at", db_path)
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute("SELECT id, title, company, description, embedding FROM jobs").fetchall()
    job_embs = []
    for r in rows:
        emb = json.loads(r[4]) if r[4] else []
        job_embs.append({
            "id": r[0],
            "title": r[1],
            "company": r[2],
            "description": r[3],
            "embedding": np.array(emb)
        })
    conn.close()
    return job_embs



