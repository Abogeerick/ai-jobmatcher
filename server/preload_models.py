from sentence_transformers import SentenceTransformer
import spacy
import subprocess, sys

print("Downloading sentence-transformers model (may take ~1-3 minutes)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model downloaded, embedding vector size:", model.get_sentence_embedding_dimension())

print("Downloading spaCy small English model...")
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
print("spaCy model downloaded.")