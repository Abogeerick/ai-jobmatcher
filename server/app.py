from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
import os
from sentence_transformers import SentenceTransformer

app = FastAPI(title="AI jobMatcher API") 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print("Loading SentenceTransformer model in app... (this may take a moment on first run)")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded (in app): embedding dim =", model.get_sentence_embedding_dimension())

@app.get("/health")
async def health():
    """
    Health check to confirm server is running
    """
    return{"status":"ok"}

@app.get("/embed_test")
async def embed_test():
    """
    Embedding test to cofirm if sentences are turned to embeddings
    """
    s = "Software engineer with Python and React experience"
    emb = model.encode(s)
    return {"text": s, "embedding_len": len(emb)}

