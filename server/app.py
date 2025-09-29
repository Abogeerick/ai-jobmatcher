import os
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from parse_cv import extract_text
from utils import get_embedding, extract_skills, top_k_matches, load_jobs_from_sqlite


app = FastAPI(title="AI jobMatcher API") 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JOB_DB = os.path.join(os.path.dirname(__file__), "jobs.db")
print("Loading job embeddings from DB...")
job_embs = load_jobs_from_sqlite(JOB_DB)
print(f"Loaded {len(job_embs)} jobs into memory.")

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

@app.post("/match")
async def match(file: UploadFile = File(...), k: int = 5):
    """
    Accepts a CV file upload (pdf/docx/txt), extracts text, computes embedding,
    returns top-k job matches plus missing skill hints.
    """
    # save uploaded file to a temporary path
    ext = os.path.splitext(file.filename)[1].lower()
    tmp_path = f"tmp_upload{ext}"
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)

    try:
        text = extract_text(tmp_path)
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {e}")

    # run embedding in threadpool to avoid blocking event loop
    loop = asyncio.get_running_loop()
    query_emb = await loop.run_in_executor(None, get_embedding, text)

    # find top matches
    matches = top_k_matches(query_emb, job_embs, k=k)

    # compute skills & missing skills per match
    cv_skills = extract_skills(text)
    for m in matches:
        job_skills = extract_skills(m["description"])
        missing = sorted(set(job_skills) - set(cv_skills))
        m["missing_skills"] = missing

    # cleanup
    os.remove(tmp_path)
    return {"matches": matches, "cv_skills": cv_skills}