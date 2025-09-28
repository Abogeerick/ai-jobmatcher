from sentence_transformers import SentenceTransformer
import spacy

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
    Convert text into embedding vector
    """
    return embedding_model.encode(text)

def extract_skills(text: str):
    """
    Very simple skill extraction: looks for nouns/proper nouns.
    Later you can make this smarter.
    """
    doc = nlp(text)
    skills = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return list(set(skills))

if __name__ == "__main__":
    sample_text = """
    Erick Aboge is a software developer skilled in Python, Django, React, and SQL.
    He worked at Optiven where he built ERP systems and automated HR workflows.
    """
    print("\n=== Testing get_embedding ===")
    emb = get_embedding(sample_text)
    print("Embedding vector length:", len(emb))
    print("First 5 numbers:", emb[:5])  
    
    print("\n=== Testing extract_skills ===")
    skills = extract_skills(sample_text)
    print("Extracted skills:", skills)