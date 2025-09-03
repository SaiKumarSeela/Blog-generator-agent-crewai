
# RAG Related
EMBEDDING_MODEL_RAG:str = 'sentence-transformers/all-MiniLM-L6-v2'
CHUNK_SIZE:int = 800
CHUNK_OVERLAP: int = 100

LLM_MODEL="gemini/gemini-1.5-flash"

# API Related
API_HOST="0.0.0.0"
API_PORT=8085
STREAMLIT_PORT=8501
API_BASE_URL = f"http://localhost:{API_PORT}"