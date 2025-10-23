import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

DB_A_DIR = os.path.join(BASE_DIR, "chroma_ragA")  # RAG: chunks fijos
DB_B_DIR = os.path.join(BASE_DIR, "chroma_ragB")  # RAG: tokens/oraciones

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOKENS_PER_CHUNK = 180
TOKENS_OVERLAP = 30
