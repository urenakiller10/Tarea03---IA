import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

# Solo RAG A
DB_DIR = os.path.join(BASE_DIR, "chroma_ragA")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo-0125"

# Configuracion RAG A (chunks fijos)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

AGENT_MODE = "A"