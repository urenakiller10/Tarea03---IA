import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

# Solo RAG B
DB_DIR = os.path.join(BASE_DIR, "chroma_ragB")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo-0125"

# Configuracion RAG B (tokens/oraciones)
TOKENS_PER_CHUNK = 180
TOKENS_OVERLAP = 30

AGENT_MODE = "B"