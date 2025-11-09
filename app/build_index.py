import os, re, glob
from dotenv import load_dotenv
from unidecode import unidecode
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # ← CAMBIO
from settings import DATA_DIR, DB_A_DIR, DB_B_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOKENS_PER_CHUNK, TOKENS_OVERLAP

def limpiar_texto(txt: str) -> str:
    txt = unidecode(txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def cargar_docs():
    pdfs = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdfs:
        raise SystemExit(f"No hay PDFs en {DATA_DIR}. Mete tus apuntes ahí.")
    docs = []
    for pdf_path in pdfs:
        loader = PyPDFLoader(pdf_path)
        for d in loader.load():
            d.page_content = limpiar_texto(d.page_content)
            d.metadata = {
                "source": os.path.basename(pdf_path),
                "page": d.metadata.get("page", None),
                "fecha": "2025-10-XX",  # Agregar si es posible extraer del nombre
                "autor": "Estudiante",   # Agregar si está disponible
            }
            docs.append(d)
    return docs

def main():
    load_dotenv()
    print(f"DATA_DIR = {DATA_DIR}")
    docs = cargar_docs()

    splitter_A = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks_A = splitter_A.split_documents(docs)
    print(f"[A] Chunks creados: {len(chunks_A)}")

    splitter_B = SentenceTransformersTokenTextSplitter(
        tokens_per_chunk=TOKENS_PER_CHUNK,
        chunk_overlap=TOKENS_OVERLAP,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    chunks_B = splitter_B.split_documents(docs)
    print(f"[B] Chunks creados: {len(chunks_B)}")

    # ← CAMBIO: usar OpenAI embeddings
    emb = OpenAIEmbeddings(model=EMBED_MODEL)

    if os.path.exists(DB_A_DIR):
        print(f"Limpiando índice A: {DB_A_DIR}")
    Chroma.from_documents(chunks_A, embedding=emb, persist_directory=DB_A_DIR).persist()
    print(f"Índice A listo en {DB_A_DIR}")

    if os.path.exists(DB_B_DIR):
        print(f"Limpiando índice B: {DB_B_DIR}")
    Chroma.from_documents(chunks_B, embedding=emb, persist_directory=DB_B_DIR).persist()
    print(f"Índice B listo en {DB_B_DIR}")

if __name__ == "__main__":
    main()
