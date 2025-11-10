import os, re, glob, sys
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

from unidecode import unidecode
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from settings import DATA_DIR, DB_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def limpiar_texto(txt: str) -> str:
    txt = unidecode(txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def cargar_docs():
    pdfs = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdfs:
        raise SystemExit(f"No hay PDFs en {DATA_DIR}")
    docs = []
    for pdf_path in pdfs:
        loader = PyPDFLoader(pdf_path)
        for d in loader.load():
            d.page_content = limpiar_texto(d.page_content)
            d.metadata = {
                "source": os.path.basename(pdf_path),
                "page": d.metadata.get("page", None),
                "autor": "Estudiante",
            }
            docs.append(d)
    return docs

def main():
    print(f"DATA_DIR = {DATA_DIR}")
    docs = cargar_docs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunks creados: {len(chunks)}")

    emb = OpenAIEmbeddings(model=EMBED_MODEL)

    if os.path.exists(DB_DIR):
        print(f"Limpiando indice: {DB_DIR}")
    Chroma.from_documents(chunks, embedding=emb, persist_directory=DB_DIR).persist()
    print(f"Indice A listo en {DB_DIR}")

if __name__ == "__main__":
    main()