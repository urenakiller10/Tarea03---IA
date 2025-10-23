import os
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from settings import DB_A_DIR, DB_B_DIR, EMBED_MODEL, CHAT_MODEL

def _load_vs(path: str):
    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=path, embedding_function=emb)

def _format_citations(docs: List[Document]) -> str:
    cites = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page", "?")
        cites.append(f"[{i}] {src}, p.{page}")
    return "\n".join(cites) if cites else "—"

RAG_PROMPT = PromptTemplate.from_template("""
Eres un asistente que responde SOLO con información de los fragmentos recuperados.
Si no está en los fragmentos, di explícitamente que no aparece en los apuntes y no inventes.
Incluye una sección "Referencias" con archivo y página.
Pregunta: {question}

Fragmentos:
{context}

Respuesta:
""")

def _answer_with_retriever(question: str, retriever, model: str = CHAT_MODEL) -> str:
    docs = retriever.get_relevant_documents(question)
    context = "\n---\n".join([d.page_content for d in docs]) if docs else "(sin contexto)"
    cites = _format_citations(docs)
    llm = ChatOpenAI(model=model, temperature=0)
    resp = llm.invoke(RAG_PROMPT.format(question=question, context=context))
    return f"{resp.content}\n\nReferencias:\n{cites}"

def rag_a_tool(query: str, k: int = 4) -> str:
    vs = _load_vs(DB_A_DIR)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return _answer_with_retriever(query, retriever)

def rag_b_tool(query: str, k: int = 4) -> str:
    vs = _load_vs(DB_B_DIR)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return _answer_with_retriever(query, retriever)

def web_search_tool(query: str) -> str:
    # Deshabilitado por política; actívalo solo si el profe lo permite.
    return ("(Búsqueda web deshabilitada por defecto; habilítala explícitamente "
            "si tu enunciado lo permite y tienes un proveedor aprobado).")
