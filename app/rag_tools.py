import os
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # ← CAMBIO
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from settings import DB_A_DIR, DB_B_DIR, EMBED_MODEL, CHAT_MODEL

def _load_vs(path: str):
    # ← CAMBIO: usar OpenAI embeddings
    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=path, embedding_function=emb)

def _format_citations(docs: List[Document]) -> str:
    cites = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page", "?")
        autor = d.metadata.get("autor", "")
        cites.append(f"[{i}] {src}, p.{page} {f'(Autor: {autor})' if autor else ''}")
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
    try:
        docs = retriever.get_relevant_documents(question)
    except AttributeError:
        docs = retriever.invoke(question)

    if not docs:
        return "(No se encontraron fragmentos relevantes en los apuntes.)"

    context = "\n---\n".join([d.page_content for d in docs[:3]])
    cites = _format_citations(docs[:3])

    # ← CAMBIO: usar ChatOpenAI
    llm = ChatOpenAI(model=model, temperature=0)

    prompt = RAG_PROMPT.format(question=question, context=context)

    try:
        answer = llm.invoke(prompt).content
    except Exception as e:
        answer = f"(Error al generar respuesta: {e})"

    return f"{answer}\n\n**Referencias:**\n{cites}"


def rag_a_tool(query: str, k: int = 4) -> str:
    vs = _load_vs(DB_A_DIR)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return _answer_with_retriever(query, retriever)

def rag_b_tool(query: str, k: int = 4) -> str:
    vs = _load_vs(DB_B_DIR)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return _answer_with_retriever(query, retriever)

def web_search_tool(query: str) -> str:
    """
    Realiza una búsqueda web y devuelve resultados formateados.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        search = DuckDuckGoSearchResults(max_results=5)
        raw = search.run(query)

        results = []
        for item in raw.split("title:"):
            if "link:" in item:
                title_part = item.split("link:")[0].strip()
                link_part = item.split("link:")[1].split(", snippet:")[0].strip()
                snippet_part = item.split("snippet:")[-1].strip()
                results.append(f"🔹 **{title_part}**\n{snippet_part}\n🔗 {link_part}\n")

        if not results:
            return "(No se encontraron resultados legibles en la web.)"

        formatted = "\n".join(results[:5])
        return f"**Resultados web (resumen):**\n\n{formatted}"

    except Exception as e:
        return f"(Error al realizar la búsqueda web: {e})"

