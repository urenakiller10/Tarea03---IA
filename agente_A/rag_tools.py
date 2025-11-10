import os
import time
from typing import List, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from settings import DB_DIR, EMBED_MODEL, CHAT_MODEL

def _load_vs():
    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=emb)

def _format_citations(docs: List[Document]) -> str:
    cites = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page", "?")
        autor = d.metadata.get("autor", "")
        cites.append(f"[{i}] {src}, p.{page} {f'(Autor: {autor})' if autor else ''}")
    return "\n".join(cites) if cites else "---"

RAG_PROMPT = PromptTemplate.from_template("""
Eres un asistente que responde SOLO con informacion de los fragmentos recuperados.
Si no esta en los fragmentos, di explicitamente que no aparece en los apuntes y no inventes.
Incluye una seccion "Referencias" con archivo y pagina.
Pregunta: {question}

Fragmentos:
{context}

Respuesta:
""")

WEB_PROMPT = PromptTemplate.from_template("""
Eres un asistente que responde preguntas usando informacion de busquedas web.
Genera una respuesta clara y concisa basada en los resultados encontrados.
NO inventes informacion que no este en los resultados.

Pregunta: {question}

Resultados de busqueda:
{web_results}

Genera una respuesta coherente y al final incluye una seccion "Referencias Web" con los enlaces relevantes.
""")

def rag_tool(query: str, k: int = 4) -> Tuple[str, float, float, List[dict]]:
    """
    Herramienta RAG unica para este agente.
    Retorna: (respuesta, t_retrieval_ms, t_generation_ms, retrieved_docs)
    """
    vs = _load_vs()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    
    start_retrieval = time.time()
    try:
        docs = retriever.get_relevant_documents(query)
    except AttributeError:
        docs = retriever.invoke(query)
    t_retrieval_ms = (time.time() - start_retrieval) * 1000

    if not docs:
        return "(No se encontraron fragmentos relevantes en los apuntes.)", t_retrieval_ms, 0.0, []

    retrieved_docs = []
    for doc in docs[:3]:
        retrieved_docs.append({
            "file": doc.metadata.get("source", "desconocido"),
            "page": doc.metadata.get("page", "?"),
            "score": 0.0
        })

    context = "\n---\n".join([d.page_content for d in docs[:3]])
    cites = _format_citations(docs[:3])

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    prompt = RAG_PROMPT.format(question=query, context=context)

    start_generation = time.time()
    try:
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        answer = f"(Error al generar respuesta: {e})"
    t_generation_ms = (time.time() - start_generation) * 1000

    return f"{answer}\n\n**Referencias:**\n{cites}", t_retrieval_ms, t_generation_ms, retrieved_docs

def web_search_tool(query: str) -> Tuple[str, float, float, List[dict]]:
    """Busqueda web."""
    start_retrieval = time.time()
    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        search = DuckDuckGoSearchResults(max_results=5)
        raw = search.run(query)

        results = []
        web_context = []
        
        for item in raw.split("title:"):
            if "link:" in item and "snippet:" in item:
                try:
                    title_part = item.split("link:")[0].strip()
                    link_part = item.split("link:")[1].split(", snippet:")[0].strip()
                    snippet_part = item.split("snippet:")[-1].strip()
                    
                    results.append({
                        "title": title_part,
                        "link": link_part,
                        "snippet": snippet_part
                    })
                    
                    web_context.append(f"Titulo: {title_part}\nContenido: {snippet_part}")
                except:
                    continue
        
        t_retrieval_ms = (time.time() - start_retrieval) * 1000

        if not results:
            return "(No se encontraron resultados en la web.)", t_retrieval_ms, 0.0, []

        start_generation = time.time()
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.3)
        prompt = WEB_PROMPT.format(
            question=query,
            web_results="\n\n".join(web_context[:5])
        )
        
        try:
            answer = llm.invoke(prompt).content
        except Exception as e:
            answer = f"Se encontraron resultados, pero hubo un error: {e}"
        
        t_generation_ms = (time.time() - start_generation) * 1000

        refs = "\n\n**Referencias Web:**\n"
        for i, res in enumerate(results[:5], 1):
            refs += f"[{i}] {res['title']}\n    Link: {res['link']}\n"

        retrieved_docs = [{"file": "web", "page": 0, "score": 0.0} for _ in results[:5]]

        return f"{answer}\n{refs}", t_retrieval_ms, t_generation_ms, retrieved_docs

    except Exception as e:
        return f"(Error al realizar la busqueda web: {e})", 0.0, 0.0, []