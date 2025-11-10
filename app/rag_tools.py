import os
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from settings import DB_A_DIR, DB_B_DIR, EMBED_MODEL, CHAT_MODEL

def _load_vs(path: str):
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

WEB_PROMPT = PromptTemplate.from_template("""
Eres un asistente que responde preguntas usando información de búsquedas web.
Genera una respuesta clara y concisa basada en los resultados encontrados.
NO inventes información que no esté en los resultados.

Pregunta: {question}

Resultados de búsqueda:
{web_results}

Genera una respuesta coherente y al final incluye una sección "Referencias Web" con los enlaces relevantes.
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
    Realiza una búsqueda web, genera respuesta con LLM y agrega referencias.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        search = DuckDuckGoSearchResults(max_results=5)
        raw = search.run(query)

        # Extraer resultados estructurados
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
                    
                    # Contexto para el LLM
                    web_context.append(f"Título: {title_part}\nContenido: {snippet_part}")
                except:
                    continue

        if not results:
            return "(No se encontraron resultados en la web.)"

        # Generar respuesta con el LLM
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.3)
        prompt = WEB_PROMPT.format(
            question=query,
            web_results="\n\n".join(web_context[:5])
        )
        
        try:
            answer = llm.invoke(prompt).content
        except Exception as e:
            answer = f"Se encontraron resultados, pero hubo un error al procesarlos: {e}"

        # Agregar referencias web
        refs = "\n\n**Referencias Web:**\n"
        for i, res in enumerate(results[:5], 1):
            refs += f"[{i}] {res['title']}\n    🔗 {res['link']}\n"

        return f"{answer}\n{refs}"

    except Exception as e:
        return f"(Error al realizar la búsqueda web: {e})"

