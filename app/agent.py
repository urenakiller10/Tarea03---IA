from typing import Literal
from langchain_community.chat_models import ChatOllama

# memoria (funciona en todas las versiones)
try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    from langchain_community.chat_message_histories import ChatMessageHistory as ConversationBufferWindowMemory

from rag_tools import rag_a_tool, rag_b_tool, web_search_tool
from settings import CHAT_MODEL




PROFILE = """
Nombre: TicoRAG
Rol: Agente conversacional del curso de IA TEC.
Especialidad: Responder preguntas usando apuntes (base vectorial).
Estilo: Claro, preciso, con citas. Evita inventar.
Restricciones:
- Usa RAG (A o B) por defecto.
- NO uses WebSearch a menos que el usuario lo pida explícitamente.
- Memoria acotada a una ventana de mensajes.
"""

class Agent:
    def __init__(self, window_k: int = 6, model: str = CHAT_MODEL):
        self.llm = ChatOllama(model="llama3", temperature=0)
        self.memory = ConversationBufferWindowMemory(k=window_k, return_messages=True)

    def decide_and_answer(self, user_query: str, rag_mode: Literal["A", "B"] = "A", allow_web: bool = True) -> str:
        text_l = user_query.lower()
        wants_web = any(w in text_l for w in ["busca en la web", "buscar en la web", "web", "internet", "google"])

        if wants_web and not allow_web:
            result = "(La búsqueda web está deshabilitada actualmente. Actívala en las opciones para permitir búsquedas en línea.)"
        elif allow_web and wants_web:
            result = web_search_tool(user_query)
        else:
            result = rag_a_tool(user_query) if rag_mode == "A" else rag_b_tool(user_query)

        # guardar en memoria
        try:
            self.memory.save_context({"human": user_query}, {"ai": result})
        except AttributeError:
            if hasattr(self.memory, "chat_memory"):
                self.memory.chat_memory.add_message({"role": "human", "content": user_query})
                self.memory.chat_memory.add_message({"role": "ai", "content": result})

        return result



