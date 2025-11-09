from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

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

class SimpleMemory:
    """Memoria conversacional simple con ventana deslizante."""
    def __init__(self, window_k: int = 6):
        self.messages = []
        self.window_k = window_k
    
    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))
        self._trim()
    
    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content=content))
        self._trim()
    
    def _trim(self):
        """Mantiene solo los últimos window_k mensajes."""
        if len(self.messages) > self.window_k:
            self.messages = self.messages[-self.window_k:]
    
    def get_context(self) -> str:
        """Devuelve el contexto conversacional como string."""
        context = []
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                context.append(f"Usuario: {msg.content}")
            elif isinstance(msg, AIMessage):
                context.append(f"Asistente: {msg.content}")
        return "\n".join(context)

class Agent:
    def __init__(self, window_k: int = 6, model: str = CHAT_MODEL):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.memory = SimpleMemory(window_k=window_k)

    def decide_and_answer(self, user_query: str, rag_mode: Literal["A", "B"] = "A", allow_web: bool = True) -> str:
        text_l = user_query.lower()
        wants_web = any(w in text_l for w in ["busca en la web", "buscar en la web", "web", "internet", "google"])

        if wants_web and not allow_web:
            result = "(La búsqueda web está deshabilitada actualmente. Actívala en las opciones para permitir búsquedas en línea.)"
        elif allow_web and wants_web:
            result = web_search_tool(user_query)
        else:
            # Agregar contexto conversacional a la consulta
            context = self.memory.get_context()
            if context:
                enriched_query = f"Contexto previo:\n{context}\n\nPregunta actual: {user_query}"
            else:
                enriched_query = user_query
            
            result = rag_a_tool(enriched_query) if rag_mode == "A" else rag_b_tool(enriched_query)

        # Guardar en memoria
        self.memory.add_user_message(user_query)
        self.memory.add_ai_message(result)

        return result



