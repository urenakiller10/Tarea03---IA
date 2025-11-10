from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from rag_tools import rag_tool, web_search_tool
from settings import CHAT_MODEL, AGENT_MODE
from metrics import MetricsCollector

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
        if len(self.messages) > self.window_k:
            self.messages = self.messages[-self.window_k:]
    
    def get_context(self) -> str:
        context = []
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                context.append(f"Usuario: {msg.content}")
            elif isinstance(msg, AIMessage):
                context.append(f"Asistente: {msg.content}")
        return "\n".join(context)

class Agent:
    def __init__(self, window_k: int = 6, model: str = CHAT_MODEL, collect_metrics: bool = False):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.memory = SimpleMemory(window_k=window_k)
        self.collect_metrics = collect_metrics
        self.metrics_collector = MetricsCollector() if collect_metrics else None
        self.question_counter = 0
        self.agent_mode = AGENT_MODE

    def decide_and_answer(self, user_query: str, allow_web: bool = True) -> str:
        """Responde usando RAG o web segun corresponda."""
        self.question_counter += 1
        text_l = user_query.lower()
        wants_web = any(w in text_l for w in ["busca en la web", "buscar en la web", "web", "internet", "google"])

        web_used = False
        t_retrieval_ms = 0.0
        t_generation_ms = 0.0
        retrieved_docs = []
        result = ""

        if wants_web and not allow_web:
            result = "(La busqueda web esta deshabilitada actualmente.)"
        elif allow_web and wants_web:
            web_used = True
            result, t_retrieval_ms, t_generation_ms, retrieved_docs = web_search_tool(user_query)
        else:
            context = self.memory.get_context()
            enriched_query = f"Contexto previo:\n{context}\n\nPregunta actual: {user_query}" if context else user_query
            result, t_retrieval_ms, t_generation_ms, retrieved_docs = rag_tool(enriched_query)

        self.memory.add_user_message(user_query)
        self.memory.add_ai_message(result)

        if self.collect_metrics and self.metrics_collector:
            tokens_in = self.metrics_collector.count_tokens(user_query)
            tokens_out = self.metrics_collector.count_tokens(result)
            
            self.metrics_collector.add_metric(
                agent_mode=self.agent_mode,
                question_id=self.question_counter,
                question_text=user_query,
                web_allowed=allow_web,
                web_used=web_used,
                t_retrieval_ms=t_retrieval_ms,
                t_generation_ms=t_generation_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                retrieved_docs=retrieved_docs,
                answer=result
            )

        return result
    
    def save_metrics(self, json_path: str = "metrics.json", csv_path: str = "metrics.csv"):
        """Guarda las metricas recolectadas."""
        if self.metrics_collector:
            self.metrics_collector.save_to_json(json_path)
            self.metrics_collector.save_to_csv(csv_path)
            return self.metrics_collector.get_summary()
        return {}