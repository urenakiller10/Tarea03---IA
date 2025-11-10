import os
import sys
from dotenv import load_dotenv
import streamlit as st

env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

from agent import Agent
from settings import AGENT_MODE

st.set_page_config(page_title=f"GPTEC - Agente {AGENT_MODE}", page_icon="A", layout="wide")
st.title(f"GPTEC - Agente {AGENT_MODE} (Chunks Fijos)")

if "agent" not in st.session_state:
    st.session_state.agent = Agent(window_k=6, collect_metrics=False)

with st.sidebar:
    st.header("Opciones")
    st.info(f"**Modo:** RAG {AGENT_MODE}\n**Estrategia:** Chunks fijos (800 caracteres)")
    
    allow_web = st.toggle("Permitir Busqueda Web", value=False)
    collect_metrics = st.toggle("Recolectar metricas", value=False)
    
    st.caption("La web solo se usa si el usuario lo solicita explicitamente.")
    
    if collect_metrics and not st.session_state.agent.collect_metrics:
        st.session_state.agent = Agent(window_k=6, collect_metrics=True)
        st.success("Metricas activadas")
    elif not collect_metrics and st.session_state.agent.collect_metrics:
        st.session_state.agent = Agent(window_k=6, collect_metrics=False)
        st.info("Metricas desactivadas")
    
    if st.button("Limpiar memoria"):
        st.session_state.agent = Agent(window_k=6, collect_metrics=collect_metrics)
        st.success("Memoria limpiada")
    
    if collect_metrics and st.session_state.agent.metrics_collector:
        if st.button("Guardar metricas"):
            summary = st.session_state.agent.save_metrics()
            st.success(f"Metricas guardadas para Agente {AGENT_MODE}")
            st.json(summary)

query = st.text_input("Pregunta (basada en los apuntes PDF):")

if st.button("Preguntar") and query.strip():
    with st.spinner("Consultando..."):
        answer = st.session_state.agent.decide_and_answer(query, allow_web=allow_web)
    st.markdown("### Respuesta")
    st.write(answer)

with st.expander("Ver memoria conversacional"):
    memory_context = st.session_state.agent.memory.get_context()
    if memory_context:
        st.text_area("Contexto:", memory_context, height=200, disabled=True)
        st.caption(f"Mensajes: {len(st.session_state.agent.memory.messages)}")
    else:
        st.info("Memoria vacia")

if collect_metrics and st.session_state.agent.metrics_collector:
    with st.expander("Metricas de la sesion"):
        summary = st.session_state.agent.metrics_collector.get_summary()
        if summary:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Preguntas", summary.get("total_questions", 0))
                st.metric("Fidelidad", f"{summary.get('fidelity_rate', 0):.2%}")
            with col2:
                st.metric("T. retrieval (ms)", f"{summary.get('avg_t_retrieval_ms', 0):.1f}")
                st.metric("T. generacion (ms)", f"{summary.get('avg_t_generation_ms', 0):.1f}")
            with col3:
                st.metric("Tokens in", f"{summary.get('avg_tokens_in', 0):.0f}")
                st.metric("Tokens out", f"{summary.get('avg_tokens_out', 0):.0f}")

st.divider()
st.caption(f"Agente {AGENT_MODE} | Chunks fijos de 800 caracteres con overlap de 120")