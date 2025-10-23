import os
from dotenv import load_dotenv
import streamlit as st
from app.agent import Agent

load_dotenv()

st.set_page_config(page_title="TicoRAG", page_icon="🤖", layout="wide")
st.title("TicoRAG – Agente (RAG A/B) con memoria")

with st.sidebar:
    st.header("Opciones")
    rag_mode = st.radio("Estrategia:", ["A (chunks fijos)", "B (tokens/oraciones)"])
    allow_web = st.toggle("Permitir Búsqueda Web (solo si la pregunta lo pide)", value=False)
    st.caption("La web solo se usa si el usuario lo solicita explícitamente.")

if "agent" not in st.session_state:
    st.session_state.agent = Agent(window_k=6)

query = st.text_input("Pregunta (basada en los apuntes PDF del folder /data):")
if st.button("Preguntar") and query.strip():
    mode = "A" if rag_mode.startswith("A") else "B"
    answer = st.session_state.agent.decide_and_answer(query, rag_mode=mode, allow_web=allow_web)
    st.markdown("### Respuesta")
    st.write(answer)

st.divider()
st.caption("Tip: cambia entre A/B y compara las referencias (archivo/página).")
