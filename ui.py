"""UI for the RAG pipeline"""

import streamlit as st
import requests

BACKEND_URL = "http://localhost:11435/query"

st.title("RAG pipeline")

query = st.text_input("Ask me something.")

if st.button("submit") and query:
    with st.spinner("Thinking..."):
        res = requests.post(BACKEND_URL, json={"Question": query}, timeout=30)
        st.markdown(res.json()["response"])
