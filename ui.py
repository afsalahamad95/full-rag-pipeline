import streamlit as st
import requests

BACKEND_URL = "http://localhost:11435/query"

st.title("RAG pipeline")

query = st.text_input("Ask me something.")

if st.button("submit") and query:
    with st.spinner("Thinking..."):
        res = requests.post(BACKEND_URL, json={"Question": query})
        st.markdown(res.json()["response"])
