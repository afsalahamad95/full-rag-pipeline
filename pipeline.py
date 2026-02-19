"""Implementation of the RAG pipeline phases - ingestion/query processing/generation"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter


def run_rag(query: str):
    """Main function to run the RAG pipeline"""
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    # ingestion phase
    kb_dir_path = "/Users/afsalahamada/Downloads/Data_as_Matrix/Text Files"
    dir_loader = DirectoryLoader(
        kb_dir_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    docs = dir_loader.load()

    # define the embedding method
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # embedding the documents to store them in our vector DB
    vector_store = Chroma(collection_name="rag", embedding_function=embeddings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store.add_documents(documents=chunks)

    results = vector_store.similarity_search(
        query="what are the traits of a good corpus?", k=1
    )
    for doc in results:
        print(doc)

    # retrieve data via a retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
    )

    # Query Processing Phase
    # user_query = input("Ask me anything, I'll answer if I know something about it.")
    results = retriever.invoke(query)

    context = ""
    for i, res in enumerate(results):
        print(f"doc {i} = {res}")
        context += res.page_content
        context += "\n---\n"

    # Generation Phase - Prompt engineering + context
    client = Groq(api_key=groq_api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond only from the given context, if you don't know, respond saying you don't know about the query.",
            },
            {"role": "user", "content": "Find the context below:"},
            {"role": "user", "content": context},
            {"role": "user", "content": query},
        ],
        model="groq/compound",  # free model from Groq, but with limited capabilities
        temperature=0.5,
        top_p=1,
    )

    return chat_completion.choices[0].message.content
