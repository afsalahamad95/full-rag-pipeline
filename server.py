from fastapi import FastAPI
from pydantic import BaseModel
from rag_application import run_rag

app = FastAPI()


class Query(BaseModel):
    Question: str


@app.post("/query")
def ask(q: Query):
    answer = run_rag(q.Question)
    return {"response": answer}
