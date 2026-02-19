from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import run_rag

app = FastAPI()


class Query(BaseModel):
    Question: str


@app.post("/query")
def ask(q: Query):
    answer = run_rag(q.Question)
    return {"response": answer}


# run with `uvicorn server:app --reload --port 11435`
