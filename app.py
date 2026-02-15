from rag import answer_question, get_llm, get_vector_store
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel

vector_store = None
llm = None

# Initialize resources on startup (on_event("startup") is deprecated)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, llm
    vector_store = get_vector_store()
    llm = get_llm()
    yield

app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    source: Optional[str]
    page: Optional[int]


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if vector_store is None or llm is None:
        raise HTTPException(status_code=500, detail="Server not initialized.")

    answer, sources, _ = answer_question(request.question, vector_store, llm)
    return QueryResponse(answer=answer, sources=[SourceItem(**s) for s in sources])
