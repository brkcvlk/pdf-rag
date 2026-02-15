from typing import Any, Dict, List, Tuple

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

LLM_MODEL = "llama3:8b-instruct-q4_0"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
TOP_K = 4


def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL)


def get_vector_store() -> QdrantVectorStore:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return QdrantVectorStore.from_existing_collection(
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        url=QDRANT_URL,
    )


def answer_question(
    question: str, vector_store: QdrantVectorStore, llm: ChatOllama
) -> Tuple[str, List[Dict[str, Any]]]:
    docs = vector_store.similarity_search(question, k=TOP_K)
    if not docs:
        return "No relevant context found in the documents.", []

    context = "\n\n".join(
        [f"Source {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context is insufficient, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    answer = llm.invoke(prompt)
    sources = [
        {"source": doc.metadata.get("source"), "page": doc.metadata.get("page")}
        for doc in docs
    ]
    return answer.content, sources, context
