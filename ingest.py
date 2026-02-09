import os
from pathlib import Path
from typing import List, Any

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import QdrantClient

DOCS_DIR = Path("docs")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"

client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)

# Load documents, with multi-doc support
def load_documents() -> List[Any]:
    if not DOCS_DIR.exists():
        os.makedirs(DOCS_DIR)

    if not any(DOCS_DIR.rglob("*.pdf")):
        raise RuntimeError(f"No PDF documents found under {DOCS_DIR}")
    
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    return loader.load()


def split_documents(docs: List[Any]) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def index_documents(chunks: List[Any]) -> None:
    if not chunks:
        raise RuntimeError("No chunks produced from documents.")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )


def main() -> None:
    docs = load_documents()
    chunks = split_documents(docs)
    index_documents(chunks)
    print(
        f"Indexed {len(chunks)} chunks from {len(docs)} pages into '{COLLECTION_NAME}' "
        f"using {EMBED_MODEL}."
    )


if __name__ == "__main__":
    main()
