# pdf-rag

## Overview
PDF-based RAG pipeline using Qdrant. PDFs in `docs/` are ingested via `ingest.py` (chunking + embedding + indexing). `app.py` serves a FastAPI retrieval and QA API, and `eval.py` evaluates responses with an LLM judge.

## Project Structure
```text
pdf-rag/
|-- docs/
|   |-- *.pdf
|-- ingest.py
|-- app.py
|-- rag.py
|-- eval.py
|-- config.py
|-- config.yaml
|-- docker-compose.yml
|-- Dockerfile
|-- requirements.txt
|-- .github/
    |-- workflows/
        |-- import-check.yml
```

- `docs/`: Input PDF corpus for ingestion.
- `ingest.py`: Loads PDFs, splits text into chunks, computes embeddings, and indexes to Qdrant.
- `app.py`: FastAPI application exposing the query endpoint.
- `rag.py`: Retrieval + prompt/answer logic.
- `eval.py`: Evaluation runner with an LLM-based judge and metric logging.
- `config.py`: Loads runtime configuration.
- `config.yaml`: Model, retrieval, chunking, and backend configuration values.
- `docker-compose.yml`: Local service definitions (Qdrant, Ollama, MLflow, FastAPI container).
- `Dockerfile`: Container image build for the API service.

## Quickstart
1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Put your PDFs under `docs/`.

```bash
# Example
cp /path/to/your/*.pdf docs/
```

3. Start required services.

```bash
docker compose up -d qdrant
```
4. Pull required Ollama models (check model names in config.yaml).
```bash
# Example format
ollama pull <model_name_from_config.yaml>
```

5. Run ingestion.

```bash
python ingest.py
```

6. Start the API.

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Optional** : To run the full stack via containers (API + services): 
```bash
docker compose up --build
```

## Notes
- If running `eval.py`, update `EVALSET_PATH` to match your evaluation dataset path.
- LLM-judge scores may be inconsistent across runs.
- Re-run ingestion whenever files in `docs/` are added, removed, or changed.
- Default models (`llm: gemma3:4b`, `judge: phi3:mini`) are intentionally small to avoid RAM issues; they can be replaced with larger models via `config.yaml`.

