from langchain_ollama import ChatOllama
from typing import Any, Dict, List
from pathlib import Path
from config import config
import mlflow
import json
import time

from rag import (
    EMBED_MODEL,
    LLM_MODEL,
    TOP_K,
    answer_question,
    get_llm,
    get_vector_store,
)

EVALSET_PATH = Path("example_evalset.json")
MLFLOW_EXPERIMENT = config["project"]["experiment_name"]
MLFLOW_TRACKING_DIR = Path("mlruns")
LLM_MODEL_JUDGE = config["model"]["judge"]

def get_llm_judge() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL_JUDGE, temperature=0.0)

def load_evalset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evalset file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("evalset.json must contain a JSON list.")

    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(f"Invalid evalset item at index {i}: expected {{'question': ...}}")
        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError(f"Invalid question at index {i}.")

    return data


def run_faithfulness_judge(llm: Any, context: str, answer: str) -> bool:
    judge_prompt = (
        "You are a faithfulness checker.\n"
        "Decide whether the answer is fully supported by the context.\n"
        "Reply with only one word: YES or NO.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n"
    )
    response = llm.invoke(judge_prompt)
    output = str(response.content).strip().upper()
    # if you want to debug judge outputs, uncomment the following lines to log them to a file
    # with open("judge_debug.txt", "a", encoding="utf-8") as f:
    #    f.write(f"[Context:\n{context}\n\nAnswer:\n{answer}\n\nJudge Output: {output}\n{'-'*80}\n]")
    return output.startswith("YES") if output else False


def main() -> None:
    evalset = load_evalset(EVALSET_PATH)
    vector_store = get_vector_store()
    llm_judge = get_llm_judge()
    llm = get_llm()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    latencies: List[float] = []
    faithful_count = 0
    total = len(evalset)
    retrieved_count = 0

    with mlflow.start_run(run_name="eval"):
        mlflow.log_param("model_name", LLM_MODEL)
        mlflow.log_param("embedding_model", EMBED_MODEL)
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("judge_model_name", LLM_MODEL_JUDGE)

        for index, item in enumerate(evalset, start=1):
            question = item["question"]

            # Measure latency of answer generation
            started = time.perf_counter()
            answer, sources, context = answer_question(question, vector_store, llm)  # Get answer and context for faithfulness check
            latency = time.perf_counter() - started
            latencies.append(latency)
            
            retrieved_count += len(sources)

            # check faithfulness
            is_faithful = run_faithfulness_judge(llm_judge, context, answer)
            faithful_count += int(is_faithful)

            print(
                f"[{index}/{total}] faithful={is_faithful} latency={latency:.3f}s"
            )

        avg_latency = sum(latencies) / total if total else 0.0
        faithfulness_rate = faithful_count / total if total else 0.0
        avg_docs = retrieved_count / total if total else 0.0
        
        
        mlflow.log_metric("average_retrieved_docs", avg_docs)
        mlflow.log_metric("average_latency", avg_latency)
        mlflow.log_metric("faithfulness_rate", faithfulness_rate)

        # If you want to print the summary metrics, uncomment the following lines
        # print(f"total_questions={total}")
        # print(f"average_latency={avg_latency:.3f}s")
        # print(f"faithfulness_rate={faithfulness_rate:.3f}")

if __name__ == "__main__":
    main()
