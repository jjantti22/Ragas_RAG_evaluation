import os
import sys
import types
import asyncio
from datetime import datetime
from pathlib import Path

# Workaround for ragas 0.4.x unconditional import of removed VertexAI module
_dummy_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
_dummy_vertex.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = _dummy_vertex

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LCOpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig
from ragas_rag_evaluation.rag import LiteratureExplorerRAG

load_dotenv()


def get_lehmus_models():
    api_key = os.getenv("LEHMUS_API_KEY")
    base_url = os.getenv("LEHMUS_BASE_URL")

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=os.getenv("LEHMUS_LARGE_MODEL"),
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        LCOpenAIEmbeddings(
            model=os.getenv("LEHMUS_EMBED_MODEL"),
            api_key=api_key,
            base_url=base_url,
            check_embedding_ctx_length=False,
        )
    )
    return judge_llm, embeddings


async def collect_responses(df: pd.DataFrame) -> list[dict]:
    rag = LiteratureExplorerRAG(max_concurrency=2)
    rows = []
    for _, row in df.iterrows():
        res = await rag.query(row["question"], use_cache=True)
        rows.append({
            "question": row["question"],
            "question_type": row["question_type"],
            "expected_answer": row["expected_answer"],
            "source_document": row["source_document"],
            "answer": res["answer"],
            "retrieved_contexts": [d["content"] for d in res["retrieved_documents"]],
            "retrieved_sources": [d["source"] for d in res["retrieved_documents"]],
            "chat_latency": res["chat_latency"],
            "context_latency": res["context_latency"],
        })
    return rows


def main():
    dataset_path = Path("evals/datasets/test_dataset.csv")
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} test questions from {dataset_path}")

    records = asyncio.run(collect_responses(df))

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["retrieved_contexts"],
            response=r["answer"],
            reference=str(r["expected_answer"]),
        )
        for r in records
    ]

    eval_dataset = EvaluationDataset(samples=samples)
    judge_llm, embeddings = get_lehmus_models()

    metrics = [
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
    ]

    print("Running Ragas evaluation via Lehmus AI (max_workers=4)...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        run_config=RunConfig(max_workers=4, timeout=180),
    )

    scores_df = result.to_pandas()
    meta_df = pd.DataFrame(records)[
        ["question_type", "source_document", "retrieved_sources", "chat_latency", "context_latency"]
    ]
    full_df = pd.concat([meta_df, scores_df], axis=1)

    exp_dir = Path("evals/experiments")
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_file = exp_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_eval_results.csv"
    full_df.to_csv(out_file, index=False)
    print(f"\nSaved detailed results to {out_file}")

    metric_cols = [c for c in scores_df.columns if c not in {"user_input", "retrieved_contexts", "response", "reference"}]
    print("\n=== Mean Ragas Scores by Question Type ===")
    print(full_df.groupby("question_type")[metric_cols].mean().round(3))

    answerable = full_df[full_df["question_type"] != "unanswerable"]
    print("\n=== Overall Mean (Answerable Questions Only) ===")
    print(answerable[metric_cols].mean().round(3))


if __name__ == "__main__":
    main()