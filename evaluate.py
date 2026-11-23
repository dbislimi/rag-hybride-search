import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import TypeAdapter
from tqdm import tqdm

from utils.eval_model import (
    ConfigEvals,
    EvalHistory,
    Evaluation,
    MethodEvals,
)
from utils.fusion import hybrid_candidates
from utils.reranker import RERANKER_MODEL, rerank
from utils.retrievers import (
    DATA_DIR,
    EMBEDDING_MODEL,
    BM25Retriever,
    DenseRetriever,
    load_corpus,
)

HISTORY_PATH = Path(__file__).resolve().parent / "output" / "evaluation_history.json"
HISTORY_ADAPTER = TypeAdapter(EvalHistory)

HYBRID_VARIANTS = {
    "hybrid balanced": {"bm25_weight": 1.0, "dense_weight": 1.0},
    "hybrid BM25": {"bm25_weight": 2.0, "dense_weight": 1.0},
    "hybrid dense": {"bm25_weight": 1.0, "dense_weight": 2.0},
}


@dataclass(frozen=True)
class EvalConfig:
    sample_size: int = 50
    seed: int = 42
    candidate_k: int = 50
    k: int = 10


def ndcg_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    dcg = sum(
        relevant.get(doc_id, 0) / math.log2(rank + 2)
        for rank, doc_id in enumerate(predicted_ids[:k])
    )
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def load_sample(data_dir: Path, size: int, seed: int):
    queries = pd.read_parquet(data_dir / "queries.parquet")
    qrels_df = pd.read_parquet(data_dir / "qrels.parquet")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)

    for qid, cid, score in zip(
        qrels_df["query-id"], qrels_df["corpus-id"], qrels_df["score"]
    ):
        qrels[str(qid)][str(cid)] = int(score)

    eligible = queries[queries["_id"].astype(str).isin(qrels)].copy()
    sample = eligible.sample(n=size, random_state=seed)
    print(f"Evaluating on {len(sample)} queries (sampled from {len(eligible)})")
    return sample, qrels


def evaluate(sample, qrels, bm25, dense, corpus_by_id, config: EvalConfig):
    results: dict[str, list[float]] = defaultdict(list)

    for qid, query in tqdm(
        zip(sample["_id"].astype(str), sample["text"]),
        total=len(sample),
        desc="Evaluating",
    ):
        relevant = qrels[qid]

        bm25_ids = [doc_id for doc_id, _ in bm25.search(query, k=config.candidate_k)]
        dense_ids = [doc_id for doc_id, _ in dense.search(query, k=config.candidate_k)]

        predictions = {
            "BM25": bm25_ids,
            "Dense": dense_ids,
            "dense + rerank": [
                doc_id for doc_id, _ in rerank(query, dense_ids, corpus_by_id, k=config.k)
            ],
        }

        for name, params in HYBRID_VARIANTS.items():
            hybrid_ids = [
                doc_id
                for doc_id, _ in hybrid_candidates(
                    query, bm25, dense, config.candidate_k, **params
                )
            ]

            predictions[name] = hybrid_ids
            predictions[f"{name} + rerank"] = [
                doc_id for doc_id, _ in rerank(query, hybrid_ids, corpus_by_id, k=config.k)
            ]

        for method, ids in predictions.items():
            results[method].append(ndcg_at_k(ids, relevant, k=config.k))

    return results


def load_history(path: Path) -> EvalHistory:
    if not path.exists():
        return []
    return HISTORY_ADAPTER.validate_json(path.read_bytes())


def record_results(
    history: EvalHistory, config: dict, parameters: dict, results: dict, k: int
):
    configuration = next((entry for entry in history if entry.config == config), None)
    if configuration is None:
        configuration = ConfigEvals(config=config)
        history.append(configuration)

    date = datetime.now(UTC).isoformat()
    for method, scores in results.items():
        method_parameters = parameters.get(method, {})
        variants = configuration.methods.setdefault(method, [])
        variant = next(
            (item for item in variants if item.parameters == method_parameters), None
        )
        if variant is None:
            variant = MethodEvals(parameters=method_parameters.copy())
            variants.append(variant)
        variant.evaluations.append(
            Evaluation(date=date, metrics={f"ndcg@{k}": float(np.mean(scores))})
        )


def save_history(path: Path, history: EvalHistory):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(HISTORY_ADAPTER.dump_json(history, indent=2) + b"\n")


def main():
    config = EvalConfig()
    history = load_history(HISTORY_PATH)
    sample, qrels = load_sample(DATA_DIR, config.sample_size, config.seed)

    results = evaluate(
        sample,
        qrels,
        BM25Retriever(),
        DenseRetriever(),
        load_corpus().set_index("_id"),
        config,
    )

    history_config = {
        **asdict(config),
        "dataset": "FiQA",
        "sample_size": len(sample),
        "query_ids": sample["_id"].astype(str).tolist(),
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
    }
    parameters = {
        name: params
        for variant, params in HYBRID_VARIANTS.items()
        for name in (variant, f"{variant} + rerank")
    }

    record_results(history, history_config, parameters, results, config.k)
    save_history(HISTORY_PATH, history)

    print(f"\nNDCG@{config.k} x100 on FiQA ({len(sample)} sampled test queries)")
    print("-" * 42)
    for method, scores in results.items():
        print(f"  {method:<22} {np.mean(scores) * 100:.2f}")
    print(f"History saved to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
