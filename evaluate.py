import math
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.fusion import hybrid_candidates
from utils.reranker import rerank
from utils.retrievers import DATA_DIR, BM25Retriever, DenseRetriever, load_corpus

RERANK_SAMPLE_SIZE = 50
SEED = 42


def ndcg_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    dcg = sum(
        relevant.get(doc_id, 0) / math.log2(rank + 2)
        for rank, doc_id in enumerate(predicted_ids[:k])
    )
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


queries = pd.read_parquet(DATA_DIR / "queries.parquet")
qrels_df = pd.read_parquet(DATA_DIR / "qrels.parquet")

qrels: dict[str, dict[str, int]] = defaultdict(dict)
for qid, cid, score in zip(
    qrels_df["query-id"], qrels_df["corpus-id"], qrels_df["score"]
):
    qrels[str(qid)][str(cid)] = int(score)

queries_with_qrels = queries[queries["_id"].astype(str).isin(qrels.keys())].copy()
sample = queries_with_qrels.sample(n=RERANK_SAMPLE_SIZE, random_state=SEED)
print(f"Evaluating on {len(sample)} queries (sampled from {len(queries_with_qrels)})")


bm25 = BM25Retriever()
dense = DenseRetriever()
corpus_by_id = load_corpus().set_index("_id")


def hybrid_topk(query: str, candidate_k: int = 50) -> list[str]:
    return [doc_id for doc_id, _ in hybrid_candidates(query, bm25, dense, candidate_k)]


def reranked_topk(query: str, candidate_k: int = 50, k: int = 10) -> list[str]:
    candidates = hybrid_topk(query, candidate_k)
    return [doc_id for doc_id, _ in rerank(query, candidates, corpus_by_id, k=k)]


def reranked_topk_denseonly(
    query: str, candidate_k: int = 50, k: int = 10
) -> list[str]:
    candidates = [doc_id for doc_id, _ in dense.search(query, candidate_k)]
    return [doc_id for doc_id, _ in rerank(query, candidates, corpus_by_id, k=k)]


results: dict[str, list[float]] = defaultdict(list)
for qid, text in tqdm(
    zip(sample["_id"].astype(str), sample["text"]), total=len(sample), desc="Evaluating"
):
    relevant = qrels[qid]

    results["BM25"].append(ndcg_at_k([d for d, _ in bm25.search(text, k=10)], relevant))
    results["Dense"].append(
        ndcg_at_k([d for d, _ in dense.search(text, k=10)], relevant)
    )
    results["hybrid (RRF)"].append(ndcg_at_k(hybrid_topk(query=text)[:10], relevant))
    results["hybrid + rerank"].append(ndcg_at_k(reranked_topk(query=text), relevant))
    results["dense + rerank"].append(
        ndcg_at_k(reranked_topk_denseonly(query=text), relevant)
    )

if __name__ == "__main__":
    print(f"\nNDCG@10 x100 on FiQA ({len(sample)} sampled test queries)")
    print("-" * 42)
    for method, scores in results.items():
        print(f"  {method:<22} {np.mean(scores) * 100:.2f}")
