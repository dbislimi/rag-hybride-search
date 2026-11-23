from collections import defaultdict

from utils.retrievers import BM25Retriever, DenseRetriever


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]], weights: dict[str, float], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for source, ranking in rankings.items():
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += weights[source] / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def hybrid_candidates(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    candidate_k: int = 50,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[tuple[str, float]]:
    bm25_ids = [doc_id for doc_id, _ in bm25.search(query, k=candidate_k)]
    dense_ids = [doc_id for doc_id, _ in dense.search(query, k=candidate_k)]
    return reciprocal_rank_fusion(
        rankings={"bm25":bm25_ids, "dense":dense_ids},
        weights={"bm25": bm25_weight, "dense": dense_weight}
    )[:candidate_k]
