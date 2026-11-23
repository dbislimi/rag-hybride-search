from collections import defaultdict

from utils.retrievers import BM25Retriever, DenseRetriever


def hybrid_candidates(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    candidate_k: int = 50,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    bm25_ids = [doc_id for doc_id, _ in bm25.search(query, k=candidate_k)]
    dense_ids = [doc_id for doc_id, _ in dense.search(query, k=candidate_k)]
    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in ((bm25_ids, bm25_weight), (dense_ids, dense_weight)):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += weight / (rrf_k + rank)

    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:candidate_k]
