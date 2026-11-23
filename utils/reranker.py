import pandas as pd
from sentence_transformers import CrossEncoder

from utils.fusion import hybrid_candidates
from utils.retrievers import BM25Retriever, DenseRetriever

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

reranker = CrossEncoder(RERANKER_MODEL, max_length=512)


def rerank(
    query: str,
    candidate_ids: list[str],
    corpus_by_id: pd.DataFrame,
    k: int = 10,
) -> list[tuple[str, float]]:
    documents = [corpus_by_id.at[doc_id, "text"] for doc_id in candidate_ids]
    results = reranker.rank(query=query, documents=documents, top_k=k)
    return [(candidate_ids[res["corpus_id"]], float(res["score"])) for res in results]


def search_reranked(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    corpus_by_id: pd.DataFrame,
    k: int = 10,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:
    candidates = hybrid_candidates(query, bm25, dense, candidate_k)
    candidate_ids = [doc_id for doc_id, _ in candidates]
    return rerank(query, candidate_ids, corpus_by_id, k=k)
