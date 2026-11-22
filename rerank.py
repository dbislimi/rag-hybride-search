from openai import OpenAI
from sentence_transformers import CrossEncoder

from utils.fusion import hybrid_candidates
from utils.retrievers import BM25Retriever, DenseRetriever, load_corpus

client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

bm25 = BM25Retriever()
dense = DenseRetriever()
corpus_by_id = load_corpus().set_index("_id")


def search_reranked(
    query: str, k: int = 10, candidate_k: int = 50
) -> list[tuple[str, float]]:
    candidates = hybrid_candidates(query, bm25, dense, candidate_k)
    candidate_ids = [doc_id for doc_id, _ in candidates]
    candidate_texts = [corpus_by_id.at[doc_id, "text"] for doc_id in candidate_ids]

    results = reranker.rank(query=query, documents=candidate_texts, top_k=k)
    return [(candidate_ids[res["corpus_id"]], float(res["score"])) for res in results]


def show(label: str, results: list[tuple[str, float]]):
    print(f"\n{label}")
    for i, (doc_id, score) in enumerate(results[:5], 1):
        text = corpus_by_id.at[doc_id, "text"]
        print(f"  {i}. [{score:.4f}] {doc_id}  {text[:70]}")


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"Query: {query}")

    show("Hybrid (RRF) only", hybrid_candidates(query, bm25, dense, candidate_k=50)[:5])
    show("Hybrid + rerank", search_reranked(query, k=5))
