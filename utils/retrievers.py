from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "fiqa"
BM25_DIR = ROOT / "indexes" / "bm25"
DENSE_DIR = ROOT / "indexes" / "dense"
EMBEDDING_MODEL = "znbang/bge:small-en-v1.5-f16"


def load_corpus() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "corpus.parquet")


class BM25Retriever:
    def __init__(self) -> None:
        if (BM25_DIR / "params.index.json").exists():
            self._retriever, self._doc_ids = self._load()
        else:
            self._retriever, self._doc_ids = self._build()

        print("Info: BM25Retriever ready.")

    def _load(self):
        print("Info: Loading BM25 index from disk...")
        doc_ids = (BM25_DIR / "doc_ids.txt").read_text().splitlines()
        retriever = bm25s.BM25.load(str(BM25_DIR))
        return retriever, doc_ids

    def _build(self):
        print("Info: BM25 index not found, building from corpus...")
        corpus = load_corpus()

        doc_ids = corpus["_id"].astype(str).to_list()
        doc_text = corpus["text"].to_list()

        tokens = bm25s.tokenize(doc_text, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(tokens)

        BM25_DIR.mkdir(parents=True, exist_ok=True)
        retriever.save(str(BM25_DIR))
        (BM25_DIR / "doc_ids.txt").write_text("\n".join(doc_ids))
        print("Info: BM25 index built and saved.")
        return retriever, doc_ids

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        query_tokens = bm25s.tokenize([query], stopwords="en")
        indices, scores = self._retriever.retrieve(query_tokens, k=k)

        return [
            (self._doc_ids[doc_idx], float(score))
            for doc_idx, score in zip(indices[0].tolist(), scores[0].tolist())
        ]


class DenseRetriever:
    def __init__(self):
        if (DENSE_DIR / "embeddings.npy").exists():
            raw, self._doc_ids = self._load()
        else:
            raw, self._doc_ids = self._build()

        self._norm_embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)

        print("Info: DenseRetriever ready.")

    def _load(self):
        print("Info: Loading retriever...")
        doc_ids = (DENSE_DIR / "doc_ids.txt").read_text().splitlines()
        return np.load(DENSE_DIR / "embeddings.npy"), doc_ids

    def _embed_batch(self, texts: list[str]):
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return np.array([d.embedding for d in response.data], dtype=np.float32)

    def _build_index(self, doc_texts: list[str], batch_size: int = 256) -> np.ndarray:
        chunks = []
        for i in tqdm(range(0, len(doc_texts), batch_size), desc="Embedding"):
            chunks.append(self._embed_batch(doc_texts[i : i + batch_size]))
        return np.vstack(chunks)

    def _build(self):
        print("Info: Dense index not found, building from corpus...")
        corpus = load_corpus()
        doc_texts = [t.strip() or "[empty document]" for t in corpus["text"].tolist()]
        doc_ids = corpus["_id"].astype(str).to_list()

        doc_embeddings = self._build_index(doc_texts, batch_size=256)
        DENSE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(DENSE_DIR / "embeddings.npy", doc_embeddings)
        (DENSE_DIR / "doc_ids.txt").write_text("\n".join(doc_ids))
        print("Info: Dense index built and saved.")
        return doc_embeddings, doc_ids

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        query_vec = self._embed_batch([query])[0]
        query_vec /= np.linalg.norm(query_vec)
        scores = self._norm_embeddings @ query_vec
        top_k = np.argsort(-scores)[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]
