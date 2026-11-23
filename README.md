# RAG Hybrid Search

A work-in-progress RAG project aiming to answer questions over a document corpus using hybrid search and verifiable sources.

The goal is to combine keyword and semantic retrieval, rerank relevant passages, and generate answers with citations that can be checked against their sources. The project is at an early stage: the current code explores the retrieval pipeline, while ingestion, answer generation, citation verification and an API are planned.

## Project direction

```text
Documents → chunking → lexical and vector indexes
Question  → hybrid retrieval → RRF → reranking → relevant passages
          → answer generation → citation verification → answer and sources
```

Current experiments use local indexes and models, so the final stack may evolve.

## Current progress

The first step explores BM25, dense retrieval, weighted Reciprocal Rank Fusion (RRF) and cross-encoder reranking. A small evaluation on the **FiQA** dataset helps compare these approaches during development; it is a personal experiment, not the final project evaluation.

## Current retrieval pipeline

1. Download the FiQA corpus, queries and relevance judgments from BEIR.
2. Build a BM25 index and dense document embeddings, cached locally for subsequent runs.
3. Retrieve candidates with BM25 and cosine similarity, then combine their rankings using weighted RRF.
4. Optionally rerank the candidates with a cross-encoder.
5. Compare methods using mean NDCG@10 on a reproducible sample of test queries.

The evaluation compares BM25, dense retrieval, dense retrieval with reranking, and three hybrid variants (balanced, BM25-weighted and dense-weighted), each with and without reranking. Answer generation and citation verification are still to come.

## Current stack

- **Python 3.11+** and [uv](https://docs.astral.sh/uv/)
- **BM25** with `bm25s`
- **Embeddings** with `znbang/bge:small-en-v1.5-f16` served by Ollama
- **Reranking** with `BAAI/bge-reranker-v2-m3` and Sentence Transformers
- **Dataset**: [BEIR FiQA](https://huggingface.co/datasets/BeIR/fiqa)

## Getting started

### Prerequisites

Install Python 3.11 or newer, uv and [Ollama](https://ollama.com/). Start the Ollama server and pull the embedding model:

```bash
ollama serve
```

In another terminal:

```bash
ollama pull znbang/bge:small-en-v1.5-f16
```

The application connects to Ollama at `http://localhost:11434/v1/`. The first run also downloads the dataset and reranker model from Hugging Face and builds the indexes, which can take time and disk space.

### Installation and running the experiment

```bash
git clone https://github.com/dbislimi/rag-hybride-search.git
cd rag-hybride-search
uv sync
uv run python data/download_beir.py
uv run python evaluate.py
```

The script prints mean NDCG@10 multiplied by 100 for each method and records the configuration and scores in `output/evaluation_history.json`. Indexes are stored in `indexes/` and reused on later runs. If the corpus or embedding model changes, remove the relevant cached index before evaluating again.

## Configuration

Edit `EvalConfig` in `evaluate.py` to change the sample size, random seed, number of candidates or evaluation cutoff. Defaults are 50 queries, seed 42, 50 candidates and a cutoff of 10. Hybrid weights are defined in `HYBRID_VARIANTS`; model names are defined in `utils/retrievers.py` and `utils/reranker.py`.

## Project structure

```text
├── data/download_beir.py  # Download and save FiQA as Parquet
├── evaluate.py            # Run and record the benchmark
├── utils/
│   ├── retrievers.py      # BM25 and dense retrieval
│   ├── fusion.py          # Weighted Reciprocal Rank Fusion
│   ├── reranker.py        # Cross-encoder reranking
│   └── eval_model.py      # Evaluation history models
├── pyproject.toml
└── uv.lock
```
