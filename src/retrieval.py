"""
retrieval.py
------------
STEP 4 (retrieval half): build a FAISS index over historical customer
messages and retrieve the k most similar past conversations (with their
brand replies) to ground reply generation.

Why FAISS over a naive cosine-similarity loop or BM25:
see reports/decision_log.md, item 3. Short version: FAISS gives us
sub-linear approximate search that scales to the full ~1M-conversation
dataset, while still being exact (IndexFlatIP) at the sample sizes used
here, so there is no accuracy/speed tradeoff being made prematurely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

try:
    import faiss
    _HAS_FAISS = True
except ImportError:  # pragma: no cover - exercised only in minimal envs
    _HAS_FAISS = False
    logger.warning("faiss not importable; falling back to a pure-numpy exact "
                    "nearest-neighbor search with identical semantics to "
                    "IndexFlatIP. Install `faiss-cpu` for production use.")


@dataclass
class RetrievalIndex:
    embeddings: np.ndarray          # L2-normalized, shape (n, d)
    corpus_df: pd.DataFrame         # aligned 1:1 with `embeddings` rows
    index: "faiss.Index | None"     # None when faiss isn't available


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return embeddings / norms


def build_index(embeddings: np.ndarray, corpus_df: pd.DataFrame) -> RetrievalIndex:
    """Build a cosine-similarity FAISS index (inner product over
    L2-normalized vectors == cosine similarity)."""
    embeddings = _normalize(embeddings.astype("float32"))
    index = None
    if _HAS_FAISS:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        logger.info("Built FAISS IndexFlatIP with %d vectors (dim=%d)",
                    embeddings.shape[0], embeddings.shape[1])
    return RetrievalIndex(embeddings=embeddings, corpus_df=corpus_df.reset_index(drop=True), index=index)


def search(query_embedding: np.ndarray, retrieval_index: RetrievalIndex, top_k: int = 5) -> pd.DataFrame:
    """Return the top_k most similar historical conversation pairs, with a
    similarity score column, for use as RAG context."""
    query_embedding = _normalize(query_embedding.reshape(1, -1).astype("float32"))

    if _HAS_FAISS and retrieval_index.index is not None:
        scores, idxs = retrieval_index.index.search(query_embedding, top_k)
        scores, idxs = scores[0], idxs[0]
    else:
        sims = (retrieval_index.embeddings @ query_embedding.T).ravel()
        idxs = np.argsort(sims)[::-1][:top_k]
        scores = sims[idxs]

    results = retrieval_index.corpus_df.iloc[idxs].copy()
    results["similarity"] = scores
    return results.reset_index(drop=True)
