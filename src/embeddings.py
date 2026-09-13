"""
embeddings.py
-------------
Single place that produces the sentence embeddings used by intent
clustering (STEP 2), the embedding classifier (STEP 3), and FAISS
retrieval (STEP 4), so all three stages are guaranteed to use the same
representation.

Backend selection:
  1. Preferred: `sentence-transformers` (e.g. all-MiniLM-L6-v2). This is
     the intended production backend and is what `requirements.txt`
     installs.
  2. Fallback: TF-IDF -> TruncatedSVD (LSA) "pseudo-embeddings". Used
     automatically if `sentence-transformers` (or the model weights,
     which require network access) is unavailable, e.g. in a fully
     offline sandbox. This keeps the *pipeline* runnable end-to-end;
     it is NOT a claim that LSA vectors match transformer embedding
     quality -- that gap is called out explicitly in
     reports/report.md ("Misleading headline metric").

Every caller goes through `get_embedder(...)` and never imports
`sentence_transformers` directly, so this fallback logic lives in
exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


class Embedder(Protocol):
    backend_name: str

    def fit(self, texts: list[str]) -> "Embedder": ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class SentenceTransformerEmbedder:
    model_name: str
    backend_name: str = "sentence-transformers"
    _model: object = None
    _fallback: object = None

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        from sentence_transformers import SentenceTransformer

        try:
            # Do not trigger network retries during an offline run.  A model
            # already present in the local Hugging Face cache is still used.
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - model may not be cached
            logger.warning(
                "Sentence-transformer model '%s' is not available locally (%s). "
                "Using the TF-IDF+SVD offline fallback.",
                self.model_name,
                exc,
            )
            self._fallback = TfidfSvdEmbedder().fit(texts)
            self.backend_name = self._fallback.backend_name
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.encode(texts)
        return np.asarray(self._model.encode(texts, show_progress_bar=False))


@dataclass
class TfidfSvdEmbedder:
    """Offline fallback: TF-IDF followed by truncated SVD (i.e. LSA),
    producing dense vectors with the same interface as a real sentence
    embedder so downstream code (clustering/classification/FAISS) is
    unchanged."""

    n_components: int = 128
    backend_name: str = "tfidf_svd_fallback"
    _vectorizer: object = None
    _svd: object = None

    def fit(self, texts: list[str]) -> "TfidfSvdEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        tfidf = self._vectorizer.fit_transform(texts)
        n_components = min(self.n_components, max(2, min(tfidf.shape) - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(tfidf)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        tfidf = self._vectorizer.transform(texts)
        return self._svd.transform(tfidf)


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Factory that tries the real sentence-transformers backend first and
    transparently falls back to TF-IDF+SVD, logging clearly which backend
    is active (this matters for interpreting downstream metrics).
    """
    try:
        import sentence_transformers  # noqa: F401

        logger.info("Using sentence-transformers backend ('%s').", model_name)
        return SentenceTransformerEmbedder(model_name=model_name)
    except Exception as exc:  # noqa: BLE001 - includes network errors on model download
        logger.warning(
            "sentence-transformers unavailable or model download failed (%s). "
            "Falling back to TF-IDF+SVD pseudo-embeddings. Install "
            "`sentence-transformers` and ensure network access for production "
            "quality embeddings -- see requirements.txt / README.", exc,
        )
        return TfidfSvdEmbedder()
