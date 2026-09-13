"""
utils.py
--------
Shared, cross-cutting utilities used by every stage of the pipeline:
- reproducible logging setup
- global seeding (numpy / random / torch if present)
- lightweight config object (avoids a hard dependency on hydra/yaml for a
  project this size, while still keeping all "magic numbers" in one place)
- small I/O helpers used by more than one module

Keeping this file free of business logic makes every other module easier to
unit test in isolation.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent, timestamped format.

    Using a factory function (instead of `logging.basicConfig` at import
    time) avoids duplicate handlers when modules are re-imported inside
    notebooks, and keeps each module's logger independently silence-able.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # guard against double-adding handlers
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_global_seed(seed: int = 42) -> None:
    """Seed every RNG we might touch so results are reproducible.

    Note: sentence-transformers / torch determinism on GPU is not
    bit-exact even with seeding; this is best-effort reproducibility,
    which is standard practice and stated explicitly in the README's
    limitations section rather than overclaimed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # optional dependency

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    """Single source of truth for pipeline parameters.

    A dataclass (rather than scattering constants across files) makes every
    tunable parameter discoverable in one place and trivially serializable
    for the run manifest we write alongside outputs (see `save_run_manifest`).
    """

    # Data
    raw_data_path: str = "data/twcs.csv"
    sample_data_path: str = "data/sample_twcs.csv"
    brand_author_id: str = "AppleSupport"
    min_conversation_turns: int = 2
    random_seed: int = 42

    # Intent discovery / classification
    n_intent_clusters: int = 8
    embedding_model_name: str = "all-MiniLM-L6-v2"
    tfidf_max_features: int = 5000
    test_size: float = 0.2

    # Retrieval / RAG
    faiss_top_k: int = 5
    llm_model_name: str = "gpt-4o-mini"
    use_live_llm: bool = False  # False => deterministic template fallback

    # Golden eval set
    golden_sample_size: int = 200

    # Paths
    outputs_dir: str = "outputs"
    reports_dir: str = "reports"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_run_manifest(config: Config, out_path: str | Path) -> None:
    """Persist the exact config used for a run, for reproducibility/audit."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config.as_dict(), f, indent=2)


# --------------------------------------------------------------------------- #
# Small shared I/O helpers
# --------------------------------------------------------------------------- #

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
