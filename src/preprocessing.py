"""
preprocessing.py
-----------------
STEP 1 of the assignment: load the Twitter Customer Support dataset,
select one brand, and build clean (customer_message, agent_reply) pairs.

Design notes
============
- The raw Kaggle file (`twcs.csv`) is ~2.8M rows of *individual tweets*,
  inbound and outbound, linked by `response_tweet_id` /
  `in_response_to_tweet_id`. The first job of this module is to turn that
  into conversational *pairs*, which is the unit every downstream stage
  (classification, retrieval, generation) actually operates on.
- Brand selection is automatic and explainable: we rank candidate brand
  accounts by (a) conversation volume, (b) text cleanliness (ratio of
  non-empty, non-boilerplate messages), and (c) topical diversity
  (approximated via vocabulary richness), then log the ranking so the
  choice is auditable rather than hard-coded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class BrandCandidateScore:
    brand: str
    n_conversations: int
    cleanliness: float
    vocab_richness: float
    composite_score: float


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the raw Kaggle CSV (or the synthetic stand-in with the same schema).

    Falls back to the synthetic sample if the real file isn't present, so
    `run_pipeline.py` works out of the box. This fallback is logged loudly
    so it is never silently mistaken for real data.
    """
    path = Path(path)
    if not path.exists():
        sample_path = Path("data/sample_twcs.csv")
        logger.warning(
            "Real dataset not found at %s. Falling back to SYNTHETIC sample "
            "at %s. Download the real Kaggle dataset "
            "(thoughtvector/customer-support-on-twitter) and place it at "
            "%s to reproduce real results.",
            path, sample_path, path,
        )
        path = sample_path

    df = pd.read_csv(path, dtype={"tweet_id": "Int64",
                                   "response_tweet_id": "string",
                                   "in_response_to_tweet_id": "string"})
    logger.info("Loaded %d raw rows from %s", len(df), path)
    return df


def clean_text(text: str) -> str:
    """Normalize a single tweet's text for modeling.

    Removes URLs and @mentions (which are noise for intent/semantics but
    not for display -- see `clean_text_for_display`), collapses whitespace,
    and strips. Keeping this pure/stateless makes it trivially unit-testable.
    """
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_text_for_display(text: str) -> str:
    """Lighter cleaning that preserves @mentions (needed for realistic
    reply generation, since the brand's real replies always @-mention the
    customer)."""
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _score_brand_candidates(df: pd.DataFrame, min_conversations: int) -> list[BrandCandidateScore]:
    """Rank outbound-author candidates (i.e. brand support accounts) by
    volume, cleanliness, and topical diversity, to justify brand selection.
    """
    outbound = df[df["inbound"].astype(str).str.lower() == "false"]
    candidates = []
    for brand, group in outbound.groupby("author_id"):
        n_conv = len(group)
        if n_conv < min_conversations:
            continue
        cleaned = group["text"].astype(str).apply(clean_text)
        non_empty_ratio = (cleaned.str.len() > 5).mean()
        vocab = set(" ".join(cleaned).lower().split())
        richness = len(vocab) / max(1, cleaned.str.split().apply(len).sum())
        composite = 0.5 * min(n_conv / 500, 1.0) + 0.3 * non_empty_ratio + 0.2 * min(richness * 10, 1.0)
        candidates.append(BrandCandidateScore(brand, n_conv, non_empty_ratio, richness, composite))
    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    return candidates


def select_brand(df: pd.DataFrame, min_conversations: int = 20) -> str:
    """Automatically select a brand and log the reasoning (STEP 1 requirement)."""
    candidates = _score_brand_candidates(df, min_conversations)
    if not candidates:
        raise ValueError("No brand meets the minimum conversation threshold.")

    logger.info("Top brand candidates (volume / cleanliness / vocab richness / score):")
    for c in candidates[:5]:
        logger.info(
            "  %-20s conv=%-5d clean=%.2f richness=%.3f score=%.3f",
            c.brand, c.n_conversations, c.cleanliness, c.vocab_richness, c.composite_score,
        )

    best = candidates[0]
    logger.info(
        "Selected brand '%s': highest composite score (%.3f), driven by "
        "%d qualifying conversations and a %.0f%% clean-text ratio.",
        best.brand, best.composite_score, best.n_conversations, best.cleanliness * 100,
    )
    return best.brand


def build_conversation_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Join inbound customer tweets to the brand's first outbound reply.

    Returns one row per (customer_message, agent_reply) pair with cleaned
    text plus the original raw text (for display / RAG grounding).
    """
    df = df.copy()
    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"

    agent_msgs = df[(~df["inbound"]) & (df["author_id"] == brand)].set_index("tweet_id")
    customer_msgs = df[df["inbound"]]

    pairs = []
    for _, row in customer_msgs.iterrows():
        resp_id = row.get("response_tweet_id")
        if pd.isna(resp_id) or resp_id in ("", "nan"):
            continue
        # response_tweet_id can contain multiple comma-separated ids; take first
        first_resp_id = str(resp_id).split(",")[0].strip()
        if not first_resp_id.isdigit():
            continue
        first_resp_id = int(first_resp_id)
        if first_resp_id not in agent_msgs.index:
            continue
        agent_row = agent_msgs.loc[first_resp_id]

        cust_clean = clean_text(row["text"])
        agent_clean = clean_text(agent_row["text"])
        if len(cust_clean) < 3 or len(agent_clean) < 3:
            continue

        pairs.append({
            "customer_tweet_id": row["tweet_id"],
            "agent_tweet_id": first_resp_id,
            "customer_text_raw": row["text"],
            "agent_text_raw": agent_row["text"],
            "customer_text": cust_clean,
            "agent_text": agent_clean,
            "customer_text_display": clean_text_for_display(row["text"]),
            "agent_text_display": clean_text_for_display(agent_row["text"]),
            "created_at": row.get("created_at"),
            "intent_true": row.get("intent_true", ""),  # only populated for synthetic data
        })

    pairs_df = pd.DataFrame(pairs).drop_duplicates(subset=["customer_tweet_id"]).reset_index(drop=True)
    logger.info("Built %d clean conversation pairs for brand '%s'", len(pairs_df), brand)
    return pairs_df


def run_preprocessing(raw_path: str, min_conversations: int = 20,
                       forced_brand: str | None = None) -> tuple[pd.DataFrame, str]:
    """Top-level entry point used by run_pipeline.py."""
    df = load_raw_dataset(raw_path)
    brand = forced_brand or select_brand(df, min_conversations=min_conversations)
    pairs_df = build_conversation_pairs(df, brand)
    return pairs_df, brand
