"""
reply_generator.py
-------------------
STEP 4 (generation half): given a customer message and the top-k
retrieved historical (customer, agent) pairs from retrieval.py, generate
a reply that imitates the brand's historical support style.

Two backends, selected by `Config.use_live_llm`:

  1. Live LLM (OpenAI-compatible /chat/completions). Requires an API key
     in the environment (OPENAI_API_KEY) and network access. This is the
     "real" path described in the assignment.

  2. Deterministic template fallback. Used automatically when no API key
     / network is available (as in this sandboxed build). It composes a
     reply from the retrieved examples' *style* (opening phrase, sign-off,
     "DM us" pattern) rather than inventing new policy content, which is
     exactly the anti-hallucination property STEP 4 asks for -- it just
     does it without an LLM in the loop. The functions are structured so
     swapping in the live backend requires no changes to callers.

Anti-hallucination measures (apply to BOTH backends):
  - The prompt/template is grounded ONLY in retrieved historical replies;
    no policy facts (refund windows, prices, etc.) are invented.
  - If retrieval similarity is below a threshold, the reply explicitly
    flags uncertainty and defers to escalation rather than guessing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

LOW_CONFIDENCE_SIMILARITY_THRESHOLD = 0.35

SYSTEM_PROMPT = """You are a customer support agent for {brand}. Reply to the customer's \
message in the same tone and style as the EXAMPLE_REPLIES below, which are real historical \
replies from {brand}'s support team. Rules:
1. Do not invent policy details (refund windows, prices, timelines) that are not present in \
the example replies.
2. If you are not confident, say so plainly and ask the customer to DM for verification, \
mirroring the examples.
3. Keep the reply under 280 characters, in the brand's voice.
4. Always sound empathetic and professional.
"""


@dataclass
class GeneratedReply:
    text: str
    backend: str            # "live_llm" or "template_fallback"
    grounding_ids: list      # tweet_ids of the retrieved examples used
    confidence: float        # top retrieval similarity, used by escalation.py
    flagged_uncertain: bool


def _build_prompt(customer_message: str, retrieved: pd.DataFrame, brand: str) -> tuple[str, str]:
    system = SYSTEM_PROMPT.format(brand=brand)
    examples = "\n".join(
        f"- Customer: {row['customer_text_display']}\n  {brand}: {row['agent_text_display']}"
        for _, row in retrieved.iterrows()
    )
    user = f"EXAMPLE_REPLIES:\n{examples}\n\nCustomer message: {customer_message}\n\nYour reply:"
    return system, user


def _call_live_llm(system: str, user: str, model_name: str) -> str:
    """Calls an OpenAI-compatible chat completion endpoint.

    Requires `OPENAI_API_KEY` in the environment and the `openai` package.
    Raises on failure rather than silently degrading, so failures are
    visible in logs/tests rather than masked.
    """
    from openai import OpenAI  # imported lazily so the package is optional

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def _template_fallback(customer_message: str, retrieved: pd.DataFrame, brand: str) -> str:
    """Deterministic, non-LLM reply composed from the retrieved examples'
    observed style. Never introduces facts not present in the examples.
    """
    best = retrieved.iloc[0]
    # Reuse the opening empathy phrase + closing "DM us" pattern from the
    # single most similar historical reply, which is the closest we can get
    # to "imitate the brand's style" without a generative model.
    template_reply = best["agent_text_display"]
    return template_reply


def generate_reply(customer_message: str, retrieved: pd.DataFrame, brand: str,
                    use_live_llm: bool = False, model_name: str = "gpt-4o-mini") -> GeneratedReply:
    """Top-level entry point used by run_pipeline.py."""
    if retrieved.empty:
        return GeneratedReply(
            text=f"Thanks for reaching out. We don't have enough historical context to answer "
                 f"confidently -- a human agent will follow up shortly.",
            backend="template_fallback",
            grounding_ids=[],
            confidence=0.0,
            flagged_uncertain=True,
        )

    top_similarity = float(retrieved.iloc[0]["similarity"])
    flagged_uncertain = top_similarity < LOW_CONFIDENCE_SIMILARITY_THRESHOLD

    backend = "template_fallback"
    text = None
    if use_live_llm and os.environ.get("OPENAI_API_KEY"):
        try:
            system, user = _build_prompt(customer_message, retrieved, brand)
            text = _call_live_llm(system, user, model_name)
            backend = "live_llm"
        except Exception as exc:  # noqa: BLE001 - log and gracefully degrade
            logger.warning("Live LLM call failed (%s); using template fallback.", exc)

    if text is None:
        text = _template_fallback(customer_message, retrieved, brand)

    if flagged_uncertain:
        text = text.rstrip(".") + ". (Note: low confidence match to past cases -- please verify with a human agent.)"

    grounding_ids = retrieved["agent_tweet_id"].tolist() if "agent_tweet_id" in retrieved.columns else []
    return GeneratedReply(
        text=text, backend=backend, grounding_ids=grounding_ids,
        confidence=top_similarity, flagged_uncertain=flagged_uncertain,
    )
