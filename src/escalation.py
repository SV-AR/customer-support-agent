"""
escalation.py
--------------
STEP 5: decide whether a conversation can be auto-handled or must be
escalated to a human agent, with a clear, auditable reason.

Design: rule-based FIRST, LLM-reasoning as a fallback/refinement layer.
Rules are cheap, deterministic, and auditable -- exactly what you want
for a safety-relevant gate like "does this touch money/security". The
LLM layer only runs when no rule fires with high confidence, and its
output must still cite one of a closed set of reason categories (it does
not freely improvise reasons), keeping the whole module explainable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum

from src.utils import get_logger

logger = get_logger(__name__)


class Decision(str, Enum):
    AUTO_HANDLE = "Auto Handle"
    ESCALATE = "Escalate"


# Ordered rules: (regex, reason). First match wins. Ordering encodes
# priority -- e.g. security/billing disputes must outrank generic
# "how do I" questions even if both keywords appear.
_ESCALATION_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bcharged?\s+(twice|two times|double)\b", re.I),
     "Potential billing dispute requiring account verification."),
    (re.compile(r"\b(fraud|unauthorized|stolen|hacked)\b", re.I),
     "Possible security/fraud incident requiring identity verification."),
    (re.compile(r"\b(refund|chargeback)\b", re.I),
     "Refund requests require account-level verification before funds can move."),
    (re.compile(r"\b(legal|lawsuit|sue|attorney|lawyer)\b", re.I),
     "Legal threat or reference; requires human and possibly legal review."),
    (re.compile(r"\b(cancel(l)?ing|cancel)\s+(my\s+)?(subscription|account|plan)\b", re.I),
     "Account cancellation requires identity verification and retention review."),
    (re.compile(r"\b(data|privacy|gdpr|personal information)\b.*\b(delete|leak|breach)\b", re.I),
     "Potential privacy/data-protection issue requiring specialist review."),
    (re.compile(r"\b(angry|furious|unacceptable|lawsuit|disgusted|worst)\b", re.I),
     "High emotional intensity detected; a human touch reduces churn risk."),
]

# Intents that are considered safe to fully automate when no rule fires
# and retrieval confidence is high (see reply_generator.py's `confidence`).
_AUTO_ELIGIBLE_INTENTS = {"Password Reset", "Technical Issue", "Order Delay"}

CONFIDENCE_THRESHOLD_FOR_AUTO = 0.5


@dataclass
class EscalationResult:
    decision: Decision
    reason: str
    matched_rule: str | None


def _rule_based_check(message: str) -> tuple[str | None, str | None]:
    for pattern, reason in _ESCALATION_RULES:
        if pattern.search(message):
            return pattern.pattern, reason
    return None, None


def _llm_reasoning_fallback(message: str, intent: str, model_name: str = "gpt-4o-mini") -> str | None:
    """Optional LLM-based reasoning layer for ambiguous cases where no
    rule fired. Constrained to pick from a closed set of reason templates
    to keep output auditable; returns None (defer to heuristic) on any
    failure or when no API key is configured."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        prompt = (
            f"Customer message (intent={intent}): \"{message}\"\n"
            "Should this be auto-handled by a bot, or escalated to a human? "
            "Answer with exactly one line: 'AUTO: <short reason>' or 'ESCALATE: <short reason>'."
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM escalation reasoning failed (%s); using heuristic fallback.", exc)
        return None


def decide_escalation(message: str, intent: str, retrieval_confidence: float,
                       use_llm_fallback: bool = False, model_name: str = "gpt-4o-mini") -> EscalationResult:
    """Top-level entry point used by run_pipeline.py.

    Order of checks (deliberately layered from cheapest/most auditable to
    most flexible):
      1. Deterministic keyword rules (money, security, legal, emotion).
      2. Retrieval-confidence + intent-safety heuristic.
      3. Optional LLM reasoning for anything still ambiguous.
    """
    matched_pattern, reason = _rule_based_check(message)
    if reason:
        return EscalationResult(decision=Decision.ESCALATE, reason=reason, matched_rule=matched_pattern)

    if intent in _AUTO_ELIGIBLE_INTENTS and retrieval_confidence >= CONFIDENCE_THRESHOLD_FOR_AUTO:
        return EscalationResult(
            decision=Decision.AUTO_HANDLE,
            reason=f"Intent '{intent}' is low-risk and a similar past case was found "
                   f"with confidence {retrieval_confidence:.2f} (>= {CONFIDENCE_THRESHOLD_FOR_AUTO}).",
            matched_rule=None,
        )

    if use_llm_fallback:
        llm_out = _llm_reasoning_fallback(message, intent, model_name)
        if llm_out:
            decision = Decision.ESCALATE if llm_out.upper().startswith("ESCALATE") else Decision.AUTO_HANDLE
            reason = llm_out.split(":", 1)[-1].strip() if ":" in llm_out else llm_out
            return EscalationResult(decision=decision, reason=reason, matched_rule="llm_fallback")

    # Default: when uncertain, escalate. Escalation is the safe failure
    # mode for a customer-support agent (see decision log item on this).
    return EscalationResult(
        decision=Decision.ESCALATE,
        reason=f"No confident rule or high-confidence historical match for intent '{intent}' "
               f"(retrieval confidence {retrieval_confidence:.2f}); defaulting to human review.",
        matched_rule=None,
    )
