"""
evaluation.py
-------------
STEP 3 (metrics), STEP 7 (full evaluation harness), and support for
STEP 8/9 (failure analysis / baseline comparison tables).

Sections:
  1. Intent classification metrics (accuracy/precision/recall/F1/confusion matrix)
  2. Reply generation metrics (BLEU, ROUGE, BERTScore)
  3. LLM-as-Judge scaffold (prompt template + parser; gracefully no-ops
     without an API key, returning None scores rather than fabricated ones)
  4. Human vs LLM-judge agreement (Cohen's Kappa)
  5. Failure analysis helper
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 1. Classification metrics
# --------------------------------------------------------------------------- #

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: np.ndarray
    labels: list
    report_text: str


def evaluate_classifier(y_true: pd.Series, y_pred: np.ndarray) -> ClassificationMetrics:
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels),
        labels=labels,
        report_text=classification_report(y_true, y_pred, zero_division=0),
    )


# --------------------------------------------------------------------------- #
# 2. Generation metrics: BLEU, ROUGE, BERTScore
# --------------------------------------------------------------------------- #

def _simple_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Dependency-free BLEU approximation (geometric mean of modified
    n-gram precisions with add-1 smoothing + brevity penalty). Used only
    when `nltk` is not installed, so the pipeline still runs offline;
    prefer the NLTK implementation (used automatically when available)
    for a standards-compliant score."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = [tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)]
        hyp_ngrams = [tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1)]
        if not hyp_ngrams:
            precisions.append(1e-9)
            continue
        from collections import Counter

        ref_counts, hyp_counts = Counter(ref_ngrams), Counter(hyp_ngrams)
        overlap = sum(min(c, ref_counts.get(g, 0)) for g, c in hyp_counts.items())
        precisions.append((overlap + 1) / (len(hyp_ngrams) + 1))  # add-1 smoothing

    geo_mean = float(np.exp(np.mean(np.log(precisions))))
    brevity_penalty = min(1.0, np.exp(1 - len(ref_tokens) / max(1, len(hyp_tokens))))
    return geo_mean * brevity_penalty


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Sentence-level BLEU. Uses NLTK with smoothing when available
    (standard, citable implementation); otherwise falls back to a
    dependency-free approximation (`_simple_bleu`) so the pipeline still
    runs fully offline. The active backend is logged once at import time."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not hyp_tokens or not ref_tokens:
            return 0.0
        smoothie = SmoothingFunction().method4
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
    except ImportError:
        return _simple_bleu(reference, hypothesis)


def _simple_rouge(reference: str, hypothesis: str) -> dict[str, float]:
    """Dependency-free ROUGE-1/2/L F-measure fallback, used only when the
    `rouge-score` package is not installed."""
    from collections import Counter

    def f1_from_counts(ref_tokens, hyp_tokens, n):
        ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1))
        overlap = sum((ref_ngrams & hyp_ngrams).values())
        if overlap == 0 or not ref_ngrams or not hyp_ngrams:
            return 0.0
        precision = overlap / sum(hyp_ngrams.values())
        recall = overlap / sum(ref_ngrams.values())
        return 2 * precision * recall / (precision + recall)

    def lcs_len(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    ref_tokens, hyp_tokens = reference.lower().split(), hypothesis.lower().split()
    rouge1 = f1_from_counts(ref_tokens, hyp_tokens, 1)
    rouge2 = f1_from_counts(ref_tokens, hyp_tokens, 2)
    lcs = lcs_len(ref_tokens, hyp_tokens)
    if lcs == 0 or not ref_tokens or not hyp_tokens:
        rougeL = 0.0
    else:
        p, r = lcs / len(hyp_tokens), lcs / len(ref_tokens)
        rougeL = 2 * p * r / (p + r)
    return {"rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL}


def compute_rouge(reference: str, hypothesis: str) -> dict[str, float]:
    """ROUGE-1/2/L F-measure via the `rouge-score` package when available,
    else a dependency-free fallback (`_simple_rouge`) for offline runs."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {k: v.fmeasure for k, v in scores.items()}
    except ImportError:
        return _simple_rouge(reference, hypothesis)


def compute_bertscore(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Corpus-level BERTScore (P/R/F1). This is the most expensive metric
    (downloads a transformer model), so it's computed once over the whole
    eval set rather than per-example."""
    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(hypotheses, references, lang="en", verbose=False)
        return {"precision": float(P.mean()), "recall": float(R.mean()), "f1": float(F1.mean())}
    except Exception as exc:  # noqa: BLE001
        logger.warning("BERTScore unavailable (%s). This commonly happens without network "
                        "access to download the underlying model checkpoint; reporting None "
                        "rather than a fabricated score.", exc)
        return {"precision": None, "recall": None, "f1": None}


# --------------------------------------------------------------------------- #
# 3. LLM-as-Judge
# --------------------------------------------------------------------------- #

JUDGE_PROMPT_TEMPLATE = """You are evaluating a customer support agent's reply.

Customer message: "{customer_message}"
Agent's reply: "{agent_reply}"
Reference (a real historical reply to a similar message): "{reference_reply}"

Score the agent's reply from 1 (worst) to 5 (best) on each criterion. Respond ONLY as JSON:
{{
  "correctness": <1-5>,
  "helpfulness": <1-5>,
  "groundedness": <1-5>,
  "tone_consistency": <1-5>,
  "safety": <1-5>,
  "rationale": "<one sentence>"
}}

Criteria definitions:
- correctness: factually consistent with the reference / does not invent policy.
- helpfulness: actually addresses the customer's problem.
- groundedness: only uses information supported by the reference reply.
- tone_consistency: matches the brand's empathetic, professional tone.
- safety: no harmful, biased, or inappropriate content.
"""


def llm_judge(customer_message: str, agent_reply: str, reference_reply: str,
              model_name: str = "gpt-4o-mini") -> dict | None:
    """Calls an LLM judge if OPENAI_API_KEY is set; otherwise returns None
    (never fabricates a judge score). See run_pipeline.py for how missing
    judge scores are surfaced in the final report rather than hidden.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        import json

        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            customer_message=customer_message, agent_reply=agent_reply, reference_reply=reference_reply,
        )
        resp = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=200, response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM judge call failed (%s); returning None.", exc)
        return None


# --------------------------------------------------------------------------- #
# 4. Human vs LLM-judge agreement
# --------------------------------------------------------------------------- #

def cohens_kappa(human_labels: list, llm_labels: list) -> float | None:
    """Cohen's Kappa between human annotations and LLM-judge decisions
    (e.g. both reduced to a binary Auto/Escalate or a pass/fail bucket).
    Returns None if either label list is empty/misaligned rather than a
    misleading 0.0.
    """
    if not human_labels or not llm_labels or len(human_labels) != len(llm_labels):
        logger.warning("Cannot compute Cohen's Kappa: missing or misaligned labels "
                        "(human=%d, llm=%d). This is expected until the golden set is "
                        "human-annotated -- see reports/report.md limitations.",
                        len(human_labels or []), len(llm_labels or []))
        return None
    from sklearn.metrics import cohen_kappa_score

    return cohen_kappa_score(human_labels, llm_labels)


def judge_quality_label(judge_score: dict) -> int | None:
    """Convert the five rubric scores into one reproducible quality label."""
    criteria = ["correctness", "helpfulness", "groundedness", "tone_consistency", "safety"]
    values = [judge_score.get(key) for key in criteria]
    if any(not isinstance(value, (int, float)) or not 1 <= value <= 5 for value in values):
        return None
    return int(round(float(np.mean(values))))


# --------------------------------------------------------------------------- #
# 5. Failure analysis
# --------------------------------------------------------------------------- #

@dataclass
class FailureCase:
    example: str
    prediction: str
    expected: str
    likely_cause: str
    possible_fix: str


def top_failure_modes(y_true: pd.Series, y_pred: np.ndarray, texts: pd.Series, k: int = 5) -> list[FailureCase]:
    """Group misclassifications by (true, predicted) pair, surface the
    most frequent confusions, and attach a heuristic likely-cause /
    possible-fix (STEP 8). The cause/fix text is templated from the
    confusion pattern itself (e.g. semantically adjacent intents), which
    keeps it grounded in the actual error rather than generic advice.
    """
    df = pd.DataFrame({"text": texts.values, "true": y_true.values, "pred": y_pred})
    mistakes = df[df["true"] != df["pred"]]
    if mistakes.empty:
        return []

    pair_counts = mistakes.groupby(["true", "pred"]).size().sort_values(ascending=False)
    cases = []
    for (true_label, pred_label), count in pair_counts.head(k).items():
        example_row = mistakes[(mistakes["true"] == true_label) & (mistakes["pred"] == pred_label)].iloc[0]
        cases.append(FailureCase(
            example=example_row["text"],
            prediction=pred_label,
            expected=true_label,
            likely_cause=(
                f"'{true_label}' and '{pred_label}' share overlapping vocabulary "
                f"({count} such confusions in the eval set), suggesting the model relies on "
                f"surface keywords rather than deeper intent semantics."
            ),
            possible_fix=(
                f"Add more contrastive training examples that distinguish '{true_label}' from "
                f"'{pred_label}', or add a rule-based tiebreaker for their most confusable keywords."
            ),
        ))
    return cases
