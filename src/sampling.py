"""
sampling.py
-----------
STEP 6: sample a golden evaluation set and produce the annotation
spreadsheet skeleton (tweet, intent, correct reply, auto/escalate, notes).

Sampling methodology
=====================
Stratified random sampling by *predicted intent*, not pure random
sampling. Pure random sampling over an intent distribution that is
itself imbalanced (some intents are far more common than others) would
under-represent rare-but-important intents (e.g. "Account Locked") in
the eval set, which is exactly where we most want the classifier and
escalation module to be trustworthy. Within each intent stratum we
sample uniformly at random with a fixed seed for reproducibility.

Labeling methodology
=====================
This script produces the *skeleton* spreadsheet with model predictions
pre-filled as a starting point, plus empty columns for a human annotator
to fill in `correct_reply`, confirm/correct `auto/escalate`, and add
free-text `notes`. In a real internship setting this would be handed to
1-2 human labelers; inter-annotator agreement would be measured the same
way we measure human-vs-LLM-judge agreement in evaluation.py (Cohen's
Kappa). Since no human labeler is available in this sandboxed build, the
`correct_reply` / final `auto_or_escalate` columns are left blank by
design rather than fabricated -- see reports/report.md limitations.
"""

from __future__ import annotations

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def stratified_sample(df: pd.DataFrame, intent_col: str, n_total: int, seed: int) -> pd.DataFrame:
    """Stratified-by-intent random sample of up to n_total rows.

    If a stratum has fewer rows than its proportional share, we take all
    of it (no replacement) and log the shortfall rather than silently
    oversampling.
    """
    if len(df) <= n_total:
        logger.info("Dataset smaller than requested sample size (%d <= %d); using full dataset.",
                     len(df), n_total)
        return df.copy()

    fractions = df[intent_col].value_counts(normalize=True)
    parts = []
    for intent, frac in fractions.items():
        stratum = df[df[intent_col] == intent]
        target_n = max(1, round(frac * n_total))
        take_n = min(target_n, len(stratum))
        if take_n < target_n:
            logger.warning("Stratum '%s' has only %d rows (< requested %d); taking all.",
                            intent, len(stratum), target_n)
        parts.append(stratum.sample(n=take_n, random_state=seed))

    sample_df = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    # Trim/pad to hit n_total as closely as possible after rounding.
    if len(sample_df) > n_total:
        sample_df = sample_df.iloc[:n_total]
    logger.info("Drew stratified golden sample of %d rows across %d intents.",
                len(sample_df), df[intent_col].nunique())
    return sample_df


def build_annotation_sheet(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Build the STEP 6 annotation spreadsheet with the required columns.
    Model-predicted fields are pre-filled; human-authored fields are left
    blank for annotators.
    """
    sheet = pd.DataFrame({
        "tweet": sample_df["customer_text_display"],
        "predicted_intent": sample_df.get("predicted_intent", ""),
        "intent": "",  # human fills in / confirms
        "model_generated_reply": sample_df.get("generated_reply", ""),
        "correct_reply": "",  # human fills in
        "predicted_auto_or_escalate": sample_df.get("predicted_decision", ""),
        "auto/escalate": "",  # human fills in / confirms
        "notes": "",  # human fills in
    })
    return sheet
