"""
sampling.py
-----------
STEP 6: sample a golden evaluation set and produce the annotation
spreadsheet (tweet, intent, reply quality, auto/escalate, notes).

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
The generated sheet contains model predictions plus explicit fields for a
human annotator to complete: `intent`, `correct_reply`, `reply_quality`,
`auto/escalate`, and `notes`. The rubric in `reports/golden_annotation_guide.md`
defines each field and the five-point reply-quality scale. The pipeline
does not treat predictions as human labels and does not fabricate agreement
when the fields are blank.
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
    """Build the STEP 6 annotation spreadsheet with the required columns."""
    sheet = pd.DataFrame({
        "tweet": sample_df["customer_text_display"],
        "predicted_intent": sample_df.get("predicted_intent", ""),
        "intent": "",
        "model_generated_reply": sample_df.get("generated_reply", ""),
        "correct_reply": "",
        "reply_quality": "",
        "predicted_auto_or_escalate": sample_df.get("predicted_decision", ""),
        "auto/escalate": "",
        "notes": "",
    })
    return sheet


def validate_annotation_sheet(sheet: pd.DataFrame, expected_rows: int = 200) -> dict:
    """Validate human annotation completeness without judging label quality."""
    required = {
        "tweet", "predicted_intent", "intent", "model_generated_reply",
        "correct_reply", "reply_quality", "predicted_auto_or_escalate",
        "auto/escalate", "notes",
    }
    missing_columns = sorted(required - set(sheet.columns))
    if missing_columns:
        return {"valid": False, "reason": f"Missing columns: {missing_columns}"}

    complete = (
        sheet["intent"].astype(str).str.strip().ne("")
        & sheet["correct_reply"].astype(str).str.strip().ne("")
        & sheet["reply_quality"].astype(str).str.fullmatch(r"[1-5]")
        & sheet["auto/escalate"].astype(str).str.strip().str.lower().isin({"auto handle", "escalate"})
        & sheet["notes"].astype(str).str.strip().ne("")
    )
    return {
        "valid": len(sheet) == expected_rows and bool(complete.all()),
        "rows": len(sheet),
        "expected_rows": expected_rows,
        "complete_rows": int(complete.sum()),
        "incomplete_rows": int((~complete).sum()),
    }
