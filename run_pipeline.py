#!/usr/bin/env python3
"""
run_pipeline.py
================
Single entry point that runs the full assignment pipeline end-to-end:

  STEP 1  Load data, select brand
  STEP 2  Discover intents via clustering
  STEP 3  Train + evaluate 3 intent classifiers (random / TF-IDF / embeddings)
  STEP 4  Build FAISS retrieval index + generate grounded replies
  STEP 5  Escalation decisions
  STEP 6  Golden evaluation sample + annotation sheet
  STEP 7  Evaluation harness (BLEU / ROUGE / BERTScore / LLM-judge / Kappa)
  STEP 8  Failure analysis
  STEP 9  Baseline comparison table
  STEP 10-12 are the static reports/README/decision-log files that ship
             alongside this script (reports/report.md, README.md,
             reports/decision_log.md) -- they reference the numbers this
             script produces in outputs/.

Usage:
    python run_pipeline.py

All numbers, tables, and figures this script produces are written to
`outputs/`. Every artifact is either (a) computed live from data.csv in
this run, or (b) explicitly logged/labeled as unavailable (e.g. an LLM
judge score with no API key configured) -- nothing is fabricated.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import escalation, evaluation, preprocessing, retrieval, sampling
from src.embeddings import get_embedder
from src.intent_classifier import (
    discover_intents,
    predict,
    split_data,
    train_embedding_classifier,
    train_random_baseline,
    train_tfidf_logreg,
)
from src.reply_generator import generate_reply
from src.utils import Config, get_logger, save_json, save_run_manifest, set_global_seed

warnings.filterwarnings("ignore", category=UserWarning)
logger = get_logger("run_pipeline")


def main(config: Config | None = None) -> None:
    config = config or Config()
    set_global_seed(config.random_seed)
    out_dir = Path(config.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_manifest(config, out_dir / "run_manifest.json")

    # ------------------------------------------------------------------ #
    # STEP 1: Load + select brand
    # ------------------------------------------------------------------ #
    logger.info("=" * 70)
    logger.info("STEP 1: Loading dataset and selecting brand")
    logger.info("=" * 70)
    pairs_df, brand = preprocessing.run_preprocessing(
        raw_path=config.raw_data_path,
        min_conversations=20,
        forced_brand=None,
    )
    pairs_df.to_csv(out_dir / "conversation_pairs.csv", index=False)

    # ------------------------------------------------------------------ #
    # Embeddings (shared across clustering / classification / retrieval)
    # ------------------------------------------------------------------ #
    embedder = get_embedder(config.embedding_model_name)
    embedder.fit(pairs_df["customer_text"].tolist())
    all_embeddings = embedder.encode(pairs_df["customer_text"].tolist())
    logger.info("Embedding backend in use: %s (dim=%d)", embedder.backend_name, all_embeddings.shape[1])

    # ------------------------------------------------------------------ #
    # STEP 2: Intent discovery
    # ------------------------------------------------------------------ #
    logger.info("=" * 70)
    logger.info("STEP 2: Discovering intents via clustering")
    logger.info("=" * 70)
    cluster_ids, cluster_label_map, top_terms_df = discover_intents(
        texts=pairs_df["customer_text"], embeddings=all_embeddings,
        n_clusters=config.n_intent_clusters, random_state=config.random_seed,
    )
    pairs_df["predicted_intent"] = [cluster_label_map[c] for c in cluster_ids]
    top_terms_df.to_csv(out_dir / "discovered_intents.csv", index=False)

    # Sanity-check against synthetic ground truth if present (only exists
    # for the synthetic sample -- real Kaggle data has no such column).
    if "intent_true" in pairs_df.columns and pairs_df["intent_true"].astype(bool).any():
        agree = (pairs_df["intent_true"] == pairs_df["predicted_intent"]).mean()
        logger.info("[Synthetic-data sanity check only] cluster label vs known intent "
                     "agreement: %.1f%% (not a metric available on the real dataset).", agree * 100)

    # For classification we need a label column. Use predicted_intent
    # (from clustering) as the training target -- this is the "custom
    # intent labels learned from the dataset" the assignment asks for in
    # STEP 1/2. On the real dataset there is no ground truth intent, so
    # this self-supervised-via-clustering label IS the target throughout.
    label_col = "predicted_intent"

    # ------------------------------------------------------------------ #
    # STEP 3: Intent classification -- 3 tiers
    # ------------------------------------------------------------------ #
    logger.info("=" * 70)
    logger.info("STEP 3: Training and evaluating intent classifiers")
    logger.info("=" * 70)

    # split_data() returns (X_train, X_test, y_train, y_test). We split on
    # the row *index* itself (rather than the text) so the same split can
    # be applied consistently to the text column, the label column, and
    # the pre-computed embedding matrix below.
    train_idx, test_idx, y_train, y_test = split_data(
        pairs_df.index.to_series(), pairs_df[label_col], test_size=config.test_size, seed=config.random_seed,
    )

    train_df = pairs_df.loc[train_idx.values].reset_index(drop=True)
    test_df = pairs_df.loc[test_idx.values].reset_index(drop=True)
    train_pos = pairs_df.index.get_indexer(train_idx.values)
    test_pos = pairs_df.index.get_indexer(test_idx.values)
    train_emb = all_embeddings[train_pos]
    test_emb = all_embeddings[test_pos]

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    random_model = train_random_baseline(y_train, seed=config.random_seed)
    tfidf_model = train_tfidf_logreg(train_df["customer_text"], y_train,
                                      max_features=config.tfidf_max_features, seed=config.random_seed)
    embed_model = train_embedding_classifier(train_emb, y_train, seed=config.random_seed, use_lightgbm=False)

    results = {}
    predictions = {}
    for name, model, kwargs in [
        ("random_baseline", random_model, dict(X_text=test_df["customer_text"])),
        ("tfidf_logreg", tfidf_model, dict(X_text=test_df["customer_text"])),
        ("embedding_classifier", embed_model, dict(X_emb=test_emb)),
    ]:
        y_pred = predict(model, **kwargs)
        predictions[name] = y_pred
        metrics = evaluation.evaluate_classifier(y_test, y_pred)
        results[name] = metrics
        logger.info("[%s] accuracy=%.3f  f1_macro=%.3f  precision_macro=%.3f  recall_macro=%.3f",
                     name, metrics.accuracy, metrics.f1_macro, metrics.precision_macro, metrics.recall_macro)
        with open(out_dir / f"classification_report_{name}.txt", "w") as f:
            f.write(metrics.report_text)

    # Comparison table (STEP 9 uses this same table)
    comparison_df = pd.DataFrame([
        {"model": name, "accuracy": m.accuracy, "precision_macro": m.precision_macro,
         "recall_macro": m.recall_macro, "f1_macro": m.f1_macro}
        for name, m in results.items()
    ])
    comparison_df.to_csv(out_dir / "model_comparison.csv", index=False)
    logger.info("Model comparison:\n%s", comparison_df.to_string(index=False))

    # Confusion matrix figure for the final model
    final_metrics = results["embedding_classifier"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(final_metrics.confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=final_metrics.labels, yticklabels=final_metrics.labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix -- Embedding Classifier ({brand})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_embedding_classifier.png", dpi=150)
    plt.close(fig)

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    comparison_df.set_index("model")[["accuracy", "f1_macro"]].plot(kind="bar", ax=ax)
    ax.set_title("Baseline Comparison: Accuracy & Macro-F1")
    ax.set_ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    # Intent distribution chart
    fig, ax = plt.subplots(figsize=(8, 5))
    pairs_df[label_col].value_counts().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Discovered Intent Distribution -- {brand}")
    ax.set_xlabel("Count")
    plt.tight_layout()
    fig.savefig(out_dir / "intent_distribution.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # STEP 4: Retrieval + reply generation (RAG)
    # ------------------------------------------------------------------ #
    logger.info("=" * 70)
    logger.info("STEP 4: Building retrieval index and generating replies")
    logger.info("=" * 70)
    # Index is built ONLY on the training split so that generating a
    # reply for a test-set message never retrieves its own ground-truth
    # answer (would trivially inflate BLEU/ROUGE).
    retrieval_index = retrieval.build_index(train_emb, train_df)

    test_df = test_df.copy()
    test_df["predicted_intent"] = predict(embed_model, X_emb=test_emb)

    generated_replies, retrieval_confidences, grounding_ids_list = [], [], []
    for i, row in test_df.iterrows():  # test_df has a fresh 0..n-1 index, matching test_emb rows
        query_emb = test_emb[i]
        retrieved = retrieval.search(query_emb, retrieval_index, top_k=config.faiss_top_k)
        gen = generate_reply(row["customer_text_display"], retrieved, brand,
                              use_live_llm=config.use_live_llm, model_name=config.llm_model_name)
        generated_replies.append(gen.text)
        retrieval_confidences.append(gen.confidence)
        grounding_ids_list.append(gen.grounding_ids)

    test_df["generated_reply"] = generated_replies
    test_df["retrieval_confidence"] = retrieval_confidences
    test_df["grounding_agent_tweet_ids"] = grounding_ids_list

    # ------------------------------------------------------------------ #
    # STEP 5: Escalation
    # ------------------------------------------------------------------ #
    logger.info("=" * 70)
    logger.info("STEP 5: Escalation decisions")
    logger.info("=" * 70)
    decisions, reasons = [], []
    for _, row in test_df.iterrows():
        result = escalation.decide_escalation(
            message=row["customer_text_display"], intent=row["predicted_intent"],
            retrieval_confidence=row["retrieval_confidence"], use_llm_fallback=config.use_live_llm,
        )
        decisions.append(result.decision.value)
        reasons.append(result.reason)
    test_df["predicted_decision"] = decisions
    test_df["escalation_reason"] = reasons

    escalation_rate = (test_df["predicted_decision"] == "Escalate").mean()
    logger.info("Escalation rate on test set: %.1f%%", escalation_rate * 100)

    test_df.to_csv(out_dir / "test_predictions_full.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 5))
    test_df["predicted_decision"].value_counts().plot(kind="pie", autopct="%1.0f%%", ax=ax)
    ax.set_ylabel("")
    ax.set_title("Auto-Handle vs Escalate (test set)")
    plt.tight_layout()
    fig.savefig(out_dir / "escalation_breakdown.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # STEP 6: Golden evaluation sample + annotation sheet
    # ------------------------------------------------------------------ #
    logger.info("=" * 70)
    logger.info("STEP 6: Building golden evaluation sample")
    logger.info("=" * 70)
    golden_sample = sampling.stratified_sample(
        test_df, intent_col="predicted_intent", n_total=config.golden_sample_size, seed=config.random_seed,
    )
    annotation_path = out_dir / "golden_eval_annotation_sheet.csv"
    if annotation_path.exists():
        existing_sheet = pd.read_csv(annotation_path)
        required_columns = {"tweet", "intent", "correct_reply", "reply_quality", "auto/escalate", "notes"}
        if len(existing_sheet) == len(golden_sample) and required_columns.issubset(existing_sheet.columns):
            annotation_sheet = existing_sheet
            logger.info("Reusing existing golden annotation sheet; human labels will not be overwritten.")
        else:
            annotation_sheet = sampling.build_annotation_sheet(golden_sample)
            annotation_sheet.to_csv(annotation_path, index=False)
            logger.info("Golden annotation sheet regenerated with %d rows; complete it using "
                        "reports/golden_annotation_guide.md.", len(annotation_sheet))
    else:
        annotation_sheet = sampling.build_annotation_sheet(golden_sample)
        annotation_sheet.to_csv(annotation_path, index=False)
        logger.info("Golden annotation sheet written with %d rows; complete it using "
                    "reports/golden_annotation_guide.md.", len(annotation_sheet))

    # ------------------------------------------------------------------ #
    # STEP 7: Evaluation harness -- generation metrics + LLM judge + kappa
    # ------------------------------------------------------------------ #
    logger.info("=" * 70)
    logger.info("STEP 7: Evaluation harness (BLEU / ROUGE / BERTScore / LLM-judge)")
    logger.info("=" * 70)
    gen_rows = []
    judge_scores_available = False
    for _, row in golden_sample.iterrows():
        reference = row["agent_text_display"]  # the REAL historical reply -- held out, not retrieved
        hypothesis = row["generated_reply"]
        bleu = evaluation.compute_bleu(reference, hypothesis)
        rouge = evaluation.compute_rouge(reference, hypothesis)
        judge = evaluation.llm_judge(row["customer_text_display"], hypothesis, reference,
                                      model_name=config.llm_model_name)
        if judge:
            judge_scores_available = True
        gen_rows.append({
            "customer_text": row["customer_text_display"], "reference_reply": reference,
            "generated_reply": hypothesis, "bleu": bleu, "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"], "rougeL": rouge["rougeL"],
            "llm_judge": json.dumps(judge) if judge else None,
        })
    gen_metrics_df = pd.DataFrame(gen_rows)
    gen_metrics_df.to_csv(out_dir / "generation_metrics_per_example.csv", index=False)

    bertscore = evaluation.compute_bertscore(
        gen_metrics_df["reference_reply"].tolist(), gen_metrics_df["generated_reply"].tolist(),
    )

    generation_summary = {
        "n_examples": len(gen_metrics_df),
        "bleu_mean": float(gen_metrics_df["bleu"].mean()),
        "rouge1_mean": float(gen_metrics_df["rouge1"].mean()),
        "rouge2_mean": float(gen_metrics_df["rouge2"].mean()),
        "rougeL_mean": float(gen_metrics_df["rougeL"].mean()),
        "bertscore": bertscore,
        "llm_judge_scores_available": judge_scores_available,
    }
    if not judge_scores_available:
        generation_summary["llm_judge_note"] = (
            "No OPENAI_API_KEY configured in this environment, so LLM-as-Judge scores are "
            "NOT computed and are reported as unavailable rather than fabricated. Set "
            "OPENAI_API_KEY and Config.use_live_llm=True to populate this section."
        )
    save_json(generation_summary, out_dir / "generation_metrics_summary.json")
    logger.info("Generation metrics summary: %s", json.dumps(generation_summary, indent=2, default=str))

    human_labels, llm_labels = [], []
    if judge_scores_available and "reply_quality" in annotation_sheet.columns:
        for raw_human, raw_judge in zip(annotation_sheet["reply_quality"], gen_rows):
            try:
                human_label = int(raw_human)
            except (TypeError, ValueError):
                continue
            judge_payload = json.loads(raw_judge["llm_judge"]) if raw_judge["llm_judge"] else None
            judge_label = evaluation.judge_quality_label(judge_payload) if judge_payload else None
            if judge_label is not None and 1 <= human_label <= 5:
                human_labels.append(human_label)
                llm_labels.append(judge_label)
    kappa = evaluation.cohens_kappa(human_labels, llm_labels)
    agreement_payload = {
        "cohens_kappa": kappa,
        "n_compared": len(human_labels),
        "human_label_field": "reply_quality",
        "judge_label": "rounded mean of correctness, helpfulness, groundedness, tone_consistency, safety",
    }
    if kappa is None:
        agreement_payload["note"] = (
            "Agreement is unavailable until the 200-row sheet is completed and a live LLM judge "
            "run is performed. See reports/golden_annotation_guide.md."
        )
    save_json(agreement_payload, out_dir / "human_llm_agreement.json")

    # ------------------------------------------------------------------ #
    # STEP 8: Failure analysis
    # ------------------------------------------------------------------ #
    logger.info("=" * 70)
    logger.info("STEP 8: Failure analysis")
    logger.info("=" * 70)
    failures = evaluation.top_failure_modes(y_test, predictions["embedding_classifier"],
                                             test_df["customer_text_display"], k=5)
    with open(out_dir / "failure_analysis.md", "w") as f:
        f.write(f"# Top Failure Modes -- {brand} Intent Classifier\n\n")
        if not failures:
            f.write("No misclassifications found on this test split "
                    "(expected on a very small/clean sample; re-run on a larger dataset).\n")
        for i, case in enumerate(failures, 1):
            f.write(f"## Failure {i}: {case.expected} -> predicted {case.prediction}\n\n")
            f.write(f"**Example:** {case.example}\n\n")
            f.write(f"**Prediction:** {case.prediction}  \n**Expected:** {case.expected}\n\n")
            f.write(f"**Why it failed:** {case.likely_cause}\n\n")
            f.write(f"**Possible fix:** {case.possible_fix}\n\n---\n\n")
    logger.info("Wrote %d failure cases to outputs/failure_analysis.md", len(failures))

    # ------------------------------------------------------------------ #
    # Wrap-up
    # ------------------------------------------------------------------ #
    save_json({
        "brand": brand, "n_conversation_pairs": len(pairs_df), "n_train": len(train_df),
        "n_test": len(test_df), "embedding_backend": embedder.backend_name,
        "escalation_rate_test": float(escalation_rate),
    }, out_dir / "run_summary.json")
    logger.info("Pipeline complete. All artifacts written to %s/", out_dir)


if __name__ == "__main__":
    main()
