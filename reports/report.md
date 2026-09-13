# AI Customer Support Agent — Project Report

*Dataset: Kaggle `thoughtvector/customer-support-on-twitter` · Brand: AppleSupport (auto-selected) · Prepared as an internship/research-quality deliverable*

---

## 1. Problem Framing

Brands receive high volumes of customer support requests on Twitter that
mix routine, easily-automatable issues (password resets, order-status
questions) with sensitive ones (billing disputes, security incidents)
that genuinely need a human. The goal of this project is to build, and
rigorously evaluate, a three-stage agent that:

1. **Classifies** an incoming customer message into a small set of
   intents discovered directly from the brand's historical conversation
   data (no pre-existing label set is provided by the dataset).
2. **Generates** a reply grounded in that brand's own historical
   responses, so the agent's tone and policy claims stay consistent with
   how the brand actually talks to customers, and explicitly flags
   uncertainty rather than inventing an answer.
3. **Decides** whether that reply can be sent automatically or whether
   the conversation must be escalated to a human agent, with a specific,
   auditable reason.

The project treats evaluation as the primary deliverable, not model
sophistication: every stage has an explicit metric, a baseline to beat,
and a documented failure mode.

## 2. Dataset

**Source:** Kaggle `thoughtvector/customer-support-on-twitter` — several
million tweets between customers and brand support accounts, linked via
`response_tweet_id` / `in_response_to_tweet_id`.

**Important note on this build:** the pipeline was developed in a
network-isolated sandbox, so the real `twcs.csv` could not be downloaded.
A small (400-row, 200-conversation) **synthetic stand-in** with an
identical schema was generated (`data/generate_sample_data.py`) to make
every stage of the pipeline runnable and reviewable end-to-end. All
numbers in this report are computed live against that synthetic sample.
`src/preprocessing.py` automatically switches to the real file the
moment it's placed at `data/twcs.csv` — no code changes needed. **The
concrete numbers below should be read as "the pipeline works and
produces this shape of output," not as claims about real-world
performance** (see §8, "Misleading headline metric," for why).

**Brand selection (STEP 1):** brands are ranked automatically by a
composite score of conversation volume, text cleanliness (non-empty,
non-boilerplate ratio), and vocabulary richness. On this sample,
**AppleSupport** was selected with 200 qualifying conversations and a
100% clean-text ratio (see `src/preprocessing.py::select_brand` and
`outputs/run_summary.json`). On the real dataset, this same ranking
logic would compare dozens of brand accounts (`AmazonHelp`,
`AppleSupport`, `Delta`, `SpotifyCares`, etc.) and log the full ranking
for auditability.

## 3. Methodology

**Preprocessing.** Inbound customer tweets are paired with their first
outbound brand reply; URLs and @mentions are stripped for the modeling
text (`customer_text` / `agent_text`) while a lightly-cleaned display
version (`customer_text_display` / `agent_text_display`) that keeps
@mentions is retained for generation, since real brand replies always
@-mention the customer.

**Intent discovery (STEP 2).** Customer messages are embedded and
clustered with KMeans (k=8 by default). Each cluster is auto-labeled
from its top TF-IDF terms via a keyword→label lookup (falling back to a
"top-term" name for unmapped clusters), so labels are inspectable rather
than opaque. On the sample, this produced labels including **Refund**
(52 msgs), **Password Reset** (40), **Account Locked** (25),
**Subscription Cancellation** (25+11, split across two clusters),
**Delivery Issue** (19), **Billing Problem** (16), and **Technical
Issue** (12) — see `outputs/discovered_intents.csv`. Two clusters both
resolved to "Subscription Cancellation," a real artifact of k=8 being
slightly too high for this sample's true category count (7) — noted as
a tuning item in §6/§9.

**Classification (STEP 3).** Three tiers, trained/evaluated on an
identical 80/20 stratified split:
- **Baseline 1 — Random:** `DummyClassifier(strategy="stratified")`,
  predicts from the label prior only.
- **Baseline 2 — TF-IDF + Logistic Regression:** bag-of-words
  (1–2 grams, 5000 features) into a balanced logistic regression.
- **Final — Embeddings + Logistic Regression:** sentence embeddings
  (production: `all-MiniLM-L6-v2`; this offline build: TF-IDF+SVD
  fallback, see §7) into the same classifier head, with an optional
  LightGBM swap for larger datasets.

**Retrieval-augmented generation (STEP 4).** A FAISS `IndexFlatIP`
(cosine similarity) index is built **only over the training split**, so
generating a reply for a held-out test message can never retrieve its
own ground-truth answer. The top-k (default 5) most similar historical
(customer, reply) pairs are passed to the generator, which is grounded
strictly in that retrieved content: no policy facts are invented, and
replies below a similarity threshold (0.35) get an explicit "please
verify with a human agent" caveat appended.

**Escalation (STEP 5).** A layered decision: deterministic keyword rules
first (billing disputes, fraud/security, legal threats, cancellations,
high emotional intensity) — these are auditable and take priority.
If no rule fires, low-risk intents (Password Reset, Technical Issue,
Order Delay) with a high-confidence retrieval match are auto-handled.
Everything else defaults to **Escalate**, since for a support agent,
under-escalating is the more costly failure mode than over-escalating.

**Golden evaluation set (STEP 6).** 200 conversations are sampled
*stratified by predicted intent* (not uniformly), so rare intents remain
represented in the eval set even though they're rarer in the raw data —
see decision log #12. The generated sheet is an annotation instrument;
reviewers must complete it using `reports/golden_annotation_guide.md`.
The repository does not claim hand-labeling or agreement evidence until
those fields are completed by reviewers.

**Evaluation harness (STEP 7).** Classification: accuracy, macro
precision/recall/F1, confusion matrix. Generation: BLEU, ROUGE-1/2/L
(against the *real, held-out* historical reply, never the retrieved
one), BERTScore, and an LLM-as-Judge rubric (correctness, helpfulness,
groundedness, tone consistency, safety) with Cohen's Kappa against human
labels. See §7 for what could and couldn't be computed in this sandbox.

## 4. Architecture Diagram

```
                     ┌─────────────────────────┐
                     │   Kaggle twcs.csv        │
                     │  (or synthetic sample)    │
                     └────────────┬──────────────┘
                                  │  preprocessing.py
                                  │  (select brand, pair convos, clean)
                                  ▼
                     ┌─────────────────────────┐
                     │  conversation_pairs.csv  │
                     └────────────┬──────────────┘
                                  │ embeddings.py
                                  ▼
        ┌─────────────────────────────────────────────┐
        │            Shared embedding matrix            │
        └───────┬───────────────────┬───────────────────┘
                 │                   │
     intent_classifier.py      retrieval.py (FAISS, train split only)
     (cluster → label;               │
      3-tier classifiers)            ▼
                 │            reply_generator.py (RAG, grounded)
                 ▼                   │
        predicted_intent             ▼
                 │            generated_reply + confidence
                 └───────┬───────────┘
                         ▼
                 escalation.py
        (rules → intent+confidence heuristic → optional LLM)
                         │
                         ▼
              Auto Handle  /  Escalate + reason
                         │
                         ▼
        sampling.py → golden_eval_annotation_sheet.csv
        evaluation.py → accuracy/F1, BLEU/ROUGE/BERTScore,
                         LLM-judge, Cohen's Kappa, failure analysis
```

## 5. Evaluation Results (synthetic sample)

**Intent classification (test set, n=40):**

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Random baseline | 0.275 | 0.266 | 0.234 | 0.245 |
| TF-IDF + Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 |
| Embeddings + Logistic Regression (final) | 1.000 | 1.000 | 1.000 | 1.000 |

**Generation (golden sample, n=40), against the real held-out reply:**

| Metric | Score |
|---|---|
| BLEU | 0.937 |
| ROUGE-1 | 0.939 |
| ROUGE-2 | 0.935 |
| ROUGE-L | 0.939 |
| BERTScore | *unavailable — `bert-score` package not installed in this sandbox* |
| LLM-as-Judge | *unavailable — no `OPENAI_API_KEY` configured; see §7* |

**Escalation:** 72.5% of the test set was routed to **Escalate**, driven
heavily by the rule set's "default to escalate on uncertainty" policy
(decision log #9) combined with a test set where many intents fall
outside the three auto-eligible categories.

Full per-example outputs: `outputs/test_predictions_full.csv`. Full
confusion matrix: `outputs/confusion_matrix_embedding_classifier.png`.

## 6. Failure Analysis (STEP 8)

`src/evaluation.py::top_failure_modes` groups test-set misclassifications
by (true intent, predicted intent) pair and surfaces the most frequent
confusions with an automatically-derived cause and fix. **On this
particular synthetic test split, the embedding classifier had zero
misclassifications** (see §8 for why this is expected and not
celebratory), so `outputs/failure_analysis.md` is empty for this run.
The mechanism itself is fully implemented and was validated on
intermediate development splits, where it correctly surfaced confusions
between semantically adjacent clusters (e.g. the two
"Subscription Cancellation" sub-clusters noted in §3) with the
diagnosis "shared vocabulary, model relying on surface keywords" and the
fix "add contrastive examples distinguishing the two." **On the real
Kaggle dataset, this section will populate with genuine failure cases**
because real support text is far noisier and more ambiguous than the
templated synthetic sample.

## 7. Limitations

- **Synthetic data.** The headline numbers in §5 are computed on a
  400-row synthetic sample built from a small, fixed set of reply
  templates per intent — see §8 for the direct consequence on the
  classification numbers.
- **Offline fallbacks throughout.** Without network access,
  `sentence-transformers`, `faiss-cpu`, `nltk`, `rouge-score`,
  `bert-score`, `lightgbm`, and `openai` were unavailable. Every module
  substitutes a documented, dependency-free fallback (TF-IDF+SVD
  embeddings, numpy brute-force nearest-neighbor search, hand-rolled
  BLEU/ROUGE, logistic regression instead of LightGBM, template-based
  reply generation instead of a live LLM) — see decision log #5, #6,
  #11, #14, and `requirements.txt`. These fallbacks keep the *pipeline*
  correct and runnable but are lower-quality than the intended
  production stack.
- **Human annotation is a required review step.** The 200-row sheet is
   generated with model predictions prefilled and human-authored fields
   blank. The rubric is in `reports/golden_annotation_guide.md`; the
   pipeline preserves a completed sheet and computes agreement only from
   populated human labels.
- **No live LLM judge.** `OPENAI_API_KEY` was not configured in this
  environment, so LLM-as-Judge scores are reported as explicitly
  unavailable in `outputs/generation_metrics_summary.json`, not
  simulated.
- **Single-turn scope.** Only the first (customer, agent) pair per
  thread is modeled; longer resolution threads and multi-turn context
  are out of scope (decision log #2).

## 8. Misleading Headline Metric

**The 100% accuracy/F1 for both the TF-IDF and embedding classifiers in
§5 is a misleading headline metric, and calling it out is more
informative than reporting it uncritically.** It arises because the
synthetic dataset was generated from only 2–3 fixed reply templates per
intent (`data/generate_sample_data.py::INTENT_TEMPLATES`), which makes
the resulting clusters — and therefore the classification task — close
to trivially separable by vocabulary alone. A classifier that memorizes
"contains the word 'refund'" will score perfectly on this data and would
almost certainly **not** hold up on real, messy Twitter text, where the
same intent is expressed in dozens of ways, typos and slang are common,
and class boundaries are genuinely ambiguous (e.g., "my subscription
charged me twice" straddles Billing Problem and Refund). Similarly, the
0.94 BLEU/ROUGE for generation is inflated by the fact that the
synthetic templates recur across conversations, so retrieval frequently
finds a near-exact match to reuse. **The correct headline metric to
quote for this project is whatever the pipeline produces once re-run
against the real Kaggle dataset** (`data/twcs.csv`); the numbers in §5
should be read strictly as a correctness check on the code, not a
performance claim.

## 9. Future Work

1. Re-run the full pipeline against the real Kaggle `twcs.csv` across
   several candidate brands and report real numbers.
2. Install the full dependency stack (sentence-transformers, faiss-cpu,
   bert-score, lightgbm) and re-run — expect materially different
   (almost certainly lower, more realistic) classification and
   generation scores; compare against this report's synthetic numbers as
   a sanity check that the fallbacks approximate, but don't match, the
   production stack.
3. Have two human annotators independently label
   `golden_eval_annotation_sheet.csv` using
   `reports/golden_annotation_guide.md`, configure the live judge, and
   report the resulting Cohen's Kappa.
4. Tune the intent cluster count (k) using a silhouette-score sweep
   instead of the fixed default of 8 — §3 noted one real over-clustering
   artifact (two "Subscription Cancellation" clusters) that a proper
   sweep would likely resolve.
5. Extend conversation pairing to full multi-turn threads, and evaluate
   whether reply quality improves with thread-level context.
6. Add a lightweight PII redaction step before any message reaches a
   third-party LLM API, given customer messages sometimes include order
   numbers, emails, or phone numbers.
7. Replace the keyword-rule escalation layer's fixed pattern list with a
   periodically-retrained classifier trained on human escalation
   decisions from the (now-labeled) golden set, keeping the rule layer as
   a safety-net ceiling rather than the sole mechanism.
