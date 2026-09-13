# AI Customer Support Agent — Customer Support on Twitter

An end-to-end pipeline that (1) classifies customer support messages into
data-driven intent labels, (2) generates brand-grounded replies via
retrieval-augmented generation, and (3) decides whether an issue can be
auto-handled or must be escalated to a human — built on the Kaggle
[`thoughtvector/customer-support-on-twitter`](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset.

> **Read this first:** this build was authored in a sandboxed environment
> with no network access. It ships with a small, clearly-labeled
> **synthetic** dataset (`data/sample_twcs.csv`, same schema as the real
> Kaggle file) so the whole pipeline runs out of the box, and every
> module gracefully degrades to a dependency-free fallback when a
> package (sentence-transformers, faiss, nltk, rouge-score, bert-score,
> lightgbm, openai) isn't installed. **Numbers produced against the
> synthetic sample are illustrations of the pipeline working, not real
> model performance** — see `reports/report.md` → "Misleading headline
> metric" for a worked example of exactly this trap. Swap in the real
> dataset (steps below) to get real results.

---

## Quickstart (reproduce in under 15 minutes)

### 1. Installation

```bash
git clone https://github.com/SV-AR/customer-support-agent.git
cd customer-support-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m nltk.downloader punkt   # only needed for the NLTK BLEU backend
```

If you skip installing `sentence-transformers` / `faiss-cpu` / `nltk` /
`rouge-score` / `bert-score` / `lightgbm` / `openai`, the pipeline still
runs — see the note at the top of `requirements.txt`.

### 2. Dataset download (to use the REAL data)

1. Download `twcs.csv` from
   https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
   (requires a free Kaggle account / API token).
2. Place it at `data/twcs.csv`.
3. That's it — `src/preprocessing.py` automatically prefers
   `data/twcs.csv` over the synthetic sample when present; no config
   changes needed.

To keep exploring without the real dataset, regenerate the synthetic
sample any time with:

```bash
python3 data/generate_sample_data.py
```

### 3. Run everything

```bash
python3 run_pipeline.py
```

This single command runs STEPs 1–9 (preprocessing → brand selection →
intent discovery → classification → retrieval → reply generation →
escalation → golden sampling → evaluation → failure analysis) and writes
every artifact to `outputs/`. Runtime on the synthetic sample is a few
seconds; on the full ~2.8M-row Kaggle file, expect embedding + FAISS
indexing to dominate runtime (tens of minutes on CPU, depending on how
many conversations you keep after brand filtering).

### 4. Inspect outputs

| File | What it is |
|---|---|
| `outputs/run_summary.json` | Brand selected, dataset sizes, escalation rate |
| `outputs/discovered_intents.csv` | STEP 2: cluster → label → top TF-IDF terms |
| `outputs/model_comparison.csv` / `.png` | STEP 3/9: random vs TF-IDF vs embedding classifier |
| `outputs/confusion_matrix_embedding_classifier.png` | STEP 3 |
| `outputs/test_predictions_full.csv` | Per-example intent, generated reply, escalation decision |
| `outputs/golden_eval_annotation_sheet.csv` | STEP 6: 200-row spreadsheet for human annotation |
| `reports/golden_annotation_guide.md` | Annotation rubric and agreement protocol |
| `outputs/generation_metrics_summary.json` | STEP 7: BLEU / ROUGE / BERTScore / LLM-judge availability |
| `outputs/failure_analysis.md` | STEP 8: top failure modes with cause + fix |
| `reports/report.md` | STEP 10: 6-page write-up |
| `reports/decision_log.md` | STEP 12: engineering decisions |

### 5. (Optional) Enable live LLM generation + judging

```bash
export OPENAI_API_KEY=sk-...
```

Then in `run_pipeline.py`, set `Config(use_live_llm=True)`. Without a
key, reply generation uses a deterministic, style-matched template
fallback (see `src/reply_generator.py`) and LLM-judge scores are
reported as explicitly unavailable rather than fabricated.

### 6. Complete the golden evaluation

The pipeline samples 200 examples by predicted-intent stratum and writes a
review sheet. Have two reviewers independently complete `intent`,
`correct_reply`, `reply_quality`, `auto/escalate`, and `notes` using
`reports/golden_annotation_guide.md`. Re-running the pipeline preserves a
complete sheet rather than overwriting it. With `OPENAI_API_KEY` configured
and `Config(use_live_llm=True)`, the pipeline runs the judge and writes the
human-versus-judge Cohen's Kappa to `outputs/human_llm_agreement.json`.

---

## Folder structure

```
customer-support-agent/
├── data/
│   ├── generate_sample_data.py   # synthetic stand-in generator (see note above)
│   └── sample_twcs.csv           # synthetic sample, schema-identical to twcs.csv
├── notebooks/
│   └── eda.ipynb                 # exploratory data analysis (EDA only; no pipeline logic)
├── src/
│   ├── preprocessing.py          # STEP 1: load, select brand, build conversation pairs
│   ├── embeddings.py             # shared embedding backend (+ offline fallback)
│   ├── intent_classifier.py      # STEP 2/3: clustering + 3-tier classifiers
│   ├── retrieval.py              # STEP 4: FAISS index + search
│   ├── reply_generator.py        # STEP 4: RAG reply generation
│   ├── escalation.py             # STEP 5: rule-based + LLM escalation decisions
│   ├── sampling.py               # STEP 6: golden set sampling + annotation sheet
│   ├── evaluation.py             # STEP 3/7/8: all metrics + failure analysis
│   └── utils.py                  # logging, config, seeding, I/O helpers
├── reports/
│   ├── report.md                 # STEP 10: 6-page project report
│   └── decision_log.md           # STEP 12: engineering decision log
├── outputs/                      # generated by run_pipeline.py (see table above)
├── run_pipeline.py                # single entry point, runs STEPs 1–9
├── requirements.txt
└── README.md
```

## Example output (from the synthetic sample)

```
[tfidf_logreg]          accuracy=1.000  f1_macro=1.000
[embedding_classifier]  accuracy=1.000  f1_macro=1.000
[random_baseline]       accuracy=0.275  f1_macro=0.245
Escalation rate on test set: 72.5%
```

The 100% scores are a direct consequence of the synthetic data being
generated from a small, fixed set of templates per intent (making
clusters trivially separable) — **not evidence that the approach
achieves perfect accuracy on real, messy Twitter text.** This exact
pattern is discussed as the report's "misleading headline metric" case
study; real-dataset numbers will be meaningfully lower and are the
numbers that should be quoted in any evaluation of this project.

## Known limitations of this sandboxed build

- Real Kaggle dataset was not downloadable (no network) → synthetic
  sample used; see warnings above.
- `sentence-transformers`, `faiss-cpu`, `nltk`, `rouge-score`,
  `bert-score`, `lightgbm`, `openai` were not installable → dependency-free
  fallbacks used automatically, logged clearly at runtime.
- The checked-in golden evaluation sheet is an annotation instrument until
  two human reviewers complete it. Blank human fields are never treated as
  labels, and Cohen's Kappa remains `null` until both human labels and live
  LLM-judge outputs exist.
- LLM-as-Judge scores require `OPENAI_API_KEY`; without it,
  `generation_metrics_summary.json` reports `llm_judge_scores_available: false`
  with an explanatory note instead of invented scores.

All of the above are explained in more depth, with their implications
for interpreting results, in `reports/report.md`.
