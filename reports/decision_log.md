# Decision Log

Fifteen engineering decisions made while building this pipeline, with the
reasoning and trade-offs behind each.

---

### 1. Automatic brand selection via a composite score, not a hard-coded name
**Decision:** Rank candidate brand accounts by conversation volume,
text cleanliness, and vocabulary richness; pick the top composite score.
**Why:** Makes STEP 1's "explain why this brand is selected" requirement
auditable rather than an arbitrary choice, and the same code re-selects
sensibly if run against a different dataset slice.
**Trade-off:** The weighting (0.5/0.3/0.2) is a reasonable heuristic, not
a tuned/validated objective — documented as such rather than presented
as optimal.

### 2. Conversation pairing via `response_tweet_id`, first-response only
**Decision:** For multi-turn threads, only pair the customer's message
with the *first* agent response.
**Why:** Keeps the unit of modeling simple (single-turn intent + reply),
matching the assignment's scope; multi-turn dialogue state tracking is
out of scope for an internship-level deliverable.
**Trade-off:** Loses information from longer resolution threads — noted
as future work.

### 3. FAISS over BM25 for retrieval
**Decision:** Use FAISS `IndexFlatIP` (cosine similarity via normalized
inner product) over embeddings, not BM25 over raw tokens.
**Why:** Customer messages paraphrase the same underlying issue in many
different words ("card charged twice" vs "billed 2x by mistake"); dense
semantic retrieval catches this, BM25's exact-token overlap often
doesn't. FAISS specifically (vs. a naive loop) because it scales
sub-linearly to the full ~1M-conversation dataset while staying exact at
smaller sizes — no premature accuracy/speed trade-off.
**Trade-off:** Dense retrieval is more expensive to build/index and less
interpretable than BM25's term-overlap scores.

### 4. Sentence-Transformers as the primary embedding backbone
**Decision:** `all-MiniLM-L6-v2` — small, fast, strong general-purpose
sentence embedding model.
**Why:** Good accuracy/latency trade-off for short (tweet-length) text
and runs on CPU, which matters for reproducibility without requiring a
GPU.
**Trade-off:** A domain-fine-tuned or larger model (e.g. `all-mpnet-base-v2`)
would likely improve retrieval/clustering quality at higher latency —
listed as future work.

### 5. Offline fallback: TF-IDF + TruncatedSVD in place of sentence embeddings
**Decision:** When `sentence-transformers` or its model weights can't be
loaded (e.g. no network), fall back to TF-IDF→SVD "pseudo-embeddings"
with an identical `.encode()` interface (`src/embeddings.py`).
**Why:** Keeps the *pipeline* runnable and testable end-to-end in
constrained environments (exactly the situation this project was built
in) instead of hard-failing.
**Trade-off:** LSA vectors capture far less semantic nuance than a
transformer embedding — clearly logged and called out in the report so
downstream metrics aren't over-trusted.

### 6. Logistic Regression as the default final-model head, LightGBM optional
**Decision:** Default the "final model" classifier head to Logistic
Regression over embeddings, with a `use_lightgbm=True` flag to swap in
LightGBM when there's enough data.
**Why:** On a few hundred to few thousand examples per intent,
LightGBM's flexibility mostly buys overfitting risk, not accuracy;
Logistic Regression is also faster, more stable across reruns, and its
coefficients are directly inspectable for debugging.
**Trade-off:** LightGBM would likely pull ahead on the full dataset
(millions of rows) where its capacity is actually useful — hence keeping
it as an opt-in, not a removal.

### 7. Unsupervised intent discovery via KMeans + auto-labeling from TF-IDF terms
**Decision:** Cluster message embeddings with KMeans, then auto-generate
a human-readable label per cluster from its top TF-IDF terms (via a
curated keyword→label lookup, falling back to "top-term" naming).
**Why:** The dataset has no ground-truth intent labels; this satisfies
STEP 2's "discover intents from the data" requirement in a way that's
inspectable (you can see exactly which terms produced each label) rather
than an opaque black box.
**Trade-off:** KMeans assumes roughly spherical, similarly-sized
clusters, which is a simplification for real support data with skewed
category sizes — HDBSCAN is a candidate upgrade, noted in future work.

### 8. This intent taxonomy (Refund / Order Delay / Account Locked / etc.)
**Decision:** Keep the taxonomy at 6–10 labels, matching the assignment
spec, rather than a much finer-grained taxonomy.
**Why:** Coarse-enough labels get enough training examples per class to
be learnable; fine-grained taxonomies fragment an already modest sample
size and hurt classifier reliability.
**Trade-off:** Coarse labels lose nuance (e.g. "Refund" doesn't
distinguish "never received item" from "wrong item sent") — acceptable
for a v1, flagged as a future refinement.

### 9. Escalation is rule-based first, LLM-reasoning second, "escalate" is the default on uncertainty
**Decision:** Deterministic keyword rules run first (billing disputes,
fraud, legal, cancellation, strong negative sentiment); only when no
rule fires does an optional LLM layer weigh in; and when nothing is
confident, the system defaults to escalate rather than auto-handle.
**Why:** For a safety-relevant gate like this, false negatives (silently
auto-handling something that needed a human) are worse than false
positives (a human reviews something that didn't strictly need it).
Rule-based-first also keeps the decision auditable — you can point to
exactly which regex fired.
**Trade-off:** This intentionally biases toward more human involvement
than a pure cost-minimization policy would — a deliberate, stated
trade-off in favor of safety over automation rate.

### 10. RAG generation grounded strictly in retrieved historical replies
**Decision:** The generation prompt/template only allows the model to
draw on retrieved historical replies for policy facts, and low-similarity
retrievals trigger an explicit "low confidence, please verify" caveat in
the output text.
**Why:** Directly satisfies STEP 4's "never hallucinate policies... mention
uncertainty when needed" requirement, and keeps every generated claim
traceable to a specific historical example (`grounding_agent_tweet_ids`).
**Trade-off:** Strict grounding can make replies feel formulaic /
repetitive compared to a fully generative model with more creative
license — an acceptable trade-off for a support context where
correctness matters more than novelty.

### 11. Template-based generation fallback instead of skipping generation when no LLM key is present
**Decision:** When no `OPENAI_API_KEY` is configured, reuse the
single most-similar historical reply's text (with an uncertainty caveat
appended if applicable) instead of a live LLM call.
**Why:** Keeps the pipeline fully runnable offline, and — because it's
built from the *same* retrieved, grounded examples the LLM path would
use — it doesn't introduce a different hallucination risk profile.
**Trade-off:** Less fluent / less tailored to the specific customer
message than a real LLM rewrite — clearly logged (`backend: "template_fallback"`
on every generated reply) so it's never confused with live-LLM output.

### 12. Stratified (not uniform) sampling for the golden evaluation set
**Decision:** Sample the 200-row golden set proportionally within each
predicted intent, not uniformly at random over all messages.
**Why:** Pure random sampling under-represents rare intents exactly
where classifier/escalation trust matters most; stratification keeps
every intent visible in the eval set.
**Trade-off:** Slightly over-represents rare intents relative to their
true frequency in production traffic — worth correcting for with
frequency-weighting if this eval set is used to estimate an overall
production accuracy number.

### 13. These specific evaluation metrics (BLEU/ROUGE/BERTScore/LLM-judge, Accuracy/F1/Kappa)
**Decision:** Pair lexical-overlap metrics (BLEU, ROUGE) with a
semantic metric (BERTScore) and a rubric-based LLM judge, rather than
relying on any single metric.
**Why:** BLEU/ROUGE alone reward near-verbatim copying (which our
grounded/template generation can trivially "win" at) and would give a
misleading picture on their own; BERTScore catches paraphrases;
LLM-judge catches things n-gram metrics structurally can't
(groundedness, safety, tone). Cohen's Kappa between human and LLM-judge
labels is the standard way to know whether the LLM judge can be trusted
as a proxy for human review at all.
**Trade-off:** More metrics to compute and reconcile, and BERTScore/LLM
judge both require heavier dependencies / API access that may not
always be available (as in this sandboxed build).

### 14. Dependency-free fallbacks for every metric, rather than skipping evaluation when a package is missing
**Decision:** `src/evaluation.py` implements minimal pure-Python BLEU and
ROUGE fallbacks used only when `nltk`/`rouge-score` aren't installed;
BERTScore and LLM-judge instead report `None`/"unavailable" rather than
being faked.
**Why:** BLEU/ROUGE have well-defined, reproducible formulas that are
feasible to hand-implement faithfully; BERTScore and LLM-judge inherently
require a specific pretrained model / API that cannot be faithfully
approximated, so honesty requires reporting them as unavailable instead.
**Trade-off:** The hand-rolled BLEU/ROUGE fallback is a simplified
implementation (e.g. no explicit brevity-penalty edge case handling
beyond the basics) — real NLTK/rouge-score should be used for any
published numbers, which is why both are still listed in
`requirements.txt` as the preferred path.

### 15. Config as a single dataclass instead of scattered constants / YAML
**Decision:** All tunable parameters (cluster count, test split, top-k,
thresholds, model names) live in one `Config` dataclass in `src/utils.py`,
serialized to `outputs/run_manifest.json` on every run.
**Why:** One discoverable place for every knob, and a guaranteed audit
trail of exactly which config produced a given set of outputs — useful
both for reproducibility and for debugging "why did this run give
different numbers."
**Trade-off:** A YAML/Hydra-based config would scale better to many
experiment variants / sweeps — unnecessary complexity at this project's
scale, noted as a natural upgrade path if the project grows.
