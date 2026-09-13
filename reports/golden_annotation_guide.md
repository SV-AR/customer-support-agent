# Golden Set Annotation Guide

The golden set contains 200 examples sampled with a fixed seed, stratified by
predicted intent. Annotators should label every row independently of the
model prediction, then record enough reasoning in `notes` for a second reviewer
to audit the decision.

## Fields

- `intent`: the best-supported intent for the customer's message. Use the
  existing predicted label when it is correct; otherwise write the corrected
  intent and explain the distinction in `notes`.
- `correct_reply`: a concise reply that addresses the customer's message
  without inventing a policy, price, deadline, or account action. It may say
  that a human must investigate when the historical evidence is insufficient.
- `reply_quality`: an integer from 1 to 5 for the generated reply: 1 harmful
  or unrelated, 2 materially wrong or unsafe, 3 partially useful, 4 correct
  and useful with minor omissions, 5 correct, useful, grounded, and safe.
- `auto/escalate`: choose `Auto Handle` only when the generated reply can be
  safely sent without account-specific investigation. Choose `Escalate` for
  billing, fraud, security, legal, access-recovery, or uncertain cases.
- `notes`: briefly cite the evidence for the quality and routing decisions.

## Agreement protocol

Two reviewers should label the same 200 rows independently. Resolve
disagreements only after the first-pass labels are exported. Set
`OPENAI_API_KEY` and `Config.use_live_llm=True` to run the LLM judge; the
pipeline then compares the human `reply_quality` bucket with the judge's
quality bucket and writes Cohen's Kappa. Blank or incomplete annotations are
reported as unavailable rather than converted into artificial scores.