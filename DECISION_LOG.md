# Decision Log

1. **One brand only** — keeps the evaluation focused and matches the assignment.
2. **Conversation-level reconstruction** — a single tweet often lacks enough context to judge the correct reply.
3. **Small intent taxonomy** — fewer, data-derived classes are more reliable than forcing 77 generic intents onto one brand.
4. **Macro-F1** — prevents common intents from hiding poor performance on rare but important issues.
5. **Hybrid retrieval** — embeddings handle paraphrases while TF-IDF provides a transparent lexical fallback.
6. **Top-5 evidence** — enough context for generation without flooding the prompt.
7. **Grounded generation only** — the model is instructed to avoid inventing policy, refunds, timelines or account facts.
8. **Escalation is asymmetric** — uncertain/risky cases should prefer humans because an incorrect automated resolution can be costly.
9. **No Banking77 labels in final classification** — Banking77 may inspire intent methodology, but the brand taxonomy should come from the primary data.
10. **Golden set is hand-labelled** — automated labels would make the evaluation circular.
11. **Stratified sampling where possible** — ensures common and uncommon intents are represented.
12. **LLM judge is secondary evidence** — human agreement is required before trusting judge scores.
13. **No full-dataset requirement** — a fixed subsample makes the repository reproducible within the assignment's time constraint.
14. **Deterministic fallback** — the project can run without an API key, improving reproducibility.
15. **Headline metric is not treated as production readiness** — historical public Twitter data has selection and distribution biases.
