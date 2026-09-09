Fill `golden.csv` after generation.

Recommended labelling protocol:
- Label intent from the customer's message plus available thread context.
- Label `auto_handle` only when a reasonable automated response can be safely drafted from historical evidence.
- Label `escalate` for ambiguity, sensitive account/payment/refund issues without sufficient evidence, or cases requiring human judgement.
- Score reply quality from 1–5 using correctness, evidence-grounding, relevance, actionability and tone.
- Keep a short note for ambiguous examples.

Sampling note: the scaffold samples with a fixed random seed for reproducibility. For the final submission, report the exact sampling procedure and any stratification used.
