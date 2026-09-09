# Hiver SDE Intern — AI Support Agent

A reproducible AI customer-support agent built for the Hiver SDE Intern take-home assignment.

## What it does

For one selected brand, the pipeline:

1. Cleans/reconstructs noisy Twitter support conversations.
2. Builds a small intent taxonomy from the data.
3. Retrieves historically similar resolved conversations using embeddings + TF-IDF fallback.
4. Drafts a grounded support reply.
5. Decides `auto_handle` vs `escalate`, with a reason.
6. Evaluates intent, retrieval, reply quality and escalation decisions.
7. Compares against two baselines.
8. Supports a 150–250 example hand-labelled golden set.

## 15-minute reproduction

### 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env`.

The pipeline works without an LLM API by using a deterministic template responder, so evaluation is reproducible. For the full agent, set `GROQ_API_KEY` and optionally `GROQ_MODEL`.

### 3. Prepare data

Download the Kaggle dataset `thoughtvector/customer-support-on-twitter` and put the CSV in `data/raw/`.

Expected filename can be any `.csv`; the loader auto-detects common column names such as `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id`.

Then run:

```bash
python -m src.pipeline --input data/raw/twcs.csv --brand AppleSupport --sample 50000
```

If your CSV has a different name:

```bash
python -m src.pipeline --input data/raw/YOUR_FILE.csv --brand YOUR_BRAND --sample 50000
```

### 4. Build a golden set

```bash
python -m src.make_golden --input data/processed/conversations.csv --output data/golden_set/golden.csv --n 200
```

Edit the generated CSV and fill the `gold_intent`, `gold_action`, and `gold_reply_quality` columns. The sampling is stratified where possible.

### 5. Run evaluation

```bash
python -m src.evaluate \
  --data data/processed/conversations.csv \
  --golden data/golden_set/golden.csv \
  --output artifacts/evaluation.json
```

### Example agent call

```bash
python -m src.agent --brand AppleSupport --message "My refund still hasn't arrived"
```

## Output contract

```json
{
  "intent": "refund_status",
  "confidence": 0.86,
  "action": "escalate",
  "reason": "The message concerns an unresolved financial issue and the retrieved evidence does not establish a safe automated resolution.",
  "reply": "I'm sorry you're still waiting for the refund. I can help check the status, but because the available information does not confirm the current refund state, this should be reviewed by a support specialist.",
  "evidence": [
    {"conversation_id": "123", "similarity": 0.82}
  ]
}
```

## Evaluation design

### Baseline 1 — Majority intent
Predicts the most frequent intent in the evaluation split.

### Baseline 2 — TF-IDF + Logistic Regression
A simple lexical classifier trained only on the training portion.

### Agent
Hybrid intent classifier + embedding retrieval + grounded generation + explicit escalation policy.

## What "good" means

For customer support, correctness and groundedness matter more than eloquence:

- Intent: Macro-F1, because rare support intents matter.
- Retrieval: Recall@5.
- Reply: factual grounding, relevance, actionability and tone.
- Escalation: high recall for risky/ambiguous cases; false automation is more costly than unnecessary escalation.
- Overall: percentage of examples where intent, reply and routing are all acceptable.

## Mandatory limitation

The headline score can be misleading because historical Twitter conversations are not an unbiased sample of future support traffic. They over-represent public complaints, successful interactions, specific brands, and issues that were actually posted. A strong score on this set does not prove safety on unseen private tickets, novel policies, or rapidly changing product rules.

## Failure analysis

After evaluation, inspect the five largest error clusters. Typical failure hypotheses to test:

1. Multiple intents in one message.
2. Short/ambiguous messages with little context.
3. New policies not represented in historical data.
4. Retrieval of lexically similar but operationally different cases.
5. Escalation mistakes caused by overconfidence.

Each failure should include the input, prediction, expected behaviour, retrieved evidence and a proposed fix.

## Decision log

See `DECISION_LOG.md`.

## Citation

Primary dataset: Customer Support on Twitter, Kaggle dataset by thoughtvector. See the Kaggle dataset page for the original data and license/terms before redistribution.

This repository contains code and a small evaluation scaffold, not a redistribution of the full dataset.
