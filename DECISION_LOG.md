<!-- # Decision Log

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
15. **Headline metric is not treated as production readiness** — historical public Twitter data has selection and distribution biases. -->




# Engineering Decision Log

This document records the major engineering and product decisions made while building the AppleSupport AI customer-support agent.

---

## Decision 1 — Brand Selection: AppleSupport

**Decision:** Use `AppleSupport` as the target brand.

**Reason:**  
AppleSupport has a large number of customer-support conversations in the Twitter customer-support dataset. This provides enough historical examples for intent discovery, retrieval, response drafting, and evaluation.

**Trade-off:**  
A single-brand system is less general than a multi-brand system, but it allows the support behavior and intent taxonomy to be specialized and evaluated more carefully.

---

## Decision 2 — Reconstruct Customer → Brand Reply Pairs

**Decision:** Reconstruct conversations by matching customer tweets with AppleSupport response tweet IDs.

**Reason:**  
The raw dataset stores tweet relationships through `response_tweet_id` and `in_response_to_tweet_id`. Using these relationships gives us actual historical support interactions rather than treating tweets as independent text samples.

**Trade-off:**  
The reconstruction pipeline is more complicated than simply filtering AppleSupport tweets, but it provides much better training and retrieval evidence.

---

## Decision 3 — Use Historical Conversations as Retrieval Evidence

**Decision:** Retrieve similar historical customer-support conversations before generating a response.

**Reason:**  
Customer-support replies should be grounded in how the brand historically handled similar issues. Retrieval provides concrete examples instead of relying only on the language model's general knowledge.

**Trade-off:**  
Retrieval adds latency and can return irrelevant examples. Therefore similarity thresholds are used before automatically handling a request.

---

## Decision 4 — Use TF-IDF for the Initial Retrieval System

**Decision:** Use TF-IDF retrieval as the default local retrieval method.

**Reason:**  
TF-IDF is deterministic, lightweight, easy to inspect, and does not require an external vector database or embedding API. It also makes the take-home project easy to reproduce on a fresh machine.

**Trade-off:**  
TF-IDF is weaker than modern semantic embeddings for paraphrases. A future version could use sentence embeddings or a vector database after establishing a reliable baseline.

---

## Decision 5 — Define a Small AppleSupport-Specific Intent Taxonomy

**Decision:** Use 12 intents:

1. `ios_update_issue`
2. `battery_issue`
3. `device_performance`
4. `screen_display_issue`
5. `app_issue`
6. `icloud_issue`
7. `camera_issue`
8. `connectivity_issue`
9. `account_security`
10. `hardware_repair`
11. `product_information`
12. `general_support`

**Reason:**  
The intents represent recurring support themes observed in the AppleSupport conversations while keeping the classification problem manageable.

**Trade-off:**  
A small taxonomy improves consistency but can lose detail. `general_support` is retained as a fallback for messages that do not clearly belong to a specialized category.

---

## Decision 6 — Use Rule-Based Intent Classification

**Decision:** Use a transparent keyword/rule-based classifier for the initial intent model.

**Reason:**  
The assignment emphasizes building and evaluating a working system. Rules are fast, deterministic, easy to debug, and make the intent decision explainable.

**Trade-off:**  
Rules do not understand language as deeply as a trained classifier or LLM. The evaluation set and failure analysis can reveal where this approach needs improvement.

---

## Decision 7 — Use Groq for Response Generation

**Decision:** Use a Groq-hosted language model to draft support replies.

**Reason:**  
Groq provides fast inference and is convenient for generating responses during development and evaluation.

**Trade-off:**  
The system depends on an external API and its rate limits. A deterministic fallback response is therefore retained when generation fails.

---

## Decision 8 — Use `openai/gpt-oss-20b` as the Current Generation Model

**Decision:** Use `openai/gpt-oss-20b` through Groq as the current model.

**Reason:**  
The previously configured model was deprecated. The replacement model provides a currently supported option for response generation and LLM judging.

**Trade-off:**  
Model behavior can change over time, so the exact model name and evaluation configuration are recorded for reproducibility.

---

## Decision 9 — Add a Safe Fallback When LLM Generation Fails

**Decision:** Return a deterministic support template when LLM generation fails.

**Reason:**  
API errors, rate limits, or temporary service failures should not make the complete support pipeline unusable.

**Trade-off:**  
Fallback responses are less personalized than generated responses, but they improve system reliability.

---

## Decision 10 — Separate Auto-Handle and Escalate Decisions

**Decision:** The agent can either `auto_handle` or `escalate`.

**Reason:**  
Not every customer-support request should be answered automatically. Requests with low classification confidence, weak retrieval evidence, security concerns, or hardware-repair requirements have higher risk.

**Auto-handle conditions include:**
- sufficiently high intent confidence
- relevant historical evidence
- acceptable retrieval similarity
- no high-risk escalation category

**Trade-off:**  
This conservative policy may escalate some requests that could have been answered automatically, but it reduces the risk of confidently giving poor support.

---

## Decision 11 — Escalate Account Security Issues

**Decision:** Always escalate `account_security` cases.

**Reason:**  
Account and security-related requests can involve sensitive customer information and potentially high-impact actions. The system should avoid pretending it can safely perform account-level operations.

**Trade-off:**  
Some simple security questions could theoretically be answered automatically, but conservative routing is safer.

---

## Decision 12 — Escalate Hardware Repair Issues

**Decision:** Always escalate `hardware_repair` cases.

**Reason:**  
Hardware failures may require physical inspection, repair logistics, warranty verification, or other actions that cannot be reliably completed through a text-only automated agent.

**Trade-off:**  
Some hardware questions are informational and could be handled automatically, but escalation provides a safer default.

---

## Decision 13 — Use a Retrieval Similarity Threshold

**Decision:** Require a minimum historical-evidence similarity before automatically handling a request.

**Reason:**  
A generated response without relevant historical evidence can become generic or unsupported. The similarity threshold acts as a grounding gate.

**Trade-off:**  
A strict threshold can increase escalation rates, while a low threshold can allow weak evidence. The threshold should therefore be tuned using the golden evaluation set.

---

## Decision 14 — Build a Human-Labeled Golden Set

**Decision:** Create a 200-example golden set with human labels.

**Labels include:**
- intent
- action
- reply quality
- notes

**Reason:**  
Automated metrics alone cannot establish whether the support agent is actually useful. A human-labeled evaluation set provides a fixed benchmark for classification, routing, and response quality.

**Trade-off:**  
Manual labeling requires time, but it provides much stronger evidence than evaluating only on automatically generated labels.

---

## Decision 15 — Compare Against Two Baselines

**Decision:** Evaluate the system against:

1. Majority-class baseline
2. TF-IDF + Logistic Regression classifier

**Reason:**  
A system's performance is meaningful only relative to simpler alternatives. The majority baseline provides a very simple lower bar, while TF-IDF + Logistic Regression provides a stronger traditional machine-learning baseline.

**Trade-off:**  
These baselines do not generate full support responses, so they mainly provide useful comparison for intent classification.

---

## Decision 16 — Use an LLM Judge for Response Quality

**Decision:** Use an LLM judge to score generated replies on:

- helpfulness
- groundedness
- correctness
- overall quality

Each dimension uses a 1–5 scale.

**Reason:**  
Response quality is difficult to measure with simple string-overlap metrics. An LLM judge provides structured qualitative evaluation while keeping the scoring process reproducible.

**Trade-off:**  
LLM judges can have their own biases and may disagree with human evaluators. Therefore human agreement is measured rather than treating the LLM judge as ground truth.

---

## Decision 17 — Measure Human–LLM Judge Agreement

**Decision:** Use exact agreement and weighted Cohen's kappa between human quality ratings and LLM judge overall ratings.

**Reason:**  
The reply-quality labels are ordinal 1–5 ratings. Weighted Cohen's kappa is appropriate because a disagreement between 4 and 5 is less severe than a disagreement between 1 and 5.

**Trade-off:**  
Agreement statistics are unstable with very small evaluation sets. Final agreement numbers should therefore be reported only after the larger golden set is labeled.

---

## Decision 18 — Track Failure Modes Instead of Only Headline Metrics

**Decision:** Analyze the top five failure modes in addition to reporting aggregate metrics.

**Reason:**  
A single accuracy or F1 number can hide important system weaknesses. Failure analysis identifies where the agent needs improvement and makes the evaluation more actionable.

**Examples of potential failure categories:**
- ambiguous intent
- weak retrieval evidence
- incorrect escalation
- unsupported response claims
- overly generic responses

---

## Decision 19 — Prefer Conservative Automation

**Decision:** When evidence or confidence is insufficient, escalate instead of generating a highly confident answer.

**Reason:**  
In customer support, a wrong confident answer can be more harmful than escalating a request to a human agent.

**Trade-off:**  
This can reduce the percentage of conversations handled automatically, but it prioritizes reliability over maximum automation.

---

## Decision 20 — Keep Evaluation Separate From Development Claims

**Decision:** Treat the current small labeled sample as preliminary and use the final larger golden set for reported metrics.

**Reason:**  
The current development labels are useful for debugging, but a small sample can produce unstable metrics and misleading headline numbers.

**Final reporting rule:**  
Final evaluation metrics should be generated after completing the required human-labeled golden set.

---