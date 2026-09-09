# AppleSupport AI Customer Support Agent

An AI-powered customer-support agent built for the Hiver SDE Intern take-home assignment.

The system uses historical AppleSupport customer-support conversations to:

- classify incoming customer messages into support intents,
- retrieve relevant historical conversations,
- draft grounded support replies,
- and decide whether to auto-handle or escalate a request.

The main focus of this project is not only building the agent, but also evaluating its reliability and understanding where it can fail.

---

# 1. Problem

Customer-support teams receive a large number of repetitive requests.

This project builds an AI support agent for `AppleSupport` that can:

1. Understand the customer's intent.
2. Find similar historical support interactions.
3. Draft a useful response using historical evidence.
4. Decide whether the request is safe to handle automatically.
5. Escalate uncertain or higher-risk cases.

---

# 2. Dataset

This project uses the Twitter Customer Support dataset.

Important fields include:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

The dataset contains conversations between customers and brands on Twitter.

For this project, only `AppleSupport` conversations are used.

The pipeline reconstructs customer → AppleSupport response pairs using the tweet relationship fields.

Current processed development dataset:

```text
50,000 customer → historical reply pairs
3. System Architecture
Customer Message
       |
       v
+---------------------+
| Intent Classification|
+---------------------+
       |
       v
+---------------------+
| Historical Retrieval|
|      TF-IDF         |
+---------------------+
       |
       v
+---------------------+
| Evidence + Policy   |
|       Checks        |
+---------------------+
       |
       +----------------+
       |                |
       v                v
 Auto Handle        Escalate
       |
       v
+---------------------+
|     Groq LLM        |
| Response Generation |
+---------------------+
       |
       v
 Final Support Reply
4. Intent Taxonomy

The system uses 12 AppleSupport-specific intents.

Intent	Description
ios_update_issue	iOS update and installation problems
battery_issue	Battery drain, charging and battery-life problems
device_performance	Slow, freezing or unstable device behavior
screen_display_issue	Screen, display and visual problems
app_issue	Problems with applications
icloud_issue	iCloud synchronization and storage issues
camera_issue	Camera and photo-related problems
connectivity_issue	Wi-Fi, Bluetooth, cellular and connectivity issues
account_security	Account access, phishing and security concerns
hardware_repair	Physical damage and repair requests
product_information	Product information and compatibility questions
general_support	Other AppleSupport requests

The taxonomy was intentionally kept small so that classification remains interpretable and measurable.

5. Intent Classification

The current classifier is a transparent rule-based classifier.

It uses intent-specific keywords and patterns extracted from recurring support themes.

Example:

"My iPhone battery is draining very quickly"
                |
                v
          battery_issue

Another example:

"My WiFi keeps disconnecting"
                |
                v
       connectivity_issue
Why a rule-based classifier?

The initial classifier was chosen because it is:

deterministic,
fast,
easy to debug,
interpretable,
and simple to reproduce.

A future version can replace it with a supervised classifier or semantic model after collecting more labeled data.

6. Historical Retrieval

After intent classification, the system searches historical AppleSupport conversations for relevant examples.

The current retriever uses TF-IDF similarity.

Each retrieved example contains:

Customer message
+
Historical AppleSupport reply
+
Similarity score

The top historical examples are passed to the response-generation stage as evidence.

Example:

Customer:
"My iPhone battery is draining very quickly"

Retrieved historical examples:

1. Similarity: 0.82
   Battery-related customer message
   Historical AppleSupport reply

2. Similarity: 0.81
   Battery-related customer message
   Historical AppleSupport reply

The retrieval threshold is also used by the routing policy.

7. Response Generation

The response generator uses a Groq-hosted language model.

Current model:

openai/gpt-oss-20b

The model receives the customer message and relevant historical support evidence.

The goal is to produce a response that is:

relevant,
helpful,
concise,
grounded in available evidence,
and appropriate for customer support.

If LLM generation fails because of an API error or rate limit, the system uses a deterministic fallback response.

8. Auto-Handle vs Escalate

The system does not automatically handle every request.

The routing policy considers:

intent confidence,
retrieval evidence,
retrieval similarity,
and risk-sensitive intent categories.
Escalation conditions

A request is escalated when:

intent confidence is too low,
historical evidence is unavailable,
retrieval similarity is too low,
intent is account_security,
or intent is hardware_repair.
Why?

Some support requests require human review.

For example:

Account Security
       |
       v
Human Support

and:

Hardware Repair
       |
       v
Human Support

This conservative strategy prioritizes reliability over maximum automation.

9. Example
Customer message
My iPhone battery is draining very quickly.
Agent output
Intent: battery_issue
Confidence: high
Action: auto_handle

The agent retrieves similar historical battery-support conversations and uses them as evidence for generating the response.

10. Evaluation Strategy

The system is evaluated using a manually labeled golden set.

Target golden set:

200 examples

Each example contains:

gold_intent
gold_action
gold_reply_quality
label_notes

The reply-quality label uses a 1–5 scale.

Intent evaluation

Metrics:

Accuracy
Macro F1
Per-intent precision, recall and F1
Routing evaluation

Metric:

Auto-handle / escalate accuracy
Response evaluation

Human:

1–5 reply-quality rating

LLM Judge:

Helpfulness
Groundedness
Correctness
Overall quality
11. Two Baselines

Two baselines are implemented.

Baseline 1 — Majority Class

Always predicts the most frequent intent.

This provides a simple lower-bound reference.

Baseline 2 — TF-IDF + Logistic Regression

A traditional supervised text-classification baseline:

TF-IDF
   +
Logistic Regression

This provides a stronger comparison against the rule-based intent classifier.

Final baseline metrics will be regenerated after completing the final golden set.

12. LLM Judge

The project includes an LLM-based response evaluator.

Each generated response is scored from 1–5 on:

Dimension	Scale
Helpfulness	1–5
Groundedness	1–5
Correctness	1–5
Overall Quality	1–5

The LLM judge also produces a short explanation for its rating.

Results are stored in:

data/judge_results.csv
13. Human–LLM Agreement

The LLM judge is not treated as ground truth.

Its overall-quality score is compared against human reply-quality labels.

The project calculates:

Exact agreement
Quadratic weighted Cohen's kappa

Weighted Cohen's kappa is suitable for ordinal 1–5 ratings because larger disagreements receive more penalty than small disagreements.

The current development set contains only 19 human-labeled examples, so the current agreement numbers are considered preliminary.

Final agreement statistics will be generated after completing the larger golden set.

14. Current Development Results

Current human-labeled examples:

19

Preliminary response-quality evaluation:

Human quality mean:       3.474
LLM judge overall mean:   3.526
Exact agreement:          0.211
Weighted Cohen's kappa:   0.045

These are development/smoke-test results only.

They are not presented as the final performance of the system because the required larger human-labeled golden set is still being completed.

15. What Is Misleading About My Headline Number?

A single headline metric can make the system appear stronger than it actually is.

For example, overall accuracy can be inflated when some intents are much more frequent than others.

Similarly, a high average response-quality score does not guarantee that the agent behaves safely on ambiguous or high-risk requests.

Therefore the final evaluation will report more than one number:

Accuracy
Macro F1
Per-intent performance
Routing accuracy
Response quality
Human–LLM agreement
Failure analysis

The main limitation of aggregate metrics is that they can hide poor performance on rare, ambiguous, or high-risk cases.

16. Top 5 Failure Modes

The main failure modes being tracked are:

1. Ambiguous Intent

Some customer messages do not contain enough information for confident classification.

2. Weak Retrieval

TF-IDF may retrieve textually similar conversations that are not actually relevant to the customer's problem.

3. Incorrect Routing

The system may auto-handle a case that should be escalated, or escalate a case that could safely be handled automatically.

4. Unsupported Claims

The LLM may generate advice that is reasonable but not directly supported by the retrieved historical evidence.

5. Generic Responses

When retrieval evidence is weak, the generated response can become generic and less useful.

These failure modes will be quantified using examples from the final golden-set evaluation.

17. Project Structure
AI/
│
├── configs/
│
├── data/
│   ├── golden_set/
│   │   └── golden.csv
│   ├── processed/
│   │   └── conversations.csv
│   ├── raw/
│   │   └── twcs.csv
│   ├── evaluation_results.json
│   ├── judge_results.csv
│   └── judge_test.csv
│
├── src/
│   ├── agent.py
│   ├── baselines.py
│   ├── data.py
│   ├── evaluate.py
│   ├── judge.py
│   ├── label_golden.py
│   ├── make_golden.py
│   ├── pipeline.py
│   ├── retrieval.py
│   └── taxonomy.py
│
├── tests/
│
├── .env
├── .gitignore
├── DECISION_LOG.md
├── README.md
└── requirements.txt
18. Installation

Python 3.10+ is recommended.

Create a virtual environment:

python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
19. Environment Variables

Create a .env file:

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

Never commit the real API key to GitHub.

Use .env.example for sharing the required environment-variable names.

20. Reconstruct Conversations

Run:

python -m src.pipeline --input data/raw/twcs.csv --brand AppleSupport --sample 50000

Expected output:

Found 106860 tweets from AppleSupport
Customer/reply pairs: 50000
Output: data\processed\conversations.csv
21. Run the AI Agent

Example:

python -m src.agent --brand AppleSupport --message "My iPhone battery is draining very quickly"

The agent returns:

intent,
confidence,
action,
reason,
generated reply,
retrieved evidence.
22. Create the Golden Set

Generate the evaluation sample:

python -m src.make_golden --input data/processed/conversations.csv --output data/golden_set/golden.csv --n 200

Start manual labeling:

python -m src.label_golden --input data/golden_set/golden.csv

The labeling script saves progress so labeling can be continued later.

23. Run Baselines
python -m src.baselines --golden data/golden_set/golden.csv

Run this after completing the final human-labeled golden set to obtain final baseline metrics.

24. Run Automated Evaluation
python -m src.evaluate --golden data/golden_set/golden.csv --data data/processed/conversations.csv

Output:

data/evaluation_results.json
25. Run LLM Judge
python -m src.judge --golden data/golden_set/golden.csv --data data/processed/conversations.csv --output data/judge_results.csv

Output:

data/judge_results.csv
26. Engineering Decisions

Major engineering decisions are documented in:

DECISION_LOG.md

The decision log covers:

brand selection,
conversation reconstruction,
intent taxonomy,
retrieval approach,
classifier design,
LLM selection,
escalation policy,
golden-set design,
baselines,
LLM judging,
and failure analysis.
27. Limitations

Current limitations include:

The intent classifier is rule-based.
TF-IDF retrieval has limited semantic understanding.
Response generation depends on an external LLM API.
API rate limits can affect large evaluation runs.
Historical support replies may contain outdated information.
Historical examples are not guaranteed to represent the ideal modern support response.
Aggregate metrics can hide failures on difficult or rare cases.
The current development evaluation is small.

These limitations are considered when interpreting the final results.

28. Next-Week Improvement Plan

If development continued for another week:

Day 1–2 — Improve Intent Classification

Train and evaluate a supervised classifier using the completed golden set.

Day 3 — Improve Retrieval

Compare TF-IDF with sentence embeddings and evaluate retrieval quality.

Day 4 — Grounding Evaluation

Measure whether generated claims are actually supported by retrieved historical evidence.

Day 5 — Safer Response Generation

Add stronger grounding constraints and structured response validation.

Day 6 — Routing Optimization

Tune escalation thresholds using the golden set and analyze false auto-handles.

Day 7 — Error Analysis

Review difficult examples and update the taxonomy, retrieval strategy and escalation policy.

29. Reproducibility Checklist

A fresh setup should follow:

1. Create virtual environment
2. Install requirements
3. Configure GROQ_API_KEY
4. Place twcs.csv in data/raw/
5. Run conversation reconstruction
6. Run the agent
7. Create the golden set
8. Label the golden set
9. Run baselines
10. Run automated evaluation
11. Run LLM judge

The project is designed so that the core pipeline can be reproduced without a vector database.

30. Final Status
Dataset reconstruction       DONE
AppleSupport selection       DONE
Intent taxonomy              DONE
Intent classifier            DONE
Historical retrieval         DONE
Response generation          DONE
Escalation policy            DONE
Two baselines                DONE
LLM Judge                    DONE
Human–LLM agreement          DONE
Golden-set labeling          IN PROGRESS
Final evaluation             PENDING
Final failure analysis       PENDING
Final report                 PENDING
31. Summary

The project implements an end-to-end AI customer-support pipeline:

Raw Twitter Support Data
          ↓
Conversation Reconstruction
          ↓
Intent Classification
          ↓
Historical Retrieval
          ↓
Evidence-Based Response Generation
          ↓
Routing / Safety Policy
          ↓
Auto-Handle or Escalate
          ↓
Human + LLM Evaluation

The goal is not simply to produce an AI-generated answer.

The goal is to build a support system that can also demonstrate:

what it understands,
what evidence it uses,
when it should answer,
when it should escalate,
how well it performs,
and where it fails.


# AppleSupport AI Customer Support Agent

**Made by Rohan Karak**

> Hiver SDE Intern — Take-Home Assignment

An AI-powered customer-support agent built for the Hiver SDE Intern take-home assignment.