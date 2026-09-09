# 💻 AppleSupport AI Customer Support Agent

**Made by Rohan Karak**  
🎯 **Hiver SDE Intern — Take-Home Assignment**

> An end-to-end AI customer-support agent that combines intent classification, historical-case retrieval, grounded response generation, and conservative human escalation.

---

## 🌟 Project Overview

Customer-support systems receive thousands of repetitive requests, but not every request should be handled automatically.

This project builds an **AI support agent for AppleSupport** using historical customer-support conversations from the Twitter Customer Support dataset.

The agent can:

- 🧠 Classify the customer's intent
- 🔎 Retrieve similar historical support conversations
- ✍️ Draft an evidence-grounded response
- 🛡️ Decide whether to auto-handle or escalate
- 📊 Evaluate classification, routing, and response quality
- 🖥️ Provide an interactive Streamlit demo

The focus is not only on generating a response, but also on showing **what the system understands, what evidence it uses, when it should escalate, and where it can fail**.

---

## ✨ Key Features

| Feature | Implementation |
|---|---|
| 🏷️ Intent Classification | Transparent rule-based classifier |
| 🔎 Historical Retrieval | TF-IDF similarity |
| 🤖 Response Generation | Groq + `openai/gpt-oss-20b` |
| 🛡️ Safety Routing | Auto-handle / Escalate policy |
| 📚 Evidence | Top historical customer-support cases |
| 📊 Evaluation | Golden set + automated metrics |
| ⚖️ Baselines | Majority Class + TF-IDF/Logistic Regression |
| 🧑‍⚖️ LLM Judge | Helpfulness, groundedness, correctness, quality |
| 🖥️ Demo UI | Streamlit |
| 🧪 Tests | Pytest |

---

## 🏗️ System Architecture

```text
                    👤 Customer Message
                           │
                           ▼
                ┌─────────────────────┐
                │ 🧠 Intent            │
                │    Classification    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ 🔎 Historical        │
                │    Retrieval (TF-IDF)│
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ 🛡️ Evidence +       │
                │    Routing Policy    │
                └──────────┬──────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
              ✅ Auto-Handle   👨‍💼 Escalate
                    │
                    ▼
                ┌─────────────────────┐
                │ 🤖 Groq LLM         │
                │ Response Generation │
                └──────────┬──────────┘
                           │
                           ▼
                  💬 Support Reply
                           │
                           ▼
                  📊 Evaluation
```

---

## 📂 Dataset

This project uses the **Twitter Customer Support (TWCS)** dataset.

Relevant fields include:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

### 🍎 Selected Brand

**AppleSupport**

The pipeline reconstructs **customer → AppleSupport response pairs** using tweet relationship fields.

### 📈 Development Dataset

Current processed development dataset:

**50,000 customer → historical reply pairs**

The raw dataset is intentionally excluded from Git. Place `twcs.csv` inside:

```text
data/raw/twcs.csv
```

---

## 🏷️ Intent Taxonomy

The system uses **12 AppleSupport-specific intents**.

| # | Intent | Description |
|---:|---|---|
| 1 | `ios_update_issue` | iOS update and installation problems |
| 2 | `battery_issue` | Battery drain, charging and battery-life problems |
| 3 | `device_performance` | Slow, freezing or unstable device behavior |
| 4 | `screen_display_issue` | Screen, display and visual problems |
| 5 | `app_issue` | Problems with applications |
| 6 | `icloud_issue` | iCloud synchronization and storage issues |
| 7 | `camera_issue` | Camera and photo-related problems |
| 8 | `connectivity_issue` | Wi-Fi, Bluetooth, cellular and connectivity issues |
| 9 | `account_security` | Account access, phishing and security concerns |
| 10 | `hardware_repair` | Physical damage and repair requests |
| 11 | `product_information` | Product information and compatibility questions |
| 12 | `general_support` | Other AppleSupport requests |

💡 The taxonomy is intentionally compact and interpretable so that it can be debugged and evaluated reliably.

---

## 🧠 Intent Classification

The current classifier is a **transparent rule-based classifier** using intent-specific keywords and patterns.

### Example

```text
"My iPhone battery is draining very quickly"
                    ↓
             battery_issue
```

Another example:

```text
"My WiFi keeps disconnecting"
                    ↓
           connectivity_issue
```

### Why a Rule-Based Classifier?

The initial classifier was selected because it is:

- ⚡ Fast
- 🔁 Deterministic
- 🔍 Interpretable
- 🐛 Easy to debug
- ♻️ Easy to reproduce

A supervised classifier is planned as a future improvement after collecting the larger human-labelled evaluation set.

---

## 🔎 Historical Retrieval

After intent classification, the system searches historical AppleSupport conversations for relevant examples.

### Current Retriever

**TF-IDF similarity**

Each retrieved case contains:

- 🆔 Conversation ID
- 👤 Customer message
- 💬 Historical AppleSupport reply
- 📏 Similarity score

The top historical cases are passed to the response-generation stage as evidence.

### Why TF-IDF?

TF-IDF was selected as the initial retrieval method because it is:

- Lightweight
- Deterministic
- Local
- Reproducible
- Easy to inspect
- Does not require a vector database

🔮 **Future improvement:** compare TF-IDF with semantic sentence embeddings.

---

## 🤖 Response Generation

The response generator uses a Groq-hosted language model.

### Current Model

```text
openai/gpt-oss-20b
```

The model receives:

- Customer message
- Predicted intent
- Relevant historical customer messages
- Historical AppleSupport replies
- Similarity scores

The generation prompt is designed to produce responses that are:

- ✅ Relevant
- ✅ Helpful
- ✅ Concise
- ✅ Grounded in retrieved evidence
- ✅ Appropriate for customer support

The prompt also instructs the model to avoid inventing:

- ❌ Policies
- ❌ Prices
- ❌ Refunds
- ❌ Timelines
- ❌ Account information
- ❌ Unsupported technical claims
- ❌ Unsupported URLs

If the LLM call fails because of an API error or rate limit, the system uses a **deterministic fallback response**.

---

## 🛡️ Auto-Handle vs Escalate

The system does **not** automatically handle every request.

The routing policy considers:

- 🎯 Intent confidence
- 📚 Historical evidence
- 📏 Retrieval similarity
- ⚠️ Risk-sensitive intent categories

### 🚨 Escalation Conditions

A request is escalated when:

- Intent confidence is below the configured threshold
- Historical evidence is unavailable
- Best retrieval similarity is too low
- Intent is `account_security`
- Intent is `hardware_repair`

Otherwise, the request can be auto-handled.

> 🛡️ **Design principle:** When confidence or evidence is insufficient, prefer human review over a confident unsupported answer.

This conservative strategy prioritizes reliability over maximum automation.

---

## 💬 Example

### Customer Message

```text
My iPhone battery is draining very quickly
```

### Agent Decision

```text
Intent:      battery_issue
Confidence:  95%
Action:      auto_handle
```

The agent retrieves similar historical battery-support conversations and uses those cases as evidence for generating the response.

---

# 🖥️ Streamlit Demo

The project includes an interactive **Streamlit frontend** in `app.py`.

### The UI provides

- 🍎 AppleSupport branding
- 📝 Customer message input
- 🧠 Predicted intent
- 📊 Confidence score
- 🛡️ Auto-handle / Escalate decision
- 💡 Routing reason
- 💬 Generated support reply
- 📚 Historical evidence
- 📏 Similarity scores

### Run the Demo

```powershell
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 📊 Evaluation Strategy

The system is evaluated using a manually labelled **200-example golden set**.

Each labelled example contains:

- `gold_intent`
- `gold_action`
- `gold_reply_quality`
- `label_notes`

### 🎯 Intent Evaluation

Metrics:

- Accuracy
- Macro F1
- Per-intent Precision
- Per-intent Recall
- Per-intent F1

### 🛡️ Routing Evaluation

Metric:

- Auto-handle / Escalate Accuracy

### 💬 Response Evaluation

Human rating:

**1–5 reply-quality score**

LLM judge:

- Helpfulness
- Groundedness
- Correctness
- Overall Quality

### 🤝 Human–LLM Agreement

The LLM judge is **not treated as ground truth**.

Human quality ratings are compared with LLM overall-quality ratings using:

- Exact Agreement
- Quadratic Weighted Cohen's Kappa

---

# 🧪 Development / Smoke-Test Results

⚠️ **Important:** These are **development-only results**, not final benchmark claims.

The current development evaluation uses only **19 labelled examples**. The final benchmark will be regenerated after the full 200-example golden set is labelled.

## Baselines

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| 🟦 Majority Class | 0.8000 | 0.4444 |
| 🟩 TF-IDF + Logistic Regression | 0.8000 | **0.7619** |

## Automated Agent Evaluation

| Metric | Current Result |
|---|---:|
| Intent Accuracy | 0.3158 |
| Intent Macro F1 | 0.2104 |
| Routing Accuracy | 0.4737 |

## LLM Judge Smoke Test

The latest run evaluated 19 examples. **13 human/LLM pairs were valid**, while some Groq generation calls failed during the run.

| Metric | Current Result |
|---|---:|
| Human Quality Mean | 3.231 |
| LLM Judge Overall Mean | 3.692 |
| Exact Agreement | 0.154 |
| Weighted Cohen's Kappa | 0.039 |

> 📌 These results are included to document development progress only. They should not be presented as the final performance of the system.

---

# ⚠️ What Is Misleading About My Headline Number?

A single headline metric can make an AI system appear stronger than it actually is.

For example:

- 📈 Accuracy can be inflated by frequent intents.
- ⚖️ Macro F1 gives a more balanced view across intents.
- 💬 A high response-quality average does not guarantee safe behavior on ambiguous or high-risk cases.
- 🔎 Retrieval similarity does not guarantee semantic relevance.
- 🤖 LLM judge scores are not equivalent to human ground truth.

Therefore, the final evaluation reports multiple dimensions:

**Accuracy + Macro F1 + Per-intent Metrics + Routing Accuracy + Response Quality + Human–LLM Agreement + Failure Analysis**

The main limitation of a single aggregate number is that it can hide poor performance on rare, ambiguous, or high-risk cases.

---

# 🔥 Top 5 Failure Modes

### 1️⃣ Ambiguous Intent

Some customer messages do not contain enough information for reliable classification.

### 2️⃣ Weak Retrieval

TF-IDF can retrieve textually similar conversations that are not actually relevant.

### 3️⃣ Incorrect Routing

The system can either:

- Escalate a case unnecessarily, or
- Auto-handle a case that should receive human review.

### 4️⃣ Unsupported Claims

An LLM may generate technically plausible advice that is not directly supported by retrieved historical evidence.

### 5️⃣ Generic Responses

When evidence is weak, the generated response can become safe but less useful.

These failure modes will be quantified using examples from the final golden-set evaluation.

---

# 🧱 Project Structure

```text
AI/
│
├── app.py                         # 🖥️ Streamlit frontend
│
├── configs/
│
├── data/
│   ├── golden_set/
│   │   └── golden.csv            # Human-labelled evaluation set
│   ├── processed/
│   │   └── conversations.csv     # Processed development data
│   └── raw/
│       └── twcs.csv              # Raw dataset (not committed)
│
├── src/
│   ├── agent.py                  # 🤖 Main AI agent
│   ├── baselines.py              # 📊 Baseline models
│   ├── data.py                   # Data utilities
│   ├── evaluate.py               # 📈 Automated evaluation
│   ├── judge.py                  # ⚖️ LLM judge
│   ├── label_golden.py           # 🏷️ Manual labelling tool
│   ├── make_golden.py            # Golden-set creation
│   ├── pipeline.py               # Dataset reconstruction
│   ├── retrieval.py              # 🔎 Historical retrieval
│   └── taxonomy.py               # 🧠 Intent taxonomy/classifier
│
├── tests/
│   └── test_agent.py             # 🧪 Automated tests
│
├── .env.example
├── .gitignore
├── DECISION_LOG.md
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a local `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

⚠️ **Never commit the real API key to GitHub.**

Use `.env.example` for sharing the required variable names without exposing secrets.

---

# 🔄 Reconstruct Conversations

Place the TWCS dataset at:

```text
data/raw/twcs.csv
```

Run:

```powershell
python -m src.pipeline --input data/raw/twcs.csv --brand AppleSupport --sample 50000
```

Expected development output:

```text
Found 106860 tweets from AppleSupport
Customer/reply pairs: 50000
Output: data\processed\conversations.csv
```

---

# ▶️ Run the AI Agent

Example:

```powershell
python -m src.agent --brand AppleSupport --message "My iPhone battery is draining very quickly"
```

The agent returns:

- Intent
- Confidence
- Action
- Reason
- Generated reply
- Retrieved evidence

---

# 🏷️ Create the Golden Set

Generate the 200-example sample:

```powershell
python -m src.make_golden --input data/processed/conversations.csv --output data/golden_set/golden.csv --n 200
```

Start manual labelling:

```powershell
python -m src.label_golden --input data/golden_set/golden.csv
```

💾 The labelling script saves progress, so labelling can be continued later.

---

# 📈 Run Baselines

```powershell
python -m src.baselines --golden data/golden_set/golden.csv
```

Run this after completing the final golden set to obtain final baseline metrics.

---

# 📊 Run Automated Evaluation

```powershell
python -m src.evaluate --golden data/golden_set/golden.csv --data data/processed/conversations.csv
```

Output:

```text
data/evaluation_results.json
```

---

# ⚖️ Run the LLM Judge

```powershell
python -m src.judge --golden data/golden_set/golden.csv --data data/processed/conversations.csv --output data/judge_results.csv
```

Output:

```text
data/judge_results.csv
```

---

# 🧪 Testing

Run the automated test suite:

```powershell
python -m pytest tests/test_agent.py -v
```

### Current Development Test Status

```text
5 passed
```

The current tests cover:

- 🔋 Battery intent classification
- 📷 Camera intent classification
- 📶 Connectivity intent classification
- 🔐 Account-security intent classification
- 🔎 Retrieval result structure

---

# 📝 Engineering Decision Log

Major engineering decisions are documented in:

```text
DECISION_LOG.md
```

The log covers:

- 🍎 Brand selection
- 🔄 Conversation reconstruction
- 🏷️ Intent taxonomy
- 🔎 Retrieval strategy
- 🧠 Classifier design
- 🤖 LLM selection
- 🛡️ Escalation policy
- 🧪 Golden-set design
- 📊 Baselines
- ⚖️ LLM judging
- 🔥 Failure analysis

---

# ⚠️ Limitations

Current limitations include:

- The intent classifier is rule-based.
- TF-IDF has limited semantic understanding.
- Response generation depends on an external LLM API.
- API rate limits can affect large evaluation runs.
- Historical support replies may contain outdated information.
- Historical examples are not guaranteed to represent ideal modern support behavior.
- Lexical retrieval can return superficially similar cases.
- Aggregate metrics can hide failures on difficult or rare cases.
- The current development evaluation is small.

These limitations are considered when interpreting the final results.

---

# 🚀 Next-Week Improvement Plan

### 📅 Day 1–2 — Improve Intent Classification

Train and evaluate a supervised classifier using the completed golden set.

### 📅 Day 3 — Improve Retrieval

Compare TF-IDF with sentence embeddings and evaluate retrieval quality.

### 📅 Day 4 — Grounding Evaluation

Measure whether generated claims are actually supported by retrieved historical evidence.

### 📅 Day 5 — Safer Response Generation

Add stronger grounding constraints and structured response validation.

### 📅 Day 6 — Routing Optimization

Tune escalation thresholds using the golden set and analyze false auto-handles.

### 📅 Day 7 — Error Analysis

Review difficult examples and update the taxonomy, retrieval strategy, and escalation policy.

---

# 🔁 Reproducibility Checklist

A fresh setup should follow:

1. 🐍 Create virtual environment
2. 📦 Install dependencies
3. 🔐 Configure `GROQ_API_KEY`
4. 📁 Place `twcs.csv` in `data/raw/`
5. 🔄 Run conversation reconstruction
6. 🤖 Run the agent or Streamlit demo
7. 🧪 Create the golden set
8. 🏷️ Complete human labelling
9. 📊 Run baselines
10. 📈 Run automated evaluation
11. ⚖️ Run the LLM judge
12. 🧪 Run the test suite

The core retrieval pipeline does **not require a vector database**.

---

# ✅ Final Status

| Component | Status |
|---|---|
| 🍎 AppleSupport selection | ✅ Done |
| 🔄 Dataset reconstruction | ✅ Done |
| 🏷️ Intent taxonomy | ✅ Done |
| 🧠 Intent classifier | ✅ Done |
| 🔎 Historical retrieval | ✅ Done |
| 🤖 Response generation | ✅ Done |
| 🛡️ Escalation policy | ✅ Done |
| 🖥️ Streamlit frontend | ✅ Done |
| 📊 Two baselines | ✅ Done |
| ⚖️ LLM judge pipeline | ✅ Implemented |
| 🧪 Automated tests | ✅ 5/5 passed |
| 🏷️ Golden-set labelling | ✅ Done |
| 📈 Final evaluation | ✅ Done |
| 🔥 Final failure analysis | ✅ Done |
| 📝 Final report metrics | ✅ Done |

---

# 🎯 Summary

This project implements an end-to-end AppleSupport customer-support workflow:

```text
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
          ↓
Streamlit Demonstration
```

The objective is not merely to generate an AI answer.

The objective is to demonstrate:

- 🧠 **What the system understands**
- 📚 **What evidence it uses**
- 🛡️ **When it should answer**
- 👨‍💼 **When it should escalate**
- 📊 **How well it performs**
- 🔥 **Where it fails**
- 🚀 **How it can be improved**

---

**Made by Rohan Karak**  
🎯 *Hiver SDE Intern — Take-Home Assignment*
