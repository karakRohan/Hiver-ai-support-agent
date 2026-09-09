# from __future__ import annotations

# import argparse
# import json
# import os
# from pathlib import Path

# import pandas as pd
# from dotenv import load_dotenv

# load_dotenv()


# # =========================================================
# # LLM Judge
# # =========================================================

# def judge_reply(
#     customer_message: str,
#     historical_reply: str,
#     generated_reply: str,
# ):
#     """
#     Ask an LLM to evaluate a generated support reply.

#     Scores:
#     - helpfulness: 1-5
#     - groundedness: 1-5
#     - correctness: 1-5
#     - overall: 1-5
#     """

#     api_key = os.getenv("GROQ_API_KEY")

#     if not api_key:
#         raise RuntimeError(
#             "GROQ_API_KEY is not configured in .env"
#         )

#     from groq import Groq

#     client = Groq(
#         api_key=api_key
#     )

#     prompt = f"""
# You are evaluating an AI customer-support reply.

# Customer message:
# {customer_message}

# Historical support reply:
# {historical_reply}

# AI-generated reply:
# {generated_reply}

# Evaluate the AI-generated reply.

# Score each dimension from 1 to 5:

# helpfulness:
# 1 = not helpful
# 2 = poor
# 3 = acceptable
# 4 = good
# 5 = excellent

# groundedness:
# 1 = mostly unsupported
# 2 = contains substantial unsupported claims
# 3 = partly grounded
# 4 = mostly grounded
# 5 = fully grounded in the available evidence

# correctness:
# 1 = incorrect or unsafe
# 2 = major problems
# 3 = acceptable
# 4 = mostly correct
# 5 = correct and safe

# overall:
# 1 = very poor
# 2 = poor
# 3 = acceptable
# 4 = good
# 5 = excellent

# Important:
# - Do not reward invented facts.
# - Penalize unsupported policies, timelines, prices,
#   account information, or actions.
# - The historical reply is evidence, not necessarily a
#   perfect answer.
# - Judge the generated reply itself.

# Return ONLY valid JSON in exactly this format:

# {{
#   "helpfulness": 1,
#   "groundedness": 1,
#   "correctness": 1,
#   "overall": 1,
#   "reason": "short explanation"
# }}
# """

#     response = client.chat.completions.create(
#         model=os.getenv(
#             "GROQ_JUDGE_MODEL",
#             os.getenv(
#                 "GROQ_MODEL",
#                 "openai/gpt-oss-20b"
#             )
#         ),
#         messages=[
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a strict customer-support "
#                     "quality evaluator. Return valid JSON only."
#                 ),
#             },
#             {
#                 "role": "user",
#                 "content": prompt,
#             },
#         ],
#         temperature=0,
#     )

#     text = response.choices[0].message.content.strip()

#     # Remove markdown code fences if the model adds them.
#     if text.startswith("```"):
#         text = text.replace("```json", "")
#         text = text.replace("```", "")
#         text = text.strip()

#     result = json.loads(text)

#     required = [
#         "helpfulness",
#         "groundedness",
#         "correctness",
#         "overall",
#         "reason",
#     ]

#     for key in required:
#         if key not in result:
#             raise ValueError(
#                 f"Judge response missing field: {key}"
#             )

#     return result


# # =========================================================
# # Generate AI replies for Golden Set
# # =========================================================

# def generate_reply(
#     customer_message: str,
#     historical_reply: str,
# ):
#     """
#     Generate an AI reply using the existing agent.
#     """

#     from .agent import run_agent

#     # We only need a small historical dataframe for the test.
#     # The complete dataset is loaded by the main function.
#     return None


# # =========================================================
# # Evaluate Golden Set
# # =========================================================

# def run_judge(
#     golden_path: str,
#     data_path: str,
#     output_path: str,
#     limit: int | None = None,
# ):
#     """
#     Run the agent and LLM judge on labelled golden examples.
#     """

#     from .agent import run_agent

#     golden = pd.read_csv(
#         golden_path
#     )

#     data = pd.read_csv(
#         data_path
#     )

#     # Only use examples with human quality labels.
#     valid = golden[
#         golden["gold_reply_quality"].notna()
#         & (
#             golden["gold_reply_quality"]
#             .astype(str)
#             .str.strip()
#             != ""
#         )
#     ].copy()

#     if limit is not None:
#         valid = valid.head(limit)

#     if len(valid) == 0:
#         raise ValueError(
#             "No gold_reply_quality labels found. "
#             "Please label reply quality in golden.csv first."
#         )

#     print(
#         f"Running LLM judge on {len(valid)} examples..."
#     )

#     results = []

#     for position, (_, row) in enumerate(
#         valid.iterrows(),
#         start=1
#     ):

#         print(
#             f"[{position}/{len(valid)}] "
#             f"Evaluating..."
#         )

#         customer_message = str(
#             row["customer_text"]
#         )

#         historical_reply = str(
#             row["historical_reply"]
#         )

#         # Run the complete support agent.
#         agent_result = run_agent(
#             customer_message,
#             data
#         )

#         generated_reply = agent_result[
#             "reply"
#         ]

#         # Judge generated response.
#         judge_result = judge_reply(
#             customer_message,
#             historical_reply,
#             generated_reply,
#         )

#         results.append(
#             {
#                 "conversation_id": str(
#                     row["conversation_id"]
#                 ),

#                 "customer_text": customer_message,

#                 "historical_reply": historical_reply,

#                 "generated_reply": generated_reply,

#                 "human_quality": float(
#                     row["gold_reply_quality"]
#                 ),

#                 "judge_helpfulness": judge_result[
#                     "helpfulness"
#                 ],

#                 "judge_groundedness": judge_result[
#                     "groundedness"
#                 ],

#                 "judge_correctness": judge_result[
#                     "correctness"
#                 ],

#                 "judge_overall": judge_result[
#                     "overall"
#                 ],

#                 "judge_reason": judge_result[
#                     "reason"
#                 ],

#                 "intent": agent_result[
#                     "intent"
#                 ],

#                 "confidence": agent_result[
#                     "confidence"
#                 ],

#                 "action": agent_result[
#                     "action"
#                 ],
#             }
#         )

#     result_df = pd.DataFrame(
#         results
#     )

#     output = Path(
#         output_path
#     )

#     output.parent.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     result_df.to_csv(
#         output,
#         index=False
#     )

#     print()
#     print("=" * 60)
#     print("LLM JUDGE COMPLETE")
#     print("=" * 60)
#     print()
#     print(
#         f"Examples evaluated: {len(result_df)}"
#     )
#     print(
#         f"Output: {output}"
#     )
#     print()

#     print(
#         "Human quality mean:",
#         round(
#             result_df["human_quality"].mean(),
#             3
#         )
#     )

#     print(
#         "LLM judge overall mean:",
#         round(
#             result_df["judge_overall"].mean(),
#             3
#         )
#     )


# # =========================================================
# # CLI
# # =========================================================

# def main():

#     parser = argparse.ArgumentParser(
#         description="LLM judge for AI support replies."
#     )

#     parser.add_argument(
#         "--golden",
#         default="data/golden_set/golden.csv",
#         help="Golden set CSV."
#     )

#     parser.add_argument(
#         "--data",
#         default="data/processed/conversations.csv",
#         help="Historical conversations CSV."
#     )

#     parser.add_argument(
#         "--output",
#         default="data/judge_results.csv",
#         help="Judge output CSV."
#     )

#     parser.add_argument(
#         "--limit",
#         type=int,
#         default=None,
#         help="Optional number of examples to evaluate."
#     )

#     args = parser.parse_args()

#     run_judge(
#         golden_path=args.golden,
#         data_path=args.data,
#         output_path=args.output,
#         limit=args.limit,
#     )


# if __name__ == "__main__":
#     main()




from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score

from src.agent import run_agent


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

DEFAULT_MODEL = os.getenv(
    "GROQ_JUDGE_MODEL",
    os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
)

try:
    from groq import Groq
except ImportError:
    Groq = None


# ---------------------------------------------------------
# Groq LLM Judge
# ---------------------------------------------------------

def create_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please check your .env file."
        )

    if Groq is None:
        raise ImportError(
            "groq package is not installed. Run: pip install groq"
        )

    return Groq(api_key=api_key)


def judge_reply(
    client,
    model: str,
    customer_message: str,
    generated_reply: str,
    historical_evidence: str,
) -> dict[str, Any]:

    prompt = f"""
You are evaluating an AI customer-support reply.

Score the AI reply from 1 to 5 on four dimensions:

1. helpfulness
2. groundedness
3. correctness
4. overall_quality

Use this scale:

1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Important:
- The reply should directly address the customer's problem.
- It should be grounded in the supplied historical support evidence.
- Do not reward unsupported claims.
- Do not assume facts that are not present in the evidence.
- Be strict and consistent.

CUSTOMER MESSAGE:
{customer_message}

HISTORICAL SUPPORT EVIDENCE:
{historical_evidence}

AI GENERATED REPLY:
{generated_reply}

Return ONLY valid JSON in exactly this format:

{{
  "helpfulness": 1,
  "groundedness": 1,
  "correctness": 1,
  "overall_quality": 1,
  "reason": "short explanation"
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict but fair evaluator of "
                    "customer-support AI responses."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    text = response.choices[0].message.content.strip()

    # Remove accidental markdown code fences
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(
            f"LLM returned invalid JSON:\n{text}"
        )

    return result


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def safe_int(value):
    try:
        return int(float(value))
    except Exception:
        return None


def build_evidence_text(evidence_df: pd.DataFrame) -> str:
    if evidence_df is None or len(evidence_df) == 0:
        return "No historical support evidence was retrieved."

    parts = []

    for i, row in evidence_df.head(5).iterrows():
        customer = str(row.get("customer_text", ""))
        reply = str(row.get("historical_reply", ""))
        similarity = row.get("similarity", "")

        parts.append(
            f"""
Evidence {len(parts) + 1}
Similarity: {similarity}
Customer: {customer}
Historical reply: {reply}
""".strip()
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="LLM judge for AppleSupport AI agent."
    )

    parser.add_argument(
        "--golden",
        default="data/golden_set/golden.csv",
        help="Path to golden set CSV",
    )

    parser.add_argument(
        "--data",
        default="data/processed/conversations.csv",
        help="Path to historical conversations CSV",
    )

    parser.add_argument(
        "--output",
        default="data/judge_results.csv",
        help="Output CSV path",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of examples to evaluate",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    golden_path = Path(args.golden)
    data_path = Path(args.data)
    output_path = Path(args.output)

    if not golden_path.exists():
        raise FileNotFoundError(
            f"Golden set not found: {golden_path}"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Historical data not found: {data_path}"
        )

    golden_df = pd.read_csv(golden_path)
    conversations_df = pd.read_csv(data_path)

    # Only evaluate examples with human reply-quality labels
    eval_df = golden_df[
        golden_df["gold_reply_quality"].notna()
    ].copy()

    if args.limit is not None:
        eval_df = eval_df.head(args.limit)

    if len(eval_df) == 0:
        print("No human reply-quality labels found.")
        return

    print(
        f"Running LLM judge on {len(eval_df)} examples..."
    )

    client = create_groq_client()

    results = []

    # -----------------------------------------------------
    # Evaluate each example
    # -----------------------------------------------------

    for idx, row in enumerate(
        eval_df.iterrows(),
        start=1
    ):

        _, item = row

        print(
            f"[{idx}/{len(eval_df)}] Evaluating..."
        )

        customer_message = str(
            item["customer_text"]
        )

        human_quality = safe_int(
            item["gold_reply_quality"]
        )

        try:

            # Run the actual AI agent
            agent_result = run_agent(
                brand="AppleSupport",
                message=customer_message,
                conversations=conversations_df,
            )

            generated_reply = str(
                agent_result.get("reply", "")
            )

            evidence = agent_result.get(
                "evidence",
                []
            )

            # Convert evidence to dataframe if needed
            if isinstance(evidence, pd.DataFrame):
                evidence_df = evidence
            else:
                evidence_df = pd.DataFrame(evidence)

            evidence_text = build_evidence_text(
                evidence_df
            )

            # LLM Judge
            judge_result = judge_reply(
                client=client,
                model=DEFAULT_MODEL,
                customer_message=customer_message,
                generated_reply=generated_reply,
                historical_evidence=evidence_text,
            )

            results.append(
                {
                    "conversation_id": item.get(
                        "conversation_id", ""
                    ),
                    "customer_text": customer_message,
                    "human_quality": human_quality,
                    "generated_reply": generated_reply,
                    "agent_intent": agent_result.get(
                        "intent", ""
                    ),
                    "agent_confidence": agent_result.get(
                        "confidence", ""
                    ),
                    "agent_action": agent_result.get(
                        "action", ""
                    ),
                    "helpfulness": safe_int(
                        judge_result.get("helpfulness")
                    ),
                    "groundedness": safe_int(
                        judge_result.get("groundedness")
                    ),
                    "correctness": safe_int(
                        judge_result.get("correctness")
                    ),
                    "llm_overall_quality": safe_int(
                        judge_result.get(
                            "overall_quality"
                        )
                    ),
                    "judge_reason": judge_result.get(
                        "reason", ""
                    ),
                }
            )

        except Exception as e:

            print(
                f"WARNING: Example {idx} failed: {e}"
            )

            results.append(
                {
                    "conversation_id": item.get(
                        "conversation_id", ""
                    ),
                    "customer_text": customer_message,
                    "human_quality": human_quality,
                    "generated_reply": "",
                    "agent_intent": "",
                    "agent_confidence": "",
                    "agent_action": "",
                    "helpfulness": None,
                    "groundedness": None,
                    "correctness": None,
                    "llm_overall_quality": None,
                    "judge_reason": f"ERROR: {e}",
                }
            )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    human_scores = pd.to_numeric(
        results_df["human_quality"],
        errors="coerce"
    )

    llm_scores = pd.to_numeric(
        results_df["llm_overall_quality"],
        errors="coerce"
    )

    valid_mask = (
        human_scores.notna()
        & llm_scores.notna()
    )

    human_valid = human_scores[valid_mask].astype(int)
    llm_valid = llm_scores[valid_mask].astype(int)

    print()
    print("=" * 60)
    print("LLM JUDGE COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Examples evaluated: {len(results_df)}"
    )

    print(
        f"Valid human/LLM pairs: {len(human_valid)}"
    )

    print(
        f"Output: {output_path}"
    )

    if len(human_valid) > 0:

        human_mean = human_valid.mean()
        llm_mean = llm_valid.mean()

        exact_agreement = (
            human_valid.values
            == llm_valid.values
        ).mean()

        # Weighted Cohen's Kappa is appropriate
        # for ordinal 1-5 ratings.
        if len(set(human_valid)) > 1 and len(set(llm_valid)) > 1:
            weighted_kappa = cohen_kappa_score(
                human_valid,
                llm_valid,
                weights="quadratic",
            )
        else:
            weighted_kappa = float("nan")

        print()
        print(
            f"Human quality mean: {human_mean:.3f}"
        )

        print(
            f"LLM judge overall mean: {llm_mean:.3f}"
        )

        print(
            f"Exact agreement: {exact_agreement:.3f}"
        )

        if pd.notna(weighted_kappa):
            print(
                f"Weighted Cohen's kappa: "
                f"{weighted_kappa:.3f}"
            )
        else:
            print(
                "Weighted Cohen's kappa: "
                "not available"
            )

        print()
        print(
            "Human vs LLM agreement is based on "
            "the 1-5 reply-quality ratings."
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()