from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# LLM Judge
# =========================================================

def judge_reply(
    customer_message: str,
    historical_reply: str,
    generated_reply: str,
):
    """
    Ask an LLM to evaluate a generated support reply.

    Scores:
    - helpfulness: 1-5
    - groundedness: 1-5
    - correctness: 1-5
    - overall: 1-5
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured in .env"
        )

    from groq import Groq

    client = Groq(
        api_key=api_key
    )

    prompt = f"""
You are evaluating an AI customer-support reply.

Customer message:
{customer_message}

Historical support reply:
{historical_reply}

AI-generated reply:
{generated_reply}

Evaluate the AI-generated reply.

Score each dimension from 1 to 5:

helpfulness:
1 = not helpful
2 = poor
3 = acceptable
4 = good
5 = excellent

groundedness:
1 = mostly unsupported
2 = contains substantial unsupported claims
3 = partly grounded
4 = mostly grounded
5 = fully grounded in the available evidence

correctness:
1 = incorrect or unsafe
2 = major problems
3 = acceptable
4 = mostly correct
5 = correct and safe

overall:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Important:
- Do not reward invented facts.
- Penalize unsupported policies, timelines, prices,
  account information, or actions.
- The historical reply is evidence, not necessarily a
  perfect answer.
- Judge the generated reply itself.

Return ONLY valid JSON in exactly this format:

{{
  "helpfulness": 1,
  "groundedness": 1,
  "correctness": 1,
  "overall": 1,
  "reason": "short explanation"
}}
"""

    response = client.chat.completions.create(
        model=os.getenv(
            "GROQ_JUDGE_MODEL",
            os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-20b"
            )
        ),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict customer-support "
                    "quality evaluator. Return valid JSON only."
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

    # Remove markdown code fences if the model adds them.
    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    result = json.loads(text)

    required = [
        "helpfulness",
        "groundedness",
        "correctness",
        "overall",
        "reason",
    ]

    for key in required:
        if key not in result:
            raise ValueError(
                f"Judge response missing field: {key}"
            )

    return result


# =========================================================
# Generate AI replies for Golden Set
# =========================================================

def generate_reply(
    customer_message: str,
    historical_reply: str,
):
    """
    Generate an AI reply using the existing agent.
    """

    from .agent import run_agent

    # We only need a small historical dataframe for the test.
    # The complete dataset is loaded by the main function.
    return None


# =========================================================
# Evaluate Golden Set
# =========================================================

def run_judge(
    golden_path: str,
    data_path: str,
    output_path: str,
    limit: int | None = None,
):
    """
    Run the agent and LLM judge on labelled golden examples.
    """

    from .agent import run_agent

    golden = pd.read_csv(
        golden_path
    )

    data = pd.read_csv(
        data_path
    )

    # Only use examples with human quality labels.
    valid = golden[
        golden["gold_reply_quality"].notna()
        & (
            golden["gold_reply_quality"]
            .astype(str)
            .str.strip()
            != ""
        )
    ].copy()

    if limit is not None:
        valid = valid.head(limit)

    if len(valid) == 0:
        raise ValueError(
            "No gold_reply_quality labels found. "
            "Please label reply quality in golden.csv first."
        )

    print(
        f"Running LLM judge on {len(valid)} examples..."
    )

    results = []

    for position, (_, row) in enumerate(
        valid.iterrows(),
        start=1
    ):

        print(
            f"[{position}/{len(valid)}] "
            f"Evaluating..."
        )

        customer_message = str(
            row["customer_text"]
        )

        historical_reply = str(
            row["historical_reply"]
        )

        # Run the complete support agent.
        agent_result = run_agent(
            customer_message,
            data
        )

        generated_reply = agent_result[
            "reply"
        ]

        # Judge generated response.
        judge_result = judge_reply(
            customer_message,
            historical_reply,
            generated_reply,
        )

        results.append(
            {
                "conversation_id": str(
                    row["conversation_id"]
                ),

                "customer_text": customer_message,

                "historical_reply": historical_reply,

                "generated_reply": generated_reply,

                "human_quality": float(
                    row["gold_reply_quality"]
                ),

                "judge_helpfulness": judge_result[
                    "helpfulness"
                ],

                "judge_groundedness": judge_result[
                    "groundedness"
                ],

                "judge_correctness": judge_result[
                    "correctness"
                ],

                "judge_overall": judge_result[
                    "overall"
                ],

                "judge_reason": judge_result[
                    "reason"
                ],

                "intent": agent_result[
                    "intent"
                ],

                "confidence": agent_result[
                    "confidence"
                ],

                "action": agent_result[
                    "action"
                ],
            }
        )

    result_df = pd.DataFrame(
        results
    )

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        output,
        index=False
    )

    print()
    print("=" * 60)
    print("LLM JUDGE COMPLETE")
    print("=" * 60)
    print()
    print(
        f"Examples evaluated: {len(result_df)}"
    )
    print(
        f"Output: {output}"
    )
    print()

    print(
        "Human quality mean:",
        round(
            result_df["human_quality"].mean(),
            3
        )
    )

    print(
        "LLM judge overall mean:",
        round(
            result_df["judge_overall"].mean(),
            3
        )
    )


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="LLM judge for AI support replies."
    )

    parser.add_argument(
        "--golden",
        default="data/golden_set/golden.csv",
        help="Golden set CSV."
    )

    parser.add_argument(
        "--data",
        default="data/processed/conversations.csv",
        help="Historical conversations CSV."
    )

    parser.add_argument(
        "--output",
        default="data/judge_results.csv",
        help="Judge output CSV."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of examples to evaluate."
    )

    args = parser.parse_args()

    run_judge(
        golden_path=args.golden,
        data_path=args.data,
        output_path=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()