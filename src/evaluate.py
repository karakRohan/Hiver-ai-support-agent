# from __future__ import annotations
# import argparse, json
# from pathlib import Path
# import pandas as pd
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# from .taxonomy import classify_rules
# from .agent import should_escalate

# def evaluate(df, golden):
#     preds = golden["customer_text"].fillna("").map(lambda x: classify_rules(str(x))[0])
#     y = golden["gold_intent"].fillna("")
#     valid = y.ne("")
#     intent = {
#         "accuracy": float(accuracy_score(y[valid], preds[valid])) if valid.any() else None,
#         "macro_f1": float(f1_score(y[valid], preds[valid], average="macro")) if valid.any() else None,
#         "n_labelled": int(valid.sum()),
#     }
#     actions = []
#     for _, r in golden.iterrows():
#         intent_name, conf = classify_rules(str(r["customer_text"]))
#         esc, _ = should_escalate(intent_name, conf, [])
#         actions.append("escalate" if esc else "auto_handle")
#     gold_action = golden["gold_action"].fillna("")
#     av = gold_action.ne("")
#     routing = {
#         "accuracy": float(accuracy_score(gold_action[av], pd.Series(actions)[av])) if av.any() else None,
#         "n_labelled": int(av.sum()),
#     }
#     return {"intent": intent, "routing": routing}

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--data", required=True)
#     p.add_argument("--golden", required=True)
#     p.add_argument("--output", required=True)
#     args = p.parse_args()
#     df = pd.read_csv(args.data)
#     golden = pd.read_csv(args.golden)
#     result = evaluate(df, golden)
#     Path(args.output).parent.mkdir(parents=True, exist_ok=True)
#     Path(args.output).write_text(json.dumps(result, indent=2))
#     print(json.dumps(result, indent=2))

# if __name__ == "__main__":
#     main()


from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from .taxonomy import classify_rules
from .agent import should_escalate
from .retrieval import Retriever


# =========================================================
# Intent Evaluation
# =========================================================

def evaluate_intent(golden: pd.DataFrame):
    """
    Evaluate the rule-based intent classifier against
    human-labelled golden examples.
    """

    valid = golden[
        golden["gold_intent"].notna()
        & (golden["gold_intent"].astype(str).str.strip() != "")
    ].copy()

    if len(valid) == 0:
        return {
            "accuracy": None,
            "macro_f1": None,
            "n_labelled": 0,
            "classification_report": {},
        }

    predictions = []

    for text in valid["customer_text"].fillna(""):
        intent, _ = classify_rules(
            str(text)
        )
        predictions.append(intent)

    y_true = (
        valid["gold_intent"]
        .astype(str)
        .str.strip()
    )

    y_pred = pd.Series(
        predictions,
        index=valid.index
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": round(
            float(accuracy),
            4
        ),
        "macro_f1": round(
            float(macro_f1),
            4
        ),
        "n_labelled": int(
            len(valid)
        ),
        "classification_report": report,
    }


# =========================================================
# Routing Evaluation
# =========================================================

def evaluate_routing(
    data: pd.DataFrame,
    golden: pd.DataFrame
):
    """
    Evaluate auto_handle vs escalate.

    Retrieval evidence is included so the escalation policy
    is evaluated in the same way as the actual agent.
    """

    valid = golden[
        golden["gold_action"].notna()
        & (golden["gold_action"].astype(str).str.strip() != "")
    ].copy()

    if len(valid) == 0:
        return {
            "accuracy": None,
            "n_labelled": 0,
        }

    # Use TF-IDF for deterministic and fast evaluation.
    retriever = Retriever(
        data,
        embedder=None
    )

    predictions = []

    for _, row in valid.iterrows():

        customer_text = str(
            row["customer_text"]
        )

        intent, confidence = classify_rules(
            customer_text
        )

        evidence = retriever.search(
            customer_text,
            k=5
        )

        escalate, _ = should_escalate(
            intent,
            confidence,
            evidence
        )

        action = (
            "escalate"
            if escalate
            else "auto_handle"
        )

        predictions.append(action)

    y_true = (
        valid["gold_action"]
        .astype(str)
        .str.strip()
    )

    y_pred = pd.Series(
        predictions,
        index=valid.index
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    return {
        "accuracy": round(
            float(accuracy),
            4
        ),
        "n_labelled": int(
            len(valid)
        ),
    }


# =========================================================
# Main Evaluation
# =========================================================

def evaluate(
    data: pd.DataFrame,
    golden: pd.DataFrame
):
    """
    Run all available evaluation metrics.
    """

    intent_results = evaluate_intent(
        golden
    )

    routing_results = evaluate_routing(
        data,
        golden
    )

    return {
        "intent": intent_results,
        "routing": routing_results,
    }


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate AppleSupport AI support agent."
    )

    parser.add_argument(
        "--data",
        default="data/processed/conversations.csv",
        help="Historical conversation dataset."
    )

    parser.add_argument(
        "--golden",
        default="data/golden_set/golden.csv",
        help="Human-labelled golden set."
    )

    parser.add_argument(
        "--output",
        default="data/evaluation_results.json",
        help="Output JSON file."
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    print("Loading historical conversations...")

    data = pd.read_csv(
        args.data
    )

    print(
        f"Historical conversations: {len(data)}"
    )

    print()

    print("Loading golden set...")

    golden = pd.read_csv(
        args.golden
    )

    labelled_intents = golden[
        golden["gold_intent"].notna()
        & (
            golden["gold_intent"]
            .astype(str)
            .str.strip()
            != ""
        )
    ]

    labelled_actions = golden[
        golden["gold_action"].notna()
        & (
            golden["gold_action"]
            .astype(str)
            .str.strip()
            != ""
        )
    ]

    print(
        f"Intent labels: {len(labelled_intents)}"
    )

    print(
        f"Action labels: {len(labelled_actions)}"
    )

    print()

    # -----------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------

    results = evaluate(
        data,
        golden
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print()

    print("INTENT CLASSIFICATION")
    print(
        f"  Accuracy : "
        f"{results['intent']['accuracy']}"
    )

    print(
        f"  Macro F1 : "
        f"{results['intent']['macro_f1']}"
    )

    print(
        f"  Labelled : "
        f"{results['intent']['n_labelled']}"
    )

    print()

    print("ROUTING")
    print(
        f"  Accuracy : "
        f"{results['routing']['accuracy']}"
    )

    print(
        f"  Labelled : "
        f"{results['routing']['n_labelled']}"
    )

    print()

    print("=" * 60)
    print(
        f"Results saved to: {output_path}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()