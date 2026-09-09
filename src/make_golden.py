from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .taxonomy import classify_rules, taxonomy

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df = df.dropna(subset=["customer_text"]).drop_duplicates("customer_text")
    n = min(args.n, len(df))
    sample = df.sample(n, random_state=42).copy()
    sample["suggested_intent"] = sample["customer_text"].map(lambda x: classify_rules(str(x))[0])
    sample["gold_intent"] = ""
    sample["gold_action"] = ""
    sample["gold_reply_quality"] = ""
    sample["label_notes"] = ""
    cols = ["conversation_id","customer_text","historical_reply","suggested_intent",
            "gold_intent","gold_action","gold_reply_quality","label_notes"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sample[cols].to_csv(args.output, index=False)
    print(f"Created {n} examples. Fill the gold_* columns manually.")
    print("Allowed intents:", ", ".join(taxonomy()))

if __name__ == "__main__":
    main()
