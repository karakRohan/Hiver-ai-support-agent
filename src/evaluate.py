from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from .taxonomy import classify_rules
from .agent import should_escalate

def evaluate(df, golden):
    preds = golden["customer_text"].fillna("").map(lambda x: classify_rules(str(x))[0])
    y = golden["gold_intent"].fillna("")
    valid = y.ne("")
    intent = {
        "accuracy": float(accuracy_score(y[valid], preds[valid])) if valid.any() else None,
        "macro_f1": float(f1_score(y[valid], preds[valid], average="macro")) if valid.any() else None,
        "n_labelled": int(valid.sum()),
    }
    actions = []
    for _, r in golden.iterrows():
        intent_name, conf = classify_rules(str(r["customer_text"]))
        esc, _ = should_escalate(intent_name, conf, [])
        actions.append("escalate" if esc else "auto_handle")
    gold_action = golden["gold_action"].fillna("")
    av = gold_action.ne("")
    routing = {
        "accuracy": float(accuracy_score(gold_action[av], pd.Series(actions)[av])) if av.any() else None,
        "n_labelled": int(av.sum()),
    }
    return {"intent": intent, "routing": routing}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--golden", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.data)
    golden = pd.read_csv(args.golden)
    result = evaluate(df, golden)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
