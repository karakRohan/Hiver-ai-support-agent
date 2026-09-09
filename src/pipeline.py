# from __future__ import annotations
# import argparse
# from pathlib import Path
# import pandas as pd
# from .data import load_twcs, filter_brand, build_pairs

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--input", required=True)
#     p.add_argument("--brand", required=True)
#     p.add_argument("--sample", type=int, default=50000)
#     args = p.parse_args()

#     df = load_twcs(args.input)
#     if len(df) > args.sample:
#         df = df.sample(args.sample, random_state=42)
#     branded = filter_brand(df, args.brand)
#     pairs = build_pairs(branded)
#     Path("data/processed").mkdir(parents=True, exist_ok=True)
#     pairs.to_csv("data/processed/conversations.csv", index=False)
#     print(f"Loaded: {len(df)} tweets")
#     print(f"Brand-filtered: {len(branded)} tweets")
#     print(f"Customer/reply pairs: {len(pairs)}")
#     print("Wrote data/processed/conversations.csv")

# if __name__ == "__main__":
#     main()
