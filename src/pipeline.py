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



from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_brand_pairs


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--brand",
        required=True
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=50000
    )

    args = parser.parse_args()

    print("Starting pipeline...")
    print(f"Dataset: {args.input}")
    print(f"Brand: {args.brand}")

    pairs = build_brand_pairs(
        args.input,
        args.brand
    )

    if len(pairs) > args.sample:

        pairs = (
            pairs
            .sample(
                args.sample,
                random_state=42
            )
            .reset_index(drop=True)
        )

    output_dir = Path(
        "data/processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir /
        "conversations.csv"
    )

    pairs.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 50)
    print("PIPELINE COMPLETED")
    print("=" * 50)

    print(
        f"Brand: {args.brand}"
    )

    print(
        f"Customer/reply pairs: "
        f"{len(pairs)}"
    )

    print(
        f"Output: {output_file}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()