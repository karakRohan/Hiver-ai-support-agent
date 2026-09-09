# from __future__ import annotations
# import re
# from pathlib import Path
# import pandas as pd

# COMMON = {
#     "tweet_id": ["tweet_id", "id"],
#     "author_id": ["author_id", "user_id"],
#     "inbound": ["inbound"],
#     "created_at": ["created_at", "timestamp", "date"],
#     "text": ["text", "tweet", "body"],
#     "response_id": ["response_tweet_id", "response_id"],
#     "parent_id": ["in_response_to_tweet_id", "in_reply_to_status_id", "parent_id"],
#     "handle": ["author_handle", "screen_name", "author"],
# }

# def find_col(df, names):
#     lower = {c.lower(): c for c in df.columns}
#     for n in names:
#         if n.lower() in lower:
#             return lower[n.lower()]
#     return None

# def load_twcs(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     out = pd.DataFrame()
#     for target, candidates in COMMON.items():
#         c = find_col(df, candidates)
#         out[target] = df[c] if c else None
#     if out["text"].isna().all():
#         raise ValueError(f"Could not find text column. Columns: {list(df.columns)}")
#     out["text"] = out["text"].fillna("").astype(str).map(clean_text)
#     out["inbound"] = out["inbound"].map(normalize_bool)
#     out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
#     out["tweet_id"] = out["tweet_id"].fillna(pd.Series(range(len(out)))).astype(str)
#     out["parent_id"] = out["parent_id"].astype("string")
#     out["response_id"] = out["response_id"].astype("string")
#     return out

# def normalize_bool(x):
#     if pd.isna(x): return None
#     s = str(x).strip().lower()
#     if s in {"true","1","yes","y","inbound"}: return True
#     if s in {"false","0","no","n","outbound"}: return False
#     return None

# def clean_text(s: str) -> str:
#     s = re.sub(r"\s+", " ", s).strip()
#     s = re.sub(r"https?://\S+", "<URL>", s)
#     return s

# def filter_brand(df: pd.DataFrame, brand: str) -> pd.DataFrame:
#     if df["author_id"].isna().all():
#         return df
#     # TWCS has author_id rather than a reliable brand field. The brand is generally
#     # identifiable by outbound author/account metadata in the raw file. For arbitrary
#     # schemas, accept author_id/author_handle containing the requested brand.
#     mask = pd.Series(False, index=df.index)
#     for col in ["author_id", "handle"]:
#         if col in df:
#             mask |= df[col].fillna("").astype(str).str.contains(re.escape(brand), case=False, regex=True)
#     # If no direct match is possible, keep data and let the pipeline inspect candidates.
#     return df.loc[mask] if mask.any() else df

# def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
#     by_id = df.set_index("tweet_id")
#     rows = []
#     for _, r in df.iterrows():
#         if r["inbound"] is not True:
#             continue
#         parent = r["parent_id"]
#         response = r["response_id"]
#         reply_text = ""
#         if pd.notna(response) and str(response) in by_id.index:
#             reply_text = str(by_id.loc[str(response), "text"])
#         elif pd.notna(parent) and str(parent) in by_id.index:
#             p = by_id.loc[str(parent)]
#             # parent can be the previous support tweet; still useful as context.
#             reply_text = str(p["text"])
#         rows.append({
#             "conversation_id": str(r["tweet_id"]),
#             "customer_text": str(r["text"]),
#             "historical_reply": reply_text,
#             "created_at": r["created_at"],
#         })
#     return pd.DataFrame(rows)


from __future__ import annotations

import re
import pandas as pd


REQUIRED = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def clean_text(s):
    s = re.sub(r"\s+", " ", str(s)).strip()
    s = re.sub(r"https?://\S+", "<URL>", s)
    return s


def load_twcs(path, nrows=None):
    df = pd.read_csv(path, nrows=nrows)

    missing = [c for c in REQUIRED if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["tweet_id"] = df["tweet_id"].astype(str)
    df["author_id"] = df["author_id"].astype(str)

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .map(clean_text)
    )

    df["inbound"] = (
        df["inbound"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce"
    )

    df["response_tweet_id"] = (
        df["response_tweet_id"]
        .fillna("")
        .astype(str)
    )

    df["in_response_to_tweet_id"] = (
        df["in_response_to_tweet_id"]
        .fillna("")
        .astype(str)
    )

    return df


def parse_ids(value):

    if not value or str(value).lower() == "nan":
        return []

    return [
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    ]


def build_brand_pairs(
    path,
    brand,
    chunksize=100000
):

    brand_ids = set()
    replies = {}

    # ---------------------------------
    # PASS 1
    # Find all tweets from the brand
    # ---------------------------------

    for chunk in pd.read_csv(
        path,
        usecols=REQUIRED,
        chunksize=chunksize
    ):

        chunk["tweet_id"] = (
            chunk["tweet_id"]
            .astype(str)
        )

        chunk["author_id"] = (
            chunk["author_id"]
            .astype(str)
        )

        chunk["text"] = (
            chunk["text"]
            .fillna("")
            .astype(str)
            .map(clean_text)
        )

        mask = (
            chunk["author_id"]
            .str.lower()
            .eq(brand.lower())
        )

        for _, row in chunk.loc[mask].iterrows():

            tweet_id = str(row["tweet_id"])

            brand_ids.add(tweet_id)

            replies[tweet_id] = str(
                row["text"]
            )

    if not brand_ids:

        raise ValueError(
            f"No tweets found for author_id='{brand}'. "
            "Use an exact brand name from the dataset."
        )

    print(
        f"Found {len(brand_ids)} tweets "
        f"from {brand}"
    )

    # ---------------------------------
    # PASS 2
    # Find customer -> brand replies
    # ---------------------------------

    rows = []

    for chunk in pd.read_csv(
        path,
        usecols=REQUIRED,
        chunksize=chunksize
    ):

        chunk["tweet_id"] = (
            chunk["tweet_id"]
            .astype(str)
        )

        chunk["text"] = (
            chunk["text"]
            .fillna("")
            .astype(str)
            .map(clean_text)
        )

        inbound = (
            chunk["inbound"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )

        customer_rows = chunk.loc[inbound]

        for _, row in customer_rows.iterrows():

            response_ids = parse_ids(
                row["response_tweet_id"]
            )

            for response_id in response_ids:

                if response_id in brand_ids:

                    rows.append({

                        "conversation_id":
                            f"{row['tweet_id']}_{response_id}",

                        "customer_tweet_id":
                            str(row["tweet_id"]),

                        "brand_tweet_id":
                            response_id,

                        "customer_text":
                            str(row["text"]),

                        "historical_reply":
                            replies[response_id],

                        "created_at":
                            row["created_at"],

                        "brand":
                            brand
                    })

    result = pd.DataFrame(rows)

    if result.empty:

        return pd.DataFrame(
            columns=[
                "conversation_id",
                "customer_tweet_id",
                "brand_tweet_id",
                "customer_text",
                "historical_reply",
                "created_at",
                "brand"
            ]
        )

    return (
        result
        .drop_duplicates(
            "conversation_id"
        )
        .reset_index(drop=True)
    )