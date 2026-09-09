# from __future__ import annotations
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# class Retriever:
#     def __init__(self, texts, embedder=None):
#         self.texts = list(texts)
#         self.embedder = embedder
#         self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
#         self.tfidf = self.vectorizer.fit_transform(self.texts) if self.texts else None
#         self.embeddings = None
#         if embedder and self.texts:
#             self.embeddings = embedder.encode(self.texts, normalize_embeddings=True, show_progress_bar=False)

#     def search(self, query, k=5):
#         if not self.texts:
#             return []
#         if self.embeddings is not None:
#             q = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
#             scores = np.asarray(self.embeddings @ q).reshape(-1)
#         else:
#             q = self.vectorizer.transform([query])
#             scores = cosine_similarity(q, self.tfidf)[0]
#         idx = np.argsort(-scores)[:k]
#         return [{"index": int(i), "text": self.texts[i], "similarity": float(scores[i])} for i in idx]


from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    """
    Retrieves historically similar AppleSupport conversations.

    Each retrieved result contains:
    - original customer message
    - historical support reply
    - similarity score
    - conversation id
    """

    def __init__(self, dataframe, embedder=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.embedder = embedder

        self.df["customer_text"] = self.df["customer_text"].fillna("").astype(str)
        self.df["historical_reply"] = (
            self.df["historical_reply"].fillna("").astype(str)
        )

        # We retrieve based primarily on the customer's problem.
        self.texts = self.df["customer_text"].tolist()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
        )

        self.tfidf = (
            self.vectorizer.fit_transform(self.texts)
            if self.texts
            else None
        )

        self.embeddings = None

        if self.embedder is not None and self.texts:
            self.embeddings = self.embedder.encode(
                self.texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    def search(self, query: str, k: int = 5):
        """
        Return top-k historically similar conversations.
        """

        if not self.texts:
            return []

        query = str(query).strip()

        if not query:
            return []

        # Semantic retrieval when SentenceTransformer is available.
        if self.embeddings is not None:
            query_embedding = self.embedder.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]

            scores = np.asarray(
                self.embeddings @ query_embedding
            ).reshape(-1)

        # TF-IDF fallback.
        else:
            query_vector = self.vectorizer.transform([query])
            scores = cosine_similarity(
                query_vector,
                self.tfidf
            )[0]

        # Highest similarity first.
        indices = np.argsort(-scores)[:k]

        results = []

        for index in indices:
            row = self.df.iloc[int(index)]

            results.append(
                {
                    "index": int(index),
                    "conversation_id": str(row["conversation_id"]),
                    "customer_text": row["customer_text"],
                    "historical_reply": row["historical_reply"],
                    "similarity": float(scores[index]),
                }
            )

        return results


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/processed/conversations.csv"
    )

    parser.add_argument(
        "--query",
        required=True
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    # Try semantic embeddings first.
    embedder = None

    try:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Using SentenceTransformer embeddings.")

    except Exception:
        print("SentenceTransformer unavailable. Using TF-IDF.")

    retriever = Retriever(df, embedder)

    results = retriever.search(args.query, args.k)

    print("\nTop retrieved historical cases:\n")

    for i, result in enumerate(results, start=1):

        print(f"--- Result {i} ---")
        print(f"Conversation ID: {result['conversation_id']}")
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Customer: {result['customer_text']}")
        print(f"Historical reply: {result['historical_reply']}")
        print()