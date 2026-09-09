from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, texts, embedder=None):
        self.texts = list(texts)
        self.embedder = embedder
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
        self.tfidf = self.vectorizer.fit_transform(self.texts) if self.texts else None
        self.embeddings = None
        if embedder and self.texts:
            self.embeddings = embedder.encode(self.texts, normalize_embeddings=True, show_progress_bar=False)

    def search(self, query, k=5):
        if not self.texts:
            return []
        if self.embeddings is not None:
            q = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            scores = np.asarray(self.embeddings @ q).reshape(-1)
        else:
            q = self.vectorizer.transform([query])
            scores = cosine_similarity(q, self.tfidf)[0]
        idx = np.argsort(-scores)[:k]
        return [{"index": int(i), "text": self.texts[i], "similarity": float(scores[i])} for i in idx]
