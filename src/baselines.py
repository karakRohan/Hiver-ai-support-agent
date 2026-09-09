from __future__ import annotations

import argparse

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# =========================================================
# Configuration
# =========================================================

RANDOM_STATE = 42


# =========================================================
# Load Golden Set
# =========================================================

def load_golden_data(path: str):
    """
    Load the manually labelled golden set.

    Only rows with a human-provided gold_intent are used.
    """

    df = pd.read_csv(path)

    required_columns = [
        "customer_text",
        "gold_intent",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    # Keep only genuinely labelled rows.
    df = df[
        df["gold_intent"].notna()
        & (df["gold_intent"].astype(str).str.strip() != "")
    ].copy()

    if len(df) < 10:
        raise ValueError(
            "Not enough labelled examples. "
            "Please label more golden-set examples first."
        )

    df["customer_text"] = (
        df["customer_text"]
        .fillna("")
        .astype(str)
    )

    df["gold_intent"] = (
        df["gold_intent"]
        .astype(str)
        .str.strip()
    )

    return df


# =========================================================
# Baseline 1 — Majority Class
# =========================================================

def majority_baseline(y_train, y_test):
    """
    Predict the most frequent intent for every example.
    """

    majority_class = (
        pd.Series(y_train)
        .value_counts()
        .idxmax()
    )

    predictions = [
        majority_class
        for _ in range(len(y_test))
    ]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "model": "Majority Baseline",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "majority_class": majority_class,
    }


# =========================================================
# Baseline 2 — TF-IDF + Logistic Regression
# =========================================================

def tfidf_baseline(
    x_train,
    x_test,
    y_train,
    y_test
):
    """
    Train a simple TF-IDF + Logistic Regression classifier.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
    )

    train_vectors = vectorizer.fit_transform(
        x_train
    )

    test_vectors = vectorizer.transform(
        x_test
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(
        train_vectors,
        y_train
    )

    predictions = model.predict(
        test_vectors
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate intent-classification baselines."
    )

    parser.add_argument(
        "--golden",
        default="data/golden_set/golden.csv",
        help="Path to the manually labelled golden set."
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    df = load_golden_data(
        args.golden
    )

    print(
        f"Loaded {len(df)} labelled examples."
    )

    print(
        f"Intents: {df['gold_intent'].nunique()}"
    )

    print()

    # -----------------------------------------------------
    # Train/test split
    # -----------------------------------------------------

    X = df["customer_text"]

    y = df["gold_intent"]

    # Stratification is only possible when every class
    # has enough examples.
    class_counts = y.value_counts()

    can_stratify = (
        len(class_counts) > 1
        and class_counts.min() >= 2
    )

    stratify = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    print(
        f"Train examples: {len(X_train)}"
    )

    print(
        f"Test examples: {len(X_test)}"
    )

    print()

    # -----------------------------------------------------
    # Baseline 1
    # -----------------------------------------------------

    majority_result = majority_baseline(
        y_train,
        y_test
    )

    # -----------------------------------------------------
    # Baseline 2
    # -----------------------------------------------------

    tfidf_result = tfidf_baseline(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)

    print()

    print(
        f"Majority Baseline"
    )

    print(
        f"  Majority class: "
        f"{majority_result['majority_class']}"
    )

    print(
        f"  Accuracy: "
        f"{majority_result['accuracy']:.4f}"
    )

    print(
        f"  Macro F1: "
        f"{majority_result['macro_f1']:.4f}"
    )

    print()

    print(
        f"TF-IDF + Logistic Regression"
    )

    print(
        f"  Accuracy: "
        f"{tfidf_result['accuracy']:.4f}"
    )

    print(
        f"  Macro F1: "
        f"{tfidf_result['macro_f1']:.4f}"
    )

    print()

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()