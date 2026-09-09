import pandas as pd

from src.taxonomy import classify_rules
from src.retrieval import Retriever


def test_battery_intent():
    intent, confidence = classify_rules(
        "My iPhone battery is draining very quickly"
    )

    assert intent == "battery_issue"
    assert confidence >= 0.70


def test_camera_intent():
    intent, confidence = classify_rules(
        "My iPhone camera is not working"
    )

    assert intent == "camera_issue"
    assert confidence >= 0.70


def test_connectivity_intent():
    intent, confidence = classify_rules(
        "My WiFi keeps disconnecting"
    )

    assert intent == "connectivity_issue"
    assert confidence >= 0.70


def test_security_intent():
    intent, confidence = classify_rules(
        "Someone is trying to access my Apple account"
    )

    assert intent == "account_security"
    assert confidence >= 0.70


def test_retrieval_returns_results():

    history = pd.DataFrame(
        {
            "conversation_id": ["1", "2"],
            "customer_text": [
                "My iPhone battery drains quickly",
                "My WiFi keeps disconnecting",
            ],
            "historical_reply": [
                "Please check Battery settings.",
                "Please check your WiFi connection.",
            ],
        }
    )

    retriever = Retriever(history)

    results = retriever.search(
        "My iPhone battery is draining quickly",
        k=1,
    )

    assert isinstance(results, list)
    assert len(results) == 1

    result = results[0]

    assert isinstance(result, dict)

    assert "conversation_id" in result
    assert "customer_text" in result
    assert "historical_reply" in result
    assert "similarity" in result