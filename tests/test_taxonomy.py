from src.taxonomy import classify_rules

def test_refund():
    intent, conf = classify_rules("I need a refund for my order")
    assert intent == "refund"
    assert conf > 0.5

def test_delivery():
    intent, _ = classify_rules("My package is late")
    assert intent == "delivery"
