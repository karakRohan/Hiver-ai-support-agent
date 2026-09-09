from __future__ import annotations
import re

KEYWORDS = {
    "delivery": ["delivery", "delivered", "shipping", "shipment", "arrive", "late", "package"],
    "refund": ["refund", "money back", "reimburse", "chargeback"],
    "payment": ["payment", "charged", "charge", "card", "billing", "transaction"],
    "account": ["account", "login", "password", "locked", "sign in", "username"],
    "cancellation": ["cancel", "cancellation"],
    "technical_issue": ["error", "bug", "not working", "broken", "crash", "issue", "problem"],
    "subscription": ["subscription", "renew", "renewal", "plan", "membership"],
    "complaint": ["terrible", "awful", "angry", "unhappy", "complaint", "disappointed"],
    "information_request": ["how do", "where", "when", "what", "can i", "is there"],
    "other": [],
}

def classify_rules(text: str):
    t = text.lower()
    scores = {k: sum(1 for w in ws if w in t) for k, ws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other", 0.35
    total = sum(scores.values())
    conf = min(0.95, 0.55 + scores[best] / max(total, 1) * 0.4)
    return best, conf

def taxonomy():
    return list(KEYWORDS)
