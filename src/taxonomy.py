# from __future__ import annotations
# import re

# KEYWORDS = {
#     "delivery": ["delivery", "delivered", "shipping", "shipment", "arrive", "late", "package"],
#     "refund": ["refund", "money back", "reimburse", "chargeback"],
#     "payment": ["payment", "charged", "charge", "card", "billing", "transaction"],
#     "account": ["account", "login", "password", "locked", "sign in", "username"],
#     "cancellation": ["cancel", "cancellation"],
#     "technical_issue": ["error", "bug", "not working", "broken", "crash", "issue", "problem"],
#     "subscription": ["subscription", "renew", "renewal", "plan", "membership"],
#     "complaint": ["terrible", "awful", "angry", "unhappy", "complaint", "disappointed"],
#     "information_request": ["how do", "where", "when", "what", "can i", "is there"],
#     "other": [],
# }

# def classify_rules(text: str):
#     t = text.lower()
#     scores = {k: sum(1 for w in ws if w in t) for k, ws in KEYWORDS.items()}
#     best = max(scores, key=scores.get)
#     if scores[best] == 0:
#         return "other", 0.35
#     total = sum(scores.values())
#     conf = min(0.95, 0.55 + scores[best] / max(total, 1) * 0.4)
#     return best, conf

# def taxonomy():
#     return list(KEYWORDS)




from __future__ import annotations

import re


INTENTS = {
    "ios_update_issue": [
        "ios",
        "ios 11",
        "ios11",
        "update",
        "updated",
        "updating",
        "software update",
        "upgrade",
        "install update",
        "cannot update",
        "can't update",
        "wont update",
    ],

    "battery_issue": [
        "battery",
        "battery life",
        "battery drain",
        "battery dying",
        "battery dies",
        "charging",
        "charge",
        "overheating",
        "gets hot",
        "heating",
    ],

    "device_performance": [
        "slow",
        "slower",
        "freezing",
        "freeze",
        "lag",
        "lagging",
        "crash",
        "crashing",
        "glitch",
        "glitches",
        "not responding",
        "keeps restarting",
    ],

    "screen_display_issue": [
        "screen",
        "display",
        "brightness",
        "dark screen",
        "black screen",
        "touch screen",
        "pixels",
        "dots",
        "screen flicker",
    ],

    "app_issue": [
        "app",
        "apps",
        "application",
        "music app",
        "facetime",
        "itunes",
        "app store",
        "not opening",
        "won't open",
        # "not working",
        # "keeps crashing",
    ],

    "icloud_issue": [
        "icloud",
        "icloud photo",
        "icloud photos",
        "photo library",
        "photos disappeared",
        "sync",
        "backup",
        "icloud backup",
    ],

    "camera_issue": [
        "camera",
        "camera app",
        "focus",
        "autofocus",
        "rear camera",
        "front camera",
        "camera not working",
    ],

    "connectivity_issue": [
        "wifi",
        "wi-fi",
        "network",
        "internet",
        "bluetooth",
        "cellular",
        "signal",
        "connection",
        "connect",
    ],

    "account_security": [
        "apple id",
        "account",
        "password",
        "phishing",
        "scam",
        "fake message",
        "security",
        "hacked",
        "login",
        "sign in",
        "verification",
    ],

    "hardware_repair": [
        "repair",
        "service",
        "broken",
        "damaged",
        "replace",
        "replacement",
        "repair shop",
        "genius bar",
        "fix my phone",
    ],

    "product_information": [
        "trade in",
        "trade-in",
        "tradein",
        "iphone x",
        "iphone 8",
        "price",
        "cost",
        "buy",
        "purchase",
        "available",
        "availability",
    ],

    "general_support": [
        "help",
        "question",
        "please help",
        "need help",
        "can you help",
        "what should i do",
        "how do i",
    ],
}


def classify_rules(text: str):

    text = str(text).lower()

    scores = {}

    for intent, keywords in INTENTS.items():

        score = 0

        for keyword in keywords:

            if keyword in text:
                score += 1

        scores[intent] = score

    best_intent = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_intent]

    if best_score == 0:

        return "general_support", 0.35

    total_score = sum(
        scores.values()
    )

    confidence = (
        0.55
        + (
            best_score /
            max(total_score, 1)
        ) * 0.40
    )

    confidence = min(
        confidence,
        0.95
    )

    return (
        best_intent,
        round(confidence, 3)
    )


def taxonomy():

    return list(
        INTENTS.keys()
    )


if __name__ == "__main__":

    examples = [
        "My iPhone battery is dying very quickly",
        "iOS update is not working",
        "My camera won't focus",
        "My iPhone screen is broken",
        "iCloud deleted all my photos",
        "WiFi is not working",
        "Someone sent me a phishing message",
        "My phone keeps freezing",
    ]

    for example in examples:

        intent, confidence = classify_rules(
            example
        )

        print(
            f"{example}\n"
            f"  -> {intent} "
            f"({confidence})"
        )