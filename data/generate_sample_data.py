"""
generate_sample_data.py
------------------------
Generates a small, schema-identical STAND-IN for the real Kaggle
"Customer Support on Twitter" dataset (thoughtvector/customer-support-on-twitter).

WHY THIS FILE EXISTS
=====================
This assignment was built in a sandboxed environment with no network access,
so the real ~2.8M-row `twcs.csv` (available at
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
could not be downloaded. To keep the pipeline fully runnable and reviewable
end-to-end, this script synthesizes a small dataset with:

  - the exact same columns as the real file
    (tweet_id, author_id, inbound, created_at, text,
     response_tweet_id, in_response_to_tweet_id)
  - the same *structure*: inbound customer tweets that are answered by a
    brand support handle, forming (customer -> agent) conversation pairs
  - realistic intent categories and phrasing patterns for a fictitious
    brand "AppleSupport" (kept as the Kaggle default brand name so the
    rest of the code, filters, and docs are drop-in compatible with the
    real dataset)

**This is clearly a synthetic placeholder, not real data.** All metrics
computed downstream on this sample are illustrative of the *pipeline*
working correctly, not of real-world model performance. See
reports/report.md -> "Misleading headline metric" section, and the README,
for how to swap in the real Kaggle CSV (drop it at data/twcs.csv and the
pipeline uses it automatically; no code changes required).
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

BRAND = "AppleSupport"

# Each entry: (intent_label, list of (customer_msg, agent_msg) templates)
INTENT_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "Refund": [
        ("I want a refund for my order, it never arrived.",
         "So sorry about that! Please DM us your order number and we'll process a refund."),
        ("Can I get my money back for the broken charger?",
         "We'd like to help. Send us your order number in a DM and we'll start the refund."),
        ("This app charged me and I want a refund now.",
         "We understand the frustration. Please share your receipt via DM so we can review the refund."),
    ],
    "Order Delay": [
        ("My order has been delayed for over a week now.",
         "Sorry for the wait! Can you DM your order number so we can check the shipping status?"),
        ("Where is my package? It was supposed to arrive yesterday.",
         "Let's look into this together, please send your tracking number via DM."),
        ("Still waiting on my order, no updates at all.",
         "Apologies for the delay. DM us your order ID and we'll get you an update."),
    ],
    "Account Locked": [
        ("My account got locked and I can't log in.",
         "Sorry about that! Please DM us the email on file so we can help unlock it."),
        ("I've been locked out of my account for no reason.",
         "We can help with that. Send us a DM with your account email."),
        ("Account locked after I changed my phone number.",
         "Let's fix this. Please DM your account details so we can verify and unlock it."),
    ],
    "Password Reset": [
        ("I can't reset my password, the link isn't working.",
         "Sorry for the trouble! Try requesting a new link, and if it still fails, DM us."),
        ("Forgot my password and the reset email never came.",
         "Please check your spam folder first, and DM us if you still don't see it."),
        ("Password reset page keeps giving an error.",
         "That's frustrating, let's help. DM us the email associated with your account."),
    ],
    "Delivery Issue": [
        ("My package arrived damaged and unusable.",
         "So sorry to hear that! Please DM us photos and your order number."),
        ("Wrong item was delivered to my address.",
         "Apologies for the mix up. DM us your order number so we can send the correct item."),
        ("Delivery driver left my package at the wrong house.",
         "That's not okay, let's sort it out. Please DM your order number."),
    ],
    "Billing Problem": [
        ("I've been charged twice for the same order.",
         "Sorry about that! Please DM your order number so we can review the double charge."),
        ("There's an unfamiliar charge on my statement.",
         "Let's look into this. DM us the last 4 digits of the card and order details."),
        ("My subscription billed me a different amount than expected.",
         "We'll help clarify this. Please DM your account email so we can check the billing."),
    ],
    "Technical Issue": [
        ("The app keeps crashing every time I open it.",
         "Sorry for the inconvenience! What device/OS version are you on? Please DM us."),
        ("I can't get the software update to install.",
         "Let's troubleshoot. Please DM your device model and current OS version."),
        ("Bluetooth won't connect to my device anymore.",
         "That's odd, let's fix it. Please DM your device model so we can help."),
    ],
    "Subscription Cancellation": [
        ("I want to cancel my subscription immediately.",
         "We're sorry to see you go. Please DM your account email and we'll assist with cancellation."),
        ("How do I cancel my monthly plan?",
         "You can cancel in Settings > Subscriptions, or DM us your account email for help."),
        ("Please cancel my subscription, I no longer need it.",
         "Understood, we'll help. Please DM your account email to confirm cancellation."),
    ],
}


def generate(n_per_intent: int = 25, out_path: str = "data/sample_twcs.csv") -> None:
    rows = []
    tweet_id = 1
    start_time = datetime(2024, 1, 1)

    for intent, templates in INTENT_TEMPLATES.items():
        for i in range(n_per_intent):
            cust_msg, agent_msg = random.choice(templates)
            # add light variation so rows aren't exact duplicates
            suffix = random.choice(["", " Please help.", " Thanks.", " This is urgent.", ""])
            cust_text = f"@{BRAND} {cust_msg}{suffix}"
            agent_text = f"@customer{tweet_id} {agent_msg} ^AB"

            created_customer = start_time + timedelta(minutes=tweet_id * 7, seconds=random.randint(0, 59))
            created_agent = created_customer + timedelta(minutes=random.randint(2, 45))

            customer_tweet_id = tweet_id
            agent_tweet_id = tweet_id + 1

            rows.append({
                "tweet_id": customer_tweet_id,
                "author_id": f"cust_{customer_tweet_id}",
                "inbound": True,
                "created_at": created_customer.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                "text": cust_text,
                "response_tweet_id": agent_tweet_id,
                "in_response_to_tweet_id": "",
                "intent_true": intent,  # ONLY present in synthetic data, used to sanity-check clustering
            })
            rows.append({
                "tweet_id": agent_tweet_id,
                "author_id": BRAND,
                "inbound": False,
                "created_at": created_agent.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                "text": agent_text,
                "response_tweet_id": "",
                "in_response_to_tweet_id": customer_tweet_id,
                "intent_true": "",
            })
            tweet_id += 2

    random.shuffle(rows)  # real dataset isn't neatly grouped by conversation

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tweet_id", "author_id", "inbound", "created_at", "text",
                  "response_tweet_id", "in_response_to_tweet_id", "intent_true"]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic rows ({len(rows)//2} conversations) to {out_file}")


if __name__ == "__main__":
    generate()
