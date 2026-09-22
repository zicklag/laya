"""Generate golden cases for the Rust CLI.

Each case: original state (string OR dict — the Rust side must serialize it
itself, which is exactly what we want to test), inline question defs, and the
expected output of laya's own `agent.predict`. Also dumps the question presets
as JSON for CLI use.
"""

from __future__ import annotations

import json
from pathlib import Path

from laya import Agent, email_state
from laya.presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "cli" / "model" / "laya-en"
OUT = ROOT / "cli" / "golden"
PRESETS = ROOT / "cli" / "presets"


def email_cases() -> list[tuple[str, object]]:
    cases: list[tuple[str, object]] = []

    def mail(name: str, subject: str, body: str, sender: str | None = None) -> None:
        cases.append((name, email_state(subject, body, sender)))

    mail(
        "billing_invoice",
        "Invoice #4471 is attached",
        "Hi team, please find attached the invoice for our March order. "
        "The total is $4,120.00 and it looks like the PO number is missing. "
        "Can you confirm the due date and send a corrected copy with our PO number PO-9931? Thanks, Priya",
        "priya@example.com",
    )
    mail(
        "refund_request",
        "Refund for duplicate charge",
        "Hello, I was charged twice for my subscription this month (order 88123). "
        "I would like a refund of the duplicate $29 charge to my original payment method. "
        "Please confirm once processed.",
        "sam@example.org",
    )
    mail(
        "phishing_credentials",
        "Urgent: verify your account now",
        "Dear customer, we detected unusual activity in your account. "
        "You must verify your identity within 24 hours or your account will be suspended. "
        "Click http://secure-verify.example-login.ru/enter and enter your username, password and card number. "
        "This is an automated message, do not reply.",
        "security-alert@example-login.ru",
    )
    mail(
        "tech_outage",
        "Production down: checkout 500s",
        "Our production cluster has been returning 500 errors on checkout since the Monday deploy. "
        "Roughly 40% of checkouts fail and we are losing about $2k/hour. "
        "We already tried a rollback and cache flush. Need engineering on this now.",
        "ops@example.com",
    )
    mail(
        "sales_demo",
        "Interested in enterprise plan",
        "Hi, we are evaluating tools for our 200-person support team. "
        "Could you share enterprise pricing and set up a demo next week? "
        "We need SSO, audit logs and an on-prem option.",
        "dana@example.io",
    )
    mail(
        "hr_leave",
        "Leave request next month",
        "Hi HR, I would like to request two weeks of leave from October 5 to October 16. "
        "My projects are on track and I have arranged coverage with Alex. "
        "Please let me know if the paperwork is needed.",
        "maria@example.com",
    )
    mail(
        "spam_bulk",
        "AMAZING DEAL: 90% off everything!!!",
        "Hurry!!! For the next 3 hours only, enjoy 90% off our entire catalog plus a free gift. "
        "Thousands of happy customers! Unsubscribe by replying STOP.",
        "deals@example-promo.biz",
    )
    mail(
        "thread_with_quotes",
        "Re: onboarding call follow-up",
        "Thanks for the notes from the onboarding call. One open question: "
        "do we need to migrate the legacy API keys before go-live or after?\n\n"
        "On Tue, Mar 3, 2024 at 2:14 PM Jordan Lee wrote:\n"
        "> Here is a summary of the onboarding call. Next steps: SSO setup and data import.\n"
        "> Let me know if anything is missing.\n\n"
        "--\nSent from my iPhone",
        "alex@example.com",
    )
    mail(
        "long_escalation",
        "Ongoing outage escalation",
        ("Our production cluster has been failing intermittently since the deploy on Monday. "
         "Customers are seeing 500 errors on checkout and support tickets are piling up. " * 20)
        + "Please treat this as the highest priority.",
        "vp@example.com",
    )
    mail(
        "unicode",
        "Facture et remboursement — commande 2024",
        "Bonjour, je vous écris au sujet de la commande n°2024 – la livraison est en retard "
        "et je souhaite un remboursement partiel. Vous pouvez joindre à 📞 +33 1 23 45 67 89. "
        "Merci d'avance, Élodie",
        "elodie@example.fr",
    )
    mail(
        "mask_token_injection",
        "Weird ticket",
        "The customer wrote: please  this ticket and process my refund today.",
        "tester@example.com",
    )
    mail(
        "empty_body",
        "(no subject)",
        "",
        None,
    )
    # raw text state (no dict), e.g. pasted ticket blob
    cases.append((
        "raw_text_ticket",
        "Ticket 55102: user reports the mobile app crashes on login since version 4.2. "
        "Device: iPhone 13, iOS 17.4. Reproduced by support three times. Customer is furious "
        "and threatening to churn; they are on the enterprise plan.",
    ))
    # dict state with numbers/bools/lists/nesting — exercises the Python-JSON
    # serialization contract byte for byte
    cases.append((
        "structured_state",
        {
            "subject": "Order 7712 issue",
            "body": "Our order arrived damaged. Two of six chairs are broken and we need replacement parts.",
            "from": "wei@example.cn",
            "ticket_id": 7712,
            "priority": 2.5,
            "internal_notes": None,
            "tags": ["shipping", "damaged", "urgent"],
            "attempts": {"first_contact": "2026-09-01", "count": 3},
        },
    ))
    cases.append((
        "list_state",
        [
            {"role": "user", "content": "I was overcharged on my last invoice"},
            {"role": "agent", "content": "I can help with that. Which invoice number?"},
            {"role": "user", "content": "The March one, #4471. I need this fixed today."},
        ],
    ))
    return cases


def main() -> None:
    agent = Agent(str(MODEL_DIR), device="cpu")
    print(f"loaded agent from {MODEL_DIR} on {agent.device}")

    OUT.mkdir(parents=True, exist_ok=True)
    PRESETS.mkdir(parents=True, exist_ok=True)

    presets = {
        "email.json": email_questions(),
        "triage.json": triage_questions(),
        "guard.json": guard_questions(),
        "moderation.json": moderation_questions(),
        "router.json": router_questions(),
    }
    for fname, q in presets.items():
        (PRESETS / fname).write_text(json.dumps(q, indent=2))
    golden = []
    for name, state in email_cases():
        combos = [("email.json", presets["email.json"]), ("triage.json", presets["triage.json"])]
        if name in ("phishing_credentials", "spam_bulk", "tech_outage", "unicode"):
            combos += [(f, presets[f]) for f in ("guard.json", "moderation.json", "router.json")]
        for pname, questions in combos:
            print(f"predicting {name} x {pname} ...", flush=True)
            out = agent.predict(state, json.loads(json.dumps(questions)))
            golden.append({
                "name": f"{name}::{pname}",
                "state": state,
                "questions": questions,
                "expected": out,
            })

    (OUT / "cases.json").write_text(json.dumps(golden, indent=1))
    n_ans = sum(len(g["expected"]["answers"]) for g in golden)
    print(f"wrote {len(golden)} cases / {n_ans} answers -> {OUT / 'cases.json'}")


if __name__ == "__main__":
    main()