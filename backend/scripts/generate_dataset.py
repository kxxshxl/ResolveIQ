"""Deterministically generate the synthetic raw corpus + evaluation sets (seeded; safe to re-run).

Outputs
  data/raw/tickets_raw.jsonl, articles_raw.jsonl            ticketing-system-style export (messy: HTML, PII, caps)
  data/raw/evolving_*.jsonl, evolving_taxonomy.json         classes that do NOT exist at launch
  data/eval/queries.jsonl                                   held-out paraphrase queries (val/test split)
  data/eval/gold.jsonl, ood.jsonl, evolving_queries.jsonl   hand-written / out-of-domain / new-class queries
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datagen.scenarios import EVOLVING_SCENARIOS, EVOLVING_TAXONOMY, GOLD, IMPACT, OOD, SCENARIOS, TONE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW, EVAL = ROOT / "data" / "raw", ROOT / "data" / "eval"
PRIORITY = {"critical": "P1", "high": "P2", "medium": "P3", "low": "P4"}
MOOD = {"angry": "ANGRY", "frustrated": "FRUSTRATED", "concerned": "WORRIED", "neutral": "NEUTRAL"}
PRODUCT_LINE = {"broadband": "Broadband", "mobile": "Mobile", "wifi_router": "WiFi Router", "tv": "TV", "account": "Account"}
PII_SNIPPETS = ["My account number is 48201937.", "You can reach me on 07911 123456.", "Email me at jane.doe84@example.com.",
                "Customer ID: 55120984.", "My card is 4111 1111 1111 1111 if you need it.", "Call me back on +44 7700 900123."]
CLOSERS = ["", "", " Customer confirmed service restored.", " Customer advised to call back if it recurs."]


def compose(rng: random.Random, symptom: str, severity: str, tone: str, impact_idx: int, tone_idx: int) -> str:
    imp, tn = IMPACT[severity][impact_idx], TONE[tone][tone_idx]
    parts = rng.choice([[symptom, imp, tn], [tn, symptom, imp], [symptom, tn, imp]])
    text = " ".join(parts)
    r = rng.random()
    if r < 0.2:
        text = text.lower()
    elif r < 0.3:
        text = text.rstrip(".") + " pls"
    return text


def slug_title(s: str) -> str:
    return s.replace("_", " ").title()


def make_tickets(rng: random.Random, scenarios: list[dict], per_scenario: int, start_id: int, tag: str) -> list[dict]:
    rows, n = [], start_id
    base = datetime(2025, 1, 6, tzinfo=timezone.utc)
    for sc in scenarios:
        for i in range(per_scenario):
            sev = sc["sev"][i % len(sc["sev"])]
            tone = rng.choice(list(TONE))
            text = compose(rng, sc["symptoms"][i % 5], sev, tone, rng.randrange(4), rng.randrange(4))
            if rng.random() < 0.2:
                text += " " + rng.choice(PII_SNIPPETS)
            if rng.random() < 0.1:
                text = f"<p>{text}</p><br/>"
            steps = list(sc["steps"]) if i % 3 else list(sc["steps"])[:4]
            closed = base + timedelta(days=rng.randrange(0, 520), hours=rng.randrange(24))
            rows.append({
                "ticket_id": f"TKT-{n}", "created_at": (closed - timedelta(hours=rng.randrange(2, 96))).isoformat(),
                "closed_at": closed.isoformat(), "description": text, "issue_type": slug_title(sc["intent"]),
                "product_line": PRODUCT_LINE[sc["product"]], "priority": PRIORITY[sev], "customer_mood": MOOD[tone],
                "agent_notes": steps, "resolution_summary": sc["summary"] + rng.choice(CLOSERS),
                "scenario_id": sc["id"], "cohort": tag})
            n += 1
    return rows


def make_articles(scenarios: list[dict], start: int) -> list[dict]:
    out = []
    for i, sc in enumerate(scenarios, start):
        steps_html = "".join(f"<li>{s}</li>" for s in sc["steps"])
        out.append({
            "kb_id": f"KB-{i:03d}", "title": sc["title"], "body": f"<p>{sc['kb']}</p><ol>{steps_html}</ol>",
            "procedure": sc["steps"], "topic": slug_title(sc["intent"]), "product_line": PRODUCT_LINE[sc["product"]],
            "labels": [sc["intent"], sc["product"]], "last_updated": "2026-03-01T00:00:00+00:00",
            "status": "published", "scenario_id": sc["id"]})
    return out


def make_queries(rng: random.Random, scenarios: list[dict], prefix: str) -> list[dict]:
    out = []
    for sc in scenarios:
        for sym_idx in (5, 6, 7):
            for w in (4, 5):
                sev = rng.choice(sc["sev"])
                tone = rng.choice(list(TONE))
                out.append({
                    "qid": f"{prefix}-{sc['id']}-{sym_idx}-{w}", "split": "val" if sym_idx == 5 else "test",
                    "text": compose(rng, sc["symptoms"][sym_idx], sev, tone, w, w), "scenario_id": sc["id"],
                    "intent": sc["intent"], "product": sc["product"], "severity": sev, "sentiment": tone})
    return out


def dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def main() -> None:
    rng = random.Random(1337)
    by_id = {s["id"]: s for s in SCENARIOS}
    tickets = make_tickets(rng, SCENARIOS, 10, 1001, "launch")
    articles = make_articles(SCENARIOS, 1)
    # A retired article that is lexically/semantically close to a live one: must never be retrieved.
    es = by_id["evening_drops"]
    articles.append({
        "kb_id": "KB-900", "title": "Evening disconnections - weekly line profile reset (superseded)",
        "body": "<p>Customers whose broadband disconnects every evening should have their line profile reset to the default "
                "once a week. Ask them to leave the router off overnight.</p>", "procedure": ["Reset the line profile every week.", "Ask the customer to leave the router off overnight."],
        "topic": slug_title(es["intent"]), "product_line": "Broadband", "labels": [es["intent"], "broadband"],
        "last_updated": "2023-05-01T00:00:00+00:00", "status": "retired", "scenario_id": "evening_drops"})
    dump(RAW / "tickets_raw.jsonl", tickets)
    dump(RAW / "articles_raw.jsonl", articles)

    ev_t = make_tickets(rng, EVOLVING_SCENARIOS, 6, 2001, "evolving")
    ev_a = make_articles(EVOLVING_SCENARIOS, 101)
    dump(RAW / "evolving_tickets_raw.jsonl", ev_t)
    dump(RAW / "evolving_articles_raw.jsonl", ev_a)
    (RAW / "evolving_taxonomy.json").write_text(json.dumps(EVOLVING_TAXONOMY, indent=2), encoding="utf-8")

    queries = make_queries(rng, SCENARIOS, "q")
    dump(EVAL / "queries.jsonl", queries)
    dump(EVAL / "evolving_queries.jsonl", make_queries(rng, EVOLVING_SCENARIOS, "ev"))
    gold = []
    for i, (sid, text, sev, tone) in enumerate(GOLD, 1):
        sc = by_id[sid]
        gold.append({"qid": f"gold-{i:02d}", "split": "gold", "text": text, "scenario_id": sid, "intent": sc["intent"],
                     "product": sc["product"], "severity": sev, "sentiment": tone})
    dump(EVAL / "gold.jsonl", gold)
    dump(EVAL / "ood.jsonl", [{"qid": f"ood-{i:02d}", "text": t, "expected": "abstain"} for i, t in enumerate(OOD, 1)])
    print(f"tickets={len(tickets)} articles={len(articles)} evolving_tickets={len(ev_t)} queries={len(queries)} "
          f"gold={len(gold)} ood={len(OOD)}")


if __name__ == "__main__":
    main()
