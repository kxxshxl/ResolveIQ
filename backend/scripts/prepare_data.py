"""raw (ticketing-system export) -> processed (internal schema, cleaned, PII-redacted JSONL).

Schema mapping
  description        -> complaint_text      (HTML stripped, whitespace normalised, PII redacted)
  issue_type         -> intent              ("Broadband Disconnection" -> broadband_disconnection)
  product_line       -> product             ("WiFi Router" -> wifi_router)
  priority P1..P4    -> severity            (critical/high/medium/low)
  customer_mood      -> sentiment           (WORRIED -> concerned, ...)
  agent_notes        -> resolution_steps
  closed_at          -> resolved_at
  scenario_id        -> metadata.scenario_id (ground truth for evaluation only; never used by retrieval)
KB: body (HTML) -> content (plain), procedure -> steps, topic -> category, labels -> tags,
    status published|retired -> active|deprecated
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.pii import redact  # noqa: E402
from app.core.text import normalize_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "processed"
SEVERITY = {"P1": "critical", "P2": "high", "P3": "medium", "P4": "low"}
SENTIMENT = {"ANGRY": "angry", "FRUSTRATED": "frustrated", "WORRIED": "concerned", "NEUTRAL": "neutral"}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def clean(text: str) -> str:
    return redact(normalize_text(text)).text


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def ticket(r: dict) -> dict:
    return {
        "ticket_id": r["ticket_id"], "complaint_text": clean(r["description"]), "intent": slug(r["issue_type"]),
        "product": slug(r["product_line"]), "severity": SEVERITY[r["priority"]], "sentiment": SENTIMENT[r["customer_mood"]],
        "resolution_steps": [clean(s) for s in r["agent_notes"]], "resolution_summary": clean(r["resolution_summary"]),
        "resolved_at": r["closed_at"], "metadata": {"scenario_id": r["scenario_id"], "cohort": r["cohort"], "created_at": r["created_at"]}}


def article(r: dict) -> dict:
    return {
        "article_id": r["kb_id"], "title": clean(r["title"]), "content": clean(re.sub(r"<li>", " ", r["body"])),
        "steps": [clean(s) for s in r["procedure"]], "category": slug(r["topic"]), "product": slug(r["product_line"]),
        "tags": r["labels"], "status": "deprecated" if r["status"] == "retired" else "active",
        "metadata": {"scenario_id": r["scenario_id"], "last_updated": r["last_updated"]}}


def main() -> None:
    stats = {}
    for src, dst, fn in (("tickets_raw", "tickets", ticket), ("articles_raw", "articles", article),
                         ("evolving_tickets_raw", "evolving_tickets", ticket), ("evolving_articles_raw", "evolving_articles", article)):
        rows = [fn(r) for r in read(RAW / f"{src}.jsonl")]
        write(PROCESSED / f"{dst}.jsonl", rows)
        stats[dst] = len(rows)
    leaked = sum(1 for r in read(PROCESSED / "tickets.jsonl") if re.search(r"@|\d{8,}", r["complaint_text"]))
    print("processed:", stats, "| rows still containing email/8+ digit runs:", leaked)


if __name__ == "__main__":
    main()
