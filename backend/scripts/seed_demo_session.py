"""Populate a RUNNING API with a demo session: real resolutions of labelled complaints, plus SIMULATED agent feedback.

    python scripts/seed_demo_session.py --api http://127.0.0.1:8100 [--n 30] [--key <api key>]

What is real and what is not:
  * the resolutions are real: each labelled hand-written complaint is sent to /api/v1/resolve and goes through the whole pipeline (retrieval, LLM, validation);
  * the feedback is SIMULATED. No agent rated anything. The rating of each resolution is derived from the ground truth that the evaluation sets carry: a
    resolution is "helpful" when it cites a source from the complaint's true root-cause scenario and "not helpful" otherwise, with the matching reasons, rejected
    sources and (when the detected intent differs from the true one) the corrected intent. It exists so the Feedback & Quality view and the case history can be
    seen working; any number it produces describes this script, not agents.

Use it against a scratch database. Refuses to run twice against the same API unless --force (it would double the demo data).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import REPO_ROOT  # noqa: E402
from app.evaluation.datasets import load_jsonl  # noqa: E402


def call(api: str, method: str, path: str, key: str | None, body: dict | None = None):
    req = urllib.request.Request(api + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **({"X-API-Key": key} if key else {})})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8100")
    ap.add_argument("--key", default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--novel", type=int, default=0, help="also resolve this many complaints from classes the corpus does not cover (simulated agents reject any answer given)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if call(a.api, "GET", "/api/v1/cases?limit=1", a.key)["total"] and not a.force:
        print("the API already has cases; refusing to add demo data (use --force)")
        return 2
    data = REPO_ROOT / "data"
    scenario_of = {t["ticket_id"]: t["metadata"].get("scenario_id") for t in load_jsonl(data / "processed" / "tickets.jsonl")}
    scenario_of |= {x["article_id"]: x["metadata"].get("scenario_id") for x in load_jsonl(data / "processed" / "articles.jsonl")}
    rows = load_jsonl(data / "eval" / "gold.jsonl") + load_jsonl(data / "eval" / "gold_v2.jsonl") + load_jsonl(data / "eval" / "gold_blind.jsonl")
    step = max(1, len(rows) // max(1, a.n))
    chosen = rows[::step][: a.n] if a.n else []
    ok = bad = 0
    for i, row in enumerate(chosen):
        r = call(a.api, "POST", "/api/v1/resolve", a.key, {"complaint": row["text"]})
        cited = [c["source_id"] for c in r["citations"]]
        right = [c for c in cited if scenario_of.get(c) == row["scenario_id"]]
        helpful = r["status"] == "resolved" and bool(right)
        body: dict = {"request_id": r["request_id"], "rating": "helpful" if helpful else "not_helpful"}
        if not helpful:
            wrong = [c for c in cited if scenario_of.get(c) != row["scenario_id"]]
            reasons = ["irrelevant_source"] if wrong else (["steps_missing"] if not r["resolution"]["steps"] else ["too_vague"])
            if r["classification"]["intent"] != row["intent"]:
                reasons.append("wrong_intent")
                body["corrected_intent"] = row["intent"]
            body.update(reasons=reasons, rejected_sources=wrong[:3])
            if i % 4 == 0 and r["resolution"]["steps"]:
                body["edited_steps"] = [s["text"] for s in r["resolution"]["steps"][:2]] + ["Escalate to the specialist team if the fault returns."]
        call(a.api, "POST", "/api/v1/feedback", a.key, body)
        ok, bad = ok + helpful, bad + (not helpful)
        print(f"{i + 1:>2}/{len(chosen)} {r['status']:<9} intent={r['classification']['intent']:<24} -> simulated rating: {body['rating']}")
    novel = load_jsonl(data / "eval" / "novel_stream.jsonl")
    for i, row in enumerate(novel[:: max(1, len(novel) // a.novel)][: a.novel] if a.novel else []):
        r = call(a.api, "POST", "/api/v1/resolve", a.key, {"complaint": row["text"]})
        if r["status"] == "abstained":
            print(f"novel {i + 1}: abstained (no rating: declining was the right call)")
            continue
        cited = [c["source_id"] for c in r["citations"]]       # nothing in the corpus covers this class, so whatever was cited is wrong
        call(a.api, "POST", "/api/v1/feedback", a.key, {"request_id": r["request_id"], "rating": "not_helpful", "reasons": ["irrelevant_source", "steps_incorrect"] if i % 2 else ["irrelevant_source"],
                                                         "rejected_sources": cited[:3], "comment": f"covers a different problem ({row['intent'].replace('_', ' ')})"})
        bad += 1
        print(f"novel {i + 1}: answered as {r['classification']['intent']} -> simulated rating: not_helpful")
    print(f"done: {ok} helpful, {bad} not helpful (simulated from ground truth)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
