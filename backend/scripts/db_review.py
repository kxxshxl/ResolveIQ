"""Query-plan review at a realistic volume: EXPLAIN (ANALYZE, BUFFERS) of the queries the console and workers run, plus a with/without comparison for an index
that was added by the database review.

    python scripts/db_review.py [--requests 20000] [--feedback 50000]

Builds a throw-away copy of the application database, fills `resolution_requests` and `feedback` with synthetic rows spread over 30 days (so the plans are those of a
database that has been in use, not of the 250-ticket demo corpus), runs ANALYZE, and prints / writes the plans' essentials. Nothing touches the real database.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import psycopg
from psycopg.types.json import Jsonb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import REPO_ROOT, get_settings  # noqa: E402

DB = "resolveiq_db_review"
INTENTS = ["billing_dispute", "slow_speed", "device_issue", "sim_mobile_connectivity", "account_authentication", "broadband_disconnection", "plan_change", "service_outage", "installation_issue"]


def swap(url: str, db: str) -> str:
    return urlunparse(urlparse(url)._replace(path=f"/{db}"))


def nodes(plan: dict, out: list | None = None) -> list[str]:
    out = [] if out is None else out
    desc = plan["Node Type"] + (f" on {plan['Relation Name']}" if "Relation Name" in plan else "") + (f" using {plan['Index Name']}" if "Index Name" in plan else "")
    out.append(desc)
    for p in plan.get("Plans", []):
        nodes(p, out)
    return out


def explain(conn, name: str, sql: str, params=()) -> dict:
    row = conn.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params).fetchone()[0][0]
    plan = row["Plan"]
    return {"query": name, "execution_ms": round(row["Execution Time"], 2), "planning_ms": round(row["Planning Time"], 2), "rows": plan["Actual Rows"],
            "shared_hit": plan.get("Shared Hit Blocks", 0), "shared_read": plan.get("Shared Read Blocks", 0), "plan": nodes(plan)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=20000)
    ap.add_argument("--feedback", type=int, default=50000)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "eval" / "results" / "db_query_plans.json"))
    a = ap.parse_args()
    s = get_settings()
    root = swap(s.database_admin_url, "postgres")
    with psycopg.connect(root, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")
        c.execute(f"CREATE DATABASE {DB} TEMPLATE {urlparse(s.database_admin_url).path.lstrip('/')}")
    res: dict = {"setup": {"requests": a.requests, "feedback": a.feedback, "synthetic": True, "note": "synthetic rows spread over 30 days; plans are for a database that has been used, not for the demo corpus"}}
    try:
        with psycopg.connect(swap(s.database_admin_url, DB), autocommit=True) as conn:
            rng = random.Random(5)
            t = time.perf_counter()
            conn.execute("DELETE FROM feedback; DELETE FROM resolution_requests")
            ids = [str(uuid.uuid4()) for _ in range(a.requests)]
            with conn.cursor().copy("COPY resolution_requests (request_id, trace_id, complaint, classification, retrieved, result, status, confidence, latency_ms, created_at) FROM STDIN") as cp:
                for i, rid in enumerate(ids):
                    intent = rng.choice(INTENTS)
                    status = rng.choices(["resolved", "abstained", "degraded", "unreliable"], [80, 12, 6, 2])[0]
                    cp.write_row((rid, "t", f"synthetic complaint {i} about {intent.replace('_', ' ')}", json.dumps({"intent": intent, "product": rng.choice(["broadband", "mobile", "tv"])}),
                                  json.dumps([{"source_id": f"TKT-{rng.randint(1000, 1250)}", "score": 0.7}]), json.dumps({"evidence": {"confidence": round(rng.random(), 2)}, "citations": [{"source_id": f"TKT-{rng.randint(1000, 1250)}"}]}),
                                  status, 0.7, rng.randint(40, 9000), f"2026-09-{rng.randint(6, 30):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00+00"))
            with conn.cursor().copy("COPY feedback (request_id, rating, created_at) FROM STDIN") as cp:
                for i in range(a.feedback):
                    cp.write_row((ids[i % a.requests], rng.choice(["helpful", "not_helpful"]), f"2026-09-{rng.randint(6, 30):02d} 12:00:00+00"))
            conn.execute("ANALYZE")
            res["setup"]["load_seconds"] = round(time.perf_counter() - t, 1)

            plans = [
                explain(conn, "case list: newest 25", "SELECT request_id, created_at, status FROM resolution_requests ORDER BY created_at DESC LIMIT 25"),
                explain(conn, "case list: abstained, newest 25", "SELECT request_id FROM resolution_requests WHERE status = 'abstained' ORDER BY created_at DESC LIMIT 25"),
                explain(conn, "cases of one intent, newest 25", "SELECT request_id FROM resolution_requests WHERE classification->>'intent' = 'billing_dispute' ORDER BY created_at DESC LIMIT 25"),
                explain(conn, "drift: last 24 h window", "SELECT status FROM resolution_requests WHERE created_at <= now() AND created_at > now() - make_interval(hours => 24)"),
                explain(conn, "timeline: requests per day, 14 days", "SELECT date_trunc('day', created_at) b, count(*), count(*) FILTER (WHERE status='abstained') FROM resolution_requests WHERE created_at > now() - interval '14 days' GROUP BY 1"),
                explain(conn, "intent mix per day, 14 days", "SELECT date_trunc('day', created_at) b, classification->>'intent', count(*) FROM resolution_requests WHERE created_at > now() - interval '14 days' GROUP BY 1, 2"),
                explain(conn, "feedback joined to requests, 30 days", "SELECT f.rating, r.classification->>'intent' FROM feedback f JOIN resolution_requests r ON r.request_id = f.request_id WHERE f.created_at > now() - interval '30 days'"),
                explain(conn, "latest rating per case (list subquery)", "SELECT (SELECT f.rating FROM feedback f WHERE f.request_id = r.request_id ORDER BY f.created_at DESC LIMIT 1) FROM (SELECT request_id FROM resolution_requests ORDER BY created_at DESC LIMIT 25) r"),
                explain(conn, "embedding backfill candidates", "SELECT request_id FROM resolution_requests WHERE created_at > now() - interval '7 days' AND embedding IS NULL ORDER BY created_at DESC LIMIT 800"),
                explain(conn, "ticket dense search (250 tickets: expect a scan)", "SELECT ticket_id FROM tickets t JOIN ticket_embeddings e ON e.ticket_pk = t.id WHERE e.model = %s ORDER BY e.embedding <=> (SELECT embedding FROM ticket_embeddings LIMIT 1) LIMIT 5", (s.embedding_model,)),
                explain(conn, "last discovery job", "SELECT max(created_at) FROM jobs WHERE kind = 'discover_classes'"),
            ]
            res["plans"] = plans

            # an index added by the review: feedback.request_id is a foreign key (ON DELETE SET NULL); without an index each deleted request scans the feedback table
            victims = ids[:300]
            conn.execute("DROP INDEX IF EXISTS feedback_request_idx")
            t = time.perf_counter()
            conn.execute("DELETE FROM resolution_requests WHERE request_id = ANY(%s::uuid[])", (victims,))
            without = time.perf_counter() - t
            conn.execute("CREATE INDEX feedback_request_idx ON feedback (request_id)")
            conn.execute("ANALYZE feedback")
            t = time.perf_counter()
            conn.execute("DELETE FROM resolution_requests WHERE request_id = ANY(%s::uuid[])", (ids[300:600],))
            with_ix = time.perf_counter() - t
            res["foreign_key_index"] = {"delete_300_requests_without_index_s": round(without, 3), "delete_300_requests_with_index_s": round(with_ix, 3),
                                        "speedup": round(without / max(with_ix, 1e-6), 1), "feedback_rows": a.feedback}
            lat = explain(conn, "latest rating per case WITH feedback_request_idx", "SELECT (SELECT f.rating FROM feedback f WHERE f.request_id = r.request_id ORDER BY f.created_at DESC LIMIT 1) FROM (SELECT request_id FROM resolution_requests ORDER BY created_at DESC LIMIT 25) r")
            res["foreign_key_index"]["latest_rating_query_ms_with_index"] = lat["execution_ms"]
            conn.execute("DROP INDEX feedback_request_idx")
            lat2 = explain(conn, "latest rating per case WITHOUT feedback_request_idx", "SELECT (SELECT f.rating FROM feedback f WHERE f.request_id = r.request_id ORDER BY f.created_at DESC LIMIT 1) FROM (SELECT request_id FROM resolution_requests ORDER BY created_at DESC LIMIT 25) r")
            res["foreign_key_index"]["latest_rating_query_ms_without_index"] = lat2["execution_ms"]
    finally:
        with psycopg.connect(root, autocommit=True) as c:
            c.execute(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    for p in res["plans"]:
        print(f"{p['execution_ms']:>8.2f} ms  {p['query']:<52} {' > '.join(p['plan'][:3])}")
    print(json.dumps(res["foreign_key_index"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
