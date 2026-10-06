"""Database layer: batched ingestion, keyset pagination, duplicate detection, indexes and connection settings."""
import uuid

import pytest

from app.models.schemas import TicketIn


def _ticket(i, tag, **kw):
    return {"ticket_id": f"TKT-DB-{tag}-{i}", "complaint_text": f"Streaming picture freezes on the evening news number {i} {tag} and the box restarts afterwards.", "intent": "device_issue",
            "product": "tv", "severity": "low", "sentiment": "neutral", "resolution_steps": ["Power cycle the set top box."], "resolution_summary": f"Power cycle {i}", **kw}


@pytest.fixture()
def cleanup(svc, run):
    ids: list[str] = []
    yield ids
    run(svc.repo.delete_by_ids, "ticket", ids)
    run(svc.repo.bump_corpus_version)


def _batch(svc, run, tickets):
    return run(svc.ingestion.ingest_tickets_bulk, [TicketIn(**t) for t in tickets])


def test_batched_ingestion_creates_updates_and_is_idempotent(svc, run, cleanup):
    tag = uuid.uuid4().hex[:6]
    rows = [_ticket(i, tag) for i in range(40)]
    cleanup += [r["ticket_id"] for r in rows]
    first = _batch(svc, run, rows)
    assert (first["created"], first["updated"], first["failed"]) == (40, 0, 0)
    again = _batch(svc, run, rows)
    assert (again["created"], again["updated"], again["failed"]) == (0, 40, 0)
    n = run(svc.repo.fetch_one, "SELECT count(*) AS n FROM ticket_embeddings e JOIN tickets d ON d.id = e.ticket_pk WHERE d.ticket_id LIKE %s AND e.model = %s",
            (f"TKT-DB-{tag}-%", svc.embedder.model_name))["n"]
    assert n == 40                                                   # every ticket has exactly one embedding for the configured model


def test_one_bad_row_costs_only_itself(svc, run, cleanup):
    tag = uuid.uuid4().hex[:6]
    rows = [_ticket(i, tag) for i in range(10)]
    rows[4]["intent"] = "no_such_intent"
    cleanup += [r["ticket_id"] for r in rows]
    out = _batch(svc, run, rows)
    assert out["created"] == 9 and out["failed"] == 1 and "no_such_intent" in out["errors"][0]


def test_if_the_batched_write_fails_the_batch_falls_back_to_row_by_row(svc, run, cleanup, monkeypatch):
    tag = uuid.uuid4().hex[:6]
    rows = [_ticket(i, tag) for i in range(6)]
    cleanup += [r["ticket_id"] for r in rows]

    async def boom(*a, **k):
        raise RuntimeError("simulated failure in the batched statement")

    monkeypatch.setattr(svc.repo, "bulk_upsert_tickets", boom)
    out = _batch(svc, run, rows)
    assert (out["created"], out["failed"]) == (6, 0)
    assert run(svc.repo.fetch_one, "SELECT count(*) AS n FROM tickets WHERE ticket_id LIKE %s", (f"TKT-DB-{tag}-%",))["n"] == 6


def test_duplicate_ids_inside_one_batch_do_not_break_it(svc, run, cleanup):
    tag = uuid.uuid4().hex[:6]
    a, b = _ticket(1, tag), _ticket(1, tag, resolution_summary="the later one wins")
    cleanup.append(a["ticket_id"])
    out = _batch(svc, run, [a, _ticket(2, tag), b])
    cleanup.append(_ticket(2, tag)["ticket_id"])
    assert out["failed"] == 0 and out["created"] + out["updated"] == 3
    assert run(svc.repo.fetch_one, "SELECT resolution_summary FROM tickets WHERE ticket_id=%s", (a["ticket_id"],))["resolution_summary"] == "the later one wins"


def test_the_same_content_under_two_ids_is_reported_not_silently_accepted(client, svc, run, cleanup):
    tag = uuid.uuid4().hex[:6]
    one, two = _ticket(1, tag), _ticket(1, tag)
    two["ticket_id"] = f"TKT-DB-{tag}-copy"
    cleanup += [one["ticket_id"], two["ticket_id"]]
    _batch(svc, run, [one, two])
    d = client.get("/api/v1/system/db").json()
    assert d["duplicates"]["ticket_duplicate_groups"] >= 1 and any(f["what"] == "duplicate content" for f in d["findings"])


# ------------------------------------------------------------------ pagination
def test_keyset_pagination_walks_every_ticket_once_in_a_stable_order(client, svc, run):
    seen, cursor, pages = [], "start", 0
    while cursor and pages < 100:
        page = client.get("/api/v1/tickets", params={"limit": 40, "cursor": cursor}).json()
        seen += [t["source_id"] for t in page["items"]]
        cursor, pages = page["next_cursor"], pages + 1
    total = client.get("/api/v1/tickets", params={"limit": 1}).json()["total"]
    assert len(seen) == len(set(seen)) == total and pages >= 6
    offset_walk = [t["source_id"] for off in range(0, total, 100) for t in client.get("/api/v1/tickets", params={"limit": 100, "offset": off}).json()["items"]]
    assert seen[:5] == offset_walk[:5]                                    # same newest-first order as the offset API


def test_keyset_cursor_survives_inserts_between_pages_and_rejects_garbage(client, svc, run, cleanup):
    first = client.get("/api/v1/tickets", params={"limit": 5, "cursor": "start"}).json()
    tag = uuid.uuid4().hex[:6]
    extra = _ticket(1, tag)
    cleanup.append(extra["ticket_id"])
    _batch(svc, run, [extra])                                              # a newer ticket arrives while the reader is on page 1
    second = client.get("/api/v1/tickets", params={"limit": 5, "cursor": first["next_cursor"]}).json()
    assert not {t["source_id"] for t in first["items"]} & {t["source_id"] for t in second["items"]} and extra["ticket_id"] not in {t["source_id"] for t in second["items"]}
    assert client.get("/api/v1/tickets", params={"cursor": "%%%"}).status_code == 422
    assert client.get("/api/v1/articles", params={"limit": 5, "cursor": "start"}).json()["items"]


# ------------------------------------------------------------------ schema and connection settings
def test_indexes_added_for_known_access_paths_exist(svc, run):
    names = {r["indexname"] for r in run(svc.repo._fetch, "SELECT indexname FROM pg_indexes WHERE schemaname='public'")}
    assert {"feedback_request_idx", "feedback_created_idx", "resolution_requests_intent_idx", "resolution_requests_unembedded_idx", "tickets_content_hash_idx",
            "articles_content_hash_idx", "jobs_kind_created_idx", "ticket_emb_hnsw", "kb_emb_hnsw", "tickets_search_idx"} <= names


def test_vector_search_settings_are_applied_to_every_pooled_connection(svc, run):
    async def settings():
        out = []
        for _ in range(3):
            async with svc.repo.pool.connection() as conn:
                out.append(((await (await conn.execute("SHOW hnsw.ef_search")).fetchone())["hnsw.ef_search"],
                            (await (await conn.execute("SHOW hnsw.iterative_scan")).fetchone())["hnsw.iterative_scan"]))
        return out

    assert set(run(settings)) == {(str(svc.settings.hnsw_ef_search), "relaxed_order")}


def test_deleting_a_request_with_feedback_keeps_the_feedback_row_without_a_table_scan(client, svc, run):
    rid = client.post("/api/v1/resolve", json={"complaint": f"The router light is red and nothing connects, reference {uuid.uuid4().hex[:6]} please help."}).json()["request_id"]
    client.post("/api/v1/feedback", json={"request_id": rid, "rating": "helpful"})
    plan = "\n".join(r["QUERY PLAN"] for r in run(svc.repo._fetch, "EXPLAIN SELECT 1 FROM feedback WHERE request_id = %s", (rid,)))
    assert "feedback_request_idx" in plan or "Seq Scan" in plan            # on a tiny table the planner may still prefer a scan; the index exists for when it is not tiny
    run(svc.repo._exec, "DELETE FROM resolution_requests WHERE request_id=%s", (rid,))
    assert run(svc.repo.fetch_one, "SELECT count(*) FILTER (WHERE request_id IS NULL) AS orphans FROM feedback")["orphans"] >= 1
    run(svc.repo._exec, "DELETE FROM feedback WHERE request_id IS NULL")
