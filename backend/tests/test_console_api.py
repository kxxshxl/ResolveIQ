"""Cases and replay, the retrieval lab, recurring clusters, feedback intelligence and system health, through the real API and Postgres."""
import json
import uuid

import pytest

from app.evaluation.drift_traffic import clear_traffic, incident, MARKER, _insert as insert_request
from app.quality.analysis import analyze, findings, to_markdown, wilson_lower

BROADBAND = "My broadband keeps dropping around 8 PM each day and I've restarted the router twice already."


def _fresh(text):
    """A complaint nobody else asked: same meaning, a unique tail, so the response cache cannot answer it."""
    import random
    import string

    return text + " Sent at " + "".join(random.choices(string.ascii_lowercase, k=8)) + "."


def _resolve(client, text=BROADBAND, fresh=True, **kw):
    r = client.post("/api/v1/resolve", json={"complaint": _fresh(text) if fresh else text, **kw})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ cases
def test_every_resolution_is_a_listed_and_inspectable_case(client):
    r = _resolve(client)
    rid = r["request_id"]
    page = client.get("/api/v1/cases", params={"limit": 5}).json()
    assert page["total"] >= 1 and page["items"][0]["request_id"] == rid and page["items"][0]["has_trace"] is True
    case = client.get(f"/api/v1/cases/{rid}").json()
    assert case["trace_available"] and [s["name"] for s in case["trace"]][:2] == ["preprocess", "cache"] and case["provenance"]["prompt_version"]
    assert case["lineage"]["checks"] and all(case["lineage"]["checks"].values())
    assert case["sources"] and all(s["in_corpus"] and s["excerpt"] for s in case["sources"])
    assert case["complaint"].startswith("My broadband keeps dropping") and case["feedback"] == []
    assert client.get(f"/api/v1/cases/{uuid.uuid4()}").status_code == 404 and client.get("/api/v1/cases/not-a-uuid").status_code == 404


def test_case_listing_filters_and_pagination(client):
    _resolve(client, "What is the best pizza place near the city centre for a team dinner tonight?")
    ab = client.get("/api/v1/cases", params={"status": "abstained"}).json()
    assert ab["items"] and all(i["status"] == "abstained" for i in ab["items"])
    assert client.get("/api/v1/cases", params={"status": "bogus"}).status_code == 422
    assert client.get("/api/v1/cases", params={"limit": 101}).status_code == 422
    first = client.get("/api/v1/cases", params={"limit": 1, "offset": 0}).json()["items"][0]["request_id"]
    second = client.get("/api/v1/cases", params={"limit": 1, "offset": 1}).json()["items"][0]["request_id"]
    assert first != second


def test_deleted_sources_are_reported_not_silently_dropped(client, svc, run):
    from app.models.schemas import TicketIn

    tid = f"TKT-REPLAY-{uuid.uuid4().hex[:6]}"
    run(svc.ingestion.ingest_ticket, TicketIn(ticket_id=tid, complaint_text="Streaming box shows a black screen with a spinning wheel after the nightly update, replay probe.",
                                               intent="device_issue", product="tv", severity="medium", sentiment="neutral", resolution_steps=["Power cycle the box for 30 seconds."],
                                               resolution_summary="Power cycle cleared the black screen"))
    r = _resolve(client, "Streaming box shows a black screen with a spinning wheel after the nightly update, replay probe.", strategy="dense")
    assert tid in {t["source_id"] for t in r["tickets"]}
    run(svc.repo.delete_by_ids, "ticket", [tid])
    case = client.get(f"/api/v1/cases/{r['request_id']}").json()
    gone = next(s for s in case["sources"] if s["source_id"] == tid)
    assert gone["in_corpus"] is False and gone["excerpt"] is None


# ------------------------------------------------------------------ replay
def test_replaying_an_unchanged_case_reproduces_it_and_persists_nothing(client, svc, run):
    rid = _resolve(client, BROADBAND, deterministic=True)["request_id"]
    before = run(svc.repo.fetch_one, "SELECT count(*) AS n FROM resolution_requests")["n"]
    out = client.post(f"/api/v1/cases/{rid}/replay", json={}).json()
    assert out["persisted"] is False and out["diff"]["reproduced"] is True and out["diff"]["provenance_changes"] == {} and out["diff"]["likely_reasons"] == []
    assert out["replay"]["request_id"] != rid and out["options"]["deterministic"] is True
    assert run(svc.repo.fetch_one, "SELECT count(*) AS n FROM resolution_requests")["n"] == before


def test_replay_explains_what_changed_and_why(client, svc, run):
    from app.models.schemas import TicketIn

    rid = _resolve(client, "Calls drop after a few minutes on the mobile network whenever I am indoors at home.", strategy="dense")["request_id"]
    run(svc.ingestion.ingest_ticket, TicketIn(ticket_id=f"TKT-NEW-{uuid.uuid4().hex[:6]}", complaint_text="Calls drop after a few minutes on the mobile network whenever I am indoors at home.",
                                               intent="sim_mobile_connectivity", product="mobile", severity="medium", sentiment="neutral",
                                               resolution_steps=["Enable WiFi calling in the phone settings."], resolution_summary="WiFi calling enabled for indoor coverage"))
    out = client.post(f"/api/v1/cases/{rid}/replay", json={"strategy": "hybrid"}).json()
    d = out["diff"]
    assert d["provenance_changes"]["corpus_version"]["new"] > d["provenance_changes"]["corpus_version"]["old"]
    assert d["provenance_changes"]["retrieval_strategy"] == {"old": "dense", "new": "hybrid"} and d["retrieval"]["added"]
    assert any("corpus changed" in w for w in d["likely_reasons"]) and any("retrieval strategy" in w for w in d["likely_reasons"]) and d["reproduced"] is False


def test_replay_without_the_llm_returns_the_evidence_only_path(client):
    rid = _resolve(client, BROADBAND, deterministic=True)["request_id"]
    out = client.post(f"/api/v1/cases/{rid}/replay", json={"generate": False}).json()
    assert out["replay"]["status"] == "degraded" and out["replay"]["generator"] == "extractive"
    assert out["replay"]["provenance"]["generation"].get("prompt_version") is None
    assert client.post(f"/api/v1/cases/{uuid.uuid4()}/replay", json={}).status_code == 404
    assert client.post(f"/api/v1/cases/{rid}/replay", json={"strategy": "nope"}).status_code == 422


def test_diff_pure_function_handles_a_case_without_a_trace(client):
    from app.cases.diff import diff_cases

    new = _resolve(client, BROADBAND)
    old = {"provenance": None, "retrieved": [], "result": {}, "classification": {}, "status": "resolved"}
    d = diff_cases(old, new)
    assert d["reproduced"] is False and d["retrieval"]["added"] and d["evidence"]["delta"] is None


# ------------------------------------------------------------------ retrieval lab
def test_retrieval_lab_shows_every_strategy_with_relevance_for_a_labelled_query(client):
    ex = client.get("/api/v1/retrieval/examples", params={"limit": 20}).json()["items"]
    assert ex and {"qid", "split", "text", "scenario_id"} <= set(ex[0])
    pick = next(e for e in ex if e["split"] == "gold")
    out = client.post("/api/v1/retrieval/compare", json={"complaint": pick["text"], "k": 5}).json()
    assert out["ground_truth"]["available"] and out["ground_truth"]["scenario_id"] == pick["scenario_id"] and out["embedding_ms"] > 0
    assert [s["name"] for s in out["strategies"]] == ["lexical", "dense", "hybrid", "hybrid_reranked", "adaptive"]
    dense = next(s for s in out["strategies"] if s["name"] == "dense")
    assert len(dense["kinds"]["ticket"]["results"]) == 5 and all(isinstance(r["relevant"], bool) for r in dense["kinds"]["ticket"]["results"])
    assert dense["kinds"]["ticket"]["summary"]["mrr"] > 0 and dense["latency_ms"] > 0
    assert out["rank_movement"]
    assert next(s for s in out["strategies"] if s["name"] == "adaptive")["kinds"]["ticket"]["adaptive"][0]["stage"] == "dense"   # the lab shows the adaptive decisions


def test_lab_never_guesses_relevance_for_free_text_and_redacts_the_complaint(client):
    out = client.post("/api/v1/retrieval/compare", json={"complaint": "Please call me on +44 7700 900123 or mail jo@example.com: the router keeps rebooting.", "strategies": ["dense", "lexical"]}).json()
    assert out["ground_truth"]["available"] is False and "not guessed" in out["ground_truth"]["note"]
    assert "7700" not in out["complaint"] and "jo@example.com" not in out["complaint"] and "[EMAIL]" in out["complaint"]
    assert all(r["relevant"] is None for s in out["strategies"] for k in s["kinds"].values() for r in k["results"])
    assert client.post("/api/v1/retrieval/compare", json={"complaint": "router keeps rebooting every hour", "strategies": ["bogus"]}).status_code == 422


def test_a_supplied_scenario_marks_relevance_and_a_case_can_be_compared(client):
    pick = client.get("/api/v1/retrieval/examples").json()["items"][0]
    out = client.post("/api/v1/retrieval/compare", json={"complaint": "my internet is unusable lately and nothing helps, " + pick["text"], "scenario_id": pick["scenario_id"], "strategies": ["dense"]}).json()
    assert out["ground_truth"]["source"] == "supplied" and any(r["relevant"] for r in out["strategies"][0]["kinds"]["ticket"]["results"])
    rid = _resolve(client)["request_id"]
    cmp = client.post(f"/api/v1/cases/{rid}/compare", json={"strategies": ["dense", "adaptive"]}).json()
    assert [s["name"] for s in cmp["strategies"]] == ["dense", "adaptive"]


# ------------------------------------------------------------------ feedback
def test_feedback_keeps_context_and_redacts_free_text(client, svc, run):
    r = _resolve(client, "Cannot log into the customer portal, the password reset mail never arrives, tried three times today.")
    rid, src = r["request_id"], r["tickets"][0]["source_id"]
    body = {"request_id": rid, "rating": "not_helpful", "reasons": ["steps_incorrect", "irrelevant_source"], "rejected_sources": [src], "corrected_intent": "account_authentication",
            "comment": "customer jo.bloggs@example.com says it still fails, call 07700 900123", "edited_steps": ["Reset the password for jo.bloggs@example.com by phone.", "Escalate to identity team."]}
    assert client.post("/api/v1/feedback", json=body).status_code == 201
    row = run(svc.repo._fetch, "SELECT comment, edited_steps, reasons, rejected_sources, provenance->>'prompt_version' AS pv FROM feedback WHERE request_id=%s", (rid,))[0]
    stored = json.dumps([row["comment"], row["edited_steps"]])
    assert "example.com" not in stored and "07700" not in stored and "[EMAIL]" in stored
    assert row["reasons"] == ["steps_incorrect", "irrelevant_source"] and row["rejected_sources"] == [src] and row["pv"]
    case = client.get(f"/api/v1/cases/{rid}").json()
    assert case["feedback"][0]["rating"] == "not_helpful" and client.get("/api/v1/cases", params={"rated": "not_helpful"}).json()["total"] >= 1


def test_feedback_rejects_sources_that_were_never_retrieved_and_bad_reasons(client):
    rid = _resolve(client)["request_id"]
    r = client.post("/api/v1/feedback", json={"request_id": rid, "rating": "not_helpful", "rejected_sources": ["TKT-DOES-NOT-EXIST"]})
    assert r.status_code == 422 and "not retrieved" in r.json()["error"]["message"]
    assert client.post("/api/v1/feedback", json={"request_id": rid, "rating": "helpful", "reasons": ["because"]}).status_code == 422
    assert client.post("/api/v1/feedback", json={"request_id": rid, "rating": "helpful", "edited_steps": ["x"] * 13}).status_code == 422


# ------------------------------------------------------------------ feedback analytics (pure) and the report
def _fb(i, intent, rating, cites=("KB-001",), **kw):
    return {"feedback_id": i, "rating": rating, "reasons": kw.get("reasons", []), "rejected_sources": kw.get("rejected", []), "corrected_intent": kw.get("corrected"), "edited": kw.get("edited", False),
            "request_id": f"r{i}", "status": "resolved", "intent": intent, "product": "broadband", "citations": [{"source_id": c} for c in cites]}


def _req(status="resolved", intent="billing_dispute", **kw):
    return {"status": status, "intent": intent, "product": "account", "abstention_reason": kw.get("reason"), "validation": kw.get("validation", {}), "fallback_reason": kw.get("fallback"),
            "evidence": kw.get("evidence", 0.8), "escalated": kw.get("escalated", False)}


def test_wilson_lower_bound_ranks_small_samples_below_large_ones():
    assert wilson_lower(3, 3) < wilson_lower(30, 40) and wilson_lower(0, 0) == 0.0 and wilson_lower(5, 10) < 0.5


def test_analysis_finds_problem_intents_weak_articles_and_confusions():
    fb = ([_fb(i, "billing_dispute", "not_helpful", cites=("KB-009", "TKT-1"), rejected=["KB-009"]) for i in range(6)] + [_fb(10 + i, "billing_dispute", "helpful", cites=("TKT-2",)) for i in range(2)]
          + [_fb(20 + i, "slow_speed", "helpful") for i in range(8)] + [_fb(40, "slow_speed", "not_helpful", corrected="plan_change"), _fb(41, "slow_speed", "not_helpful", corrected="plan_change")])
    reqs = [_req(intent="billing_dispute")] * 20 + [_req(intent="slow_speed")] * 20 + [_req("abstained", "service_outage", reason="weak_evidence: x")] * 6 + [_req("degraded", fallback="LLMUnavailable")] * 4
    s = analyze(fb, reqs, 30)
    assert s["totals"]["helpful"] == 10 and s["totals"]["not_helpful"] == 8 and s["totals"]["rejection_rate"] == pytest.approx(8 / 18, abs=1e-3)
    assert s["problem_intents"][0]["intent"] == "billing_dispute" and s["weak_articles"][0]["source_id"] == "KB-009" and s["weak_articles"][0]["explicit_rejections"] == 6
    assert s["intent_confusions"] == [{"predicted": "slow_speed", "corrected": "plan_change", "count": 2}]
    assert s["abstention"]["by_reason"] == {"weak_evidence": 6} and s["abstention"]["by_intent"] == {"service_outage": 6}
    kinds = {f["kind"] for f in findings(s)}
    assert {"problem_intent", "weak_article", "intent_confusion", "abstention_cluster"} <= kinds
    assert findings(s) == findings(s) and to_markdown(s, findings(s)) == to_markdown(s, findings(s))        # deterministic
    assert "Advisory only" in to_markdown(s, findings(s))


def test_findings_need_a_minimum_sample_and_say_when_data_is_thin():
    s = analyze([_fb(1, "billing_dispute", "not_helpful"), _fb(2, "billing_dispute", "not_helpful")], [_req()] * 50, 30)
    f = findings(s)
    assert not any(x["kind"] in ("problem_intent", "weak_article") for x in f) and f[0]["kind"] == "data_sufficiency"
    assert "Only 2 rating" in f[0]["title"]


def test_quality_endpoints_serve_the_summary_and_a_deterministic_report(client):
    s = client.get("/api/v1/quality/summary", params={"days": 30}).json()
    assert {"totals", "by_intent", "problem_intents", "rejected_sources", "abstention", "failure_patterns"} <= set(s)
    a, b = client.get("/api/v1/quality/report", params={"days": 30}).json(), client.get("/api/v1/quality/report", params={"days": 30}).json()
    assert a["markdown"] == b["markdown"] and a["advisory_only"] is True and "changes no model" in a["note"]
    assert client.get("/api/v1/quality/summary", params={"days": 0}).status_code == 422


# ------------------------------------------------------------------ recurring complaint clusters
@pytest.fixture()
def recurring_traffic(svc, run):
    async def seed():
        await clear_traffic(svc.repo)
        for i, text in enumerate(incident(12)):
            await insert_request(svc.repo, text, {"intent": "device_issue", "product": "wifi_router", "severity": "medium", "sentiment": "frustrated"}, "abstained", 0.3, 1 + i)
        await svc.repo._exec("DELETE FROM taxonomy_proposals")
    run(seed)
    svc.clusters._cache.clear()
    yield
    run(clear_traffic, svc.repo)
    run(svc.repo._exec, "DELETE FROM taxonomy_proposals")
    svc.clusters._cache.clear()


def test_recurring_clusters_find_the_group_and_describe_it(client, svc, run, recurring_traffic):
    out = client.get("/api/v1/clusters/recurring", params={"days": 7, "min_size": 4}).json()
    c = next(c for c in out["clusters"] if "smart home hub" in c["representative"]["complaint"])
    assert c["size"] == 12 and c["kind"] == "recurring_complaint_cluster" and "Not a confirmed network incident" in c["disclaimer"] and "Not a confirmed network incident" in out["disclaimer"]
    assert c["intent"] == "device_issue" and c["intent_share"] == 1.0 and c["product"] == "wifi_router" and 0 < c["confidence"] <= 1
    assert c["time_pattern"]["label"].startswith("new") and sum(c["time_pattern"]["daily_counts"]) == 12 and len(c["time_pattern"]["daily_counts"]) == 7
    assert len(c["examples"]) == 3 and c["abstained_share"] == 1.0 and c["discovery"] == "no proposal yet" and c["proposals"] == [] and "member_request_ids" not in c
    # the embedding was computed once and stored: a second computation reads it back
    assert run(svc.repo.fetch_one, "SELECT count(*) AS n FROM resolution_requests WHERE trace_id=%s AND embedding IS NOT NULL", (MARKER,))["n"] == 12
    assert client.get("/api/v1/clusters/recurring", params={"days": 7, "min_size": 4}).json()["cached"] is True


def test_recurring_clusters_link_to_discovery_proposals(client, svc, run, recurring_traffic):
    run(svc.discovery.run)
    out = client.get("/api/v1/clusters/recurring", params={"days": 7, "min_size": 4}).json()
    c = next(c for c in out["clusters"] if "smart home hub" in c["representative"]["complaint"])
    assert c["proposals"] and c["proposals"][0]["status"] == "pending" and c["proposals"][0]["overlap_share"] >= 0.5 and c["discovery"] == "proposal pending"
    assert client.get("/api/v1/clusters/recurring", params={"days": 0}).status_code == 422 and client.get("/api/v1/clusters/recurring", params={"distance": 5}).status_code == 422


def test_cluster_engine_is_deterministic_and_ignores_scatter():
    import numpy as np
    from datetime import datetime, timezone

    from app.clusters.engine import cluster_requests

    rng = np.random.default_rng(0)
    centre = rng.normal(size=16); centre /= np.linalg.norm(centre)
    tight = [centre + rng.normal(0, 0.03, 16) for _ in range(5)]
    scatter = [rng.normal(size=16) for _ in range(6)]
    X = np.array([v / np.linalg.norm(v) for v in tight + scatter], dtype=np.float32)
    now = datetime.now(timezone.utc)
    rows = [{"request_id": f"r{i}", "complaint": f"complaint {i} about topic zeta", "created_at": now, "intent": "device_issue" if i < 5 else "other", "product": "wifi_router",
             "severity": "medium", "status": "resolved", "evidence": 0.7, "occurrences": 1} for i in range(11)]
    a, b = cluster_requests(rows, X, min_size=3, distance=0.3, now=now), cluster_requests(rows, X, min_size=3, distance=0.3, now=now)
    assert [c["cluster_id"] for c in a] == [c["cluster_id"] for c in b] and len(a) == 1 and a[0]["size"] == 5 and a[0]["intent_share"] == 1.0


# ------------------------------------------------------------------ system
def test_system_status_reports_versions_dependencies_llm_and_queue(client, svc):
    s = client.get("/api/v1/system/status").json()
    assert s["status"] == "ready" and s["checks"]["database"] == "ok" and s["versions"]["prompt"]["version"] and s["versions"]["embedding_model"] == svc.embedder.model_name
    assert s["llm"]["providers"][0]["circuit"] == "closed" and s["llm"]["providers"][0]["slots"]["limit"] == svc.settings.llm_max_concurrency
    assert s["queue"]["mode"] == svc.settings.job_execution and s["database"]["status"] in ("ok", "warn") and s["security"]["authentication"] is False


def test_database_health_is_informative_and_exposes_no_row_contents(client, svc):
    d = client.get("/api/v1/system/db").json()
    assert d["status"] in ("ok", "warn") and d["server"]["pgvector"] and d["vector_search"]["hnsw_ef_search"] == svc.settings.hnsw_ef_search
    assert {i["table"] for i in d["vector_search"]["indexes"]} == {"ticket_embeddings", "kb_embeddings"} and all(i["m"] and i["ef_construction"] for i in d["vector_search"]["indexes"])
    assert d["embedding_coverage"]["tickets_missing"] == 0 and d["embedding_coverage"]["articles_missing"] == 0 and d["integrity"]["orphan_ticket_embeddings"] == 0
    names = {t["name"] for t in d["tables"]}
    assert {"tickets", "knowledge_articles", "resolution_requests", "feedback"} <= names and all(t["total_bytes"] > 0 for t in d["tables"])
    assert any(i["name"] == "feedback_request_idx" for i in d["indexes"]) and d["pool"]["max"] == svc.settings.db_pool_max
    text = json.dumps(d)
    assert "broadband keeps dropping" not in text and "postgresql://" not in text
