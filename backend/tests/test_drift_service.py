"""Drift monitoring end to end against Postgres: logged requests in two windows -> worker job -> stored analysis -> API -> link to discovery proposals.

The traffic (app.evaluation.drift_traffic) is the project's labelled complaints placed in time (baseline: days 2-12 ago, recent: the last day), plus one
synthetic incident: 25 distinct complaints about a smart-home hub that the corpus has never seen, all in the recent window.
"""
import pytest

from app.evaluation.drift_traffic import seed_traffic


@pytest.fixture(scope="module")
def traffic(svc, run):
    """Isolate the table: other test modules log requests too, and those would land in the recent window."""
    async def seed():
        for t in ("taxonomy_proposals", "drift_snapshots", "resolution_requests"):
            await svc.repo._exec(f"DELETE FROM {t}")
        await svc.repo._exec("DELETE FROM jobs WHERE kind IN ('drift_analysis','discover_classes')")
        return await seed_traffic(svc.repo)

    ids = run(seed)
    yield ids

    async def clean():
        for t in ("taxonomy_proposals", "drift_snapshots", "resolution_requests"):
            await svc.repo._exec(f"DELETE FROM {t}")
        await svc.repo._exec("DELETE FROM jobs WHERE kind IN ('drift_analysis','discover_classes')")
        await svc.taxonomy.refresh()

    run(clean)


def test_analysis_finds_the_incident_and_stores_an_explainable_snapshot(svc, run, traffic):
    summary = run(svc.drift.run, 24, 14)
    assert summary["status"] == "alert" and summary["emerging_clusters"] >= 1 and summary["n_recent"] == 95 and summary["n_baseline"] == 170

    rep = run(svc.drift.latest)
    assert rep["snapshot_id"] == summary["snapshot_id"] and rep["thresholds"]["alpha"] == 0.01
    assert rep["tests"]["method"] == "Holm-Bonferroni"
    cluster = max((c for c in rep["emerging_clusters"] if c["significant"]), key=lambda c: c["size"])
    assert cluster["size"] >= 15 and cluster["kind"] == "new_topic" and not cluster["covered_by_corpus"]
    assert "hub" in " ".join(cluster["keywords"]) or "smart" in " ".join(cluster["keywords"])
    assert any("smart home hub" in e for e in cluster["examples"])
    assert "request_ids" not in cluster                      # served without the member id list
    signals = {a["signal"] for a in rep["alerts"]}
    assert "emerging_cluster" in signals and "abstention_rate" in signals   # 25 of 95 recent requests abstained vs none in the baseline
    for dist in ("intent", "product", "severity", "sentiment"):
        assert rep["distributions"][dist]["categories"], dist


def test_without_a_worker_the_discovery_decision_is_a_recommendation_not_an_action(svc, run, traffic):
    rep = run(svc.drift.latest)
    d = rep["discovery"]
    assert d["recommended"] is True and d["triggered"] is False and "inline" in d["reason"]
    assert run(svc.repo._fetch, "SELECT 1 FROM jobs WHERE kind='discover_classes'") == []


def test_cluster_links_to_the_proposals_discovery_builds_from_the_same_requests(client, svc, run, traffic):
    assert client.get("/api/v1/monitoring/drift/status").json()["emerging_clusters"][0]["related_proposals"] == []
    result = run(svc.discovery.run)
    assert result["proposals"] >= 1
    rep = client.get("/api/v1/monitoring/drift/status").json()
    cluster = max((c for c in rep["emerging_clusters"] if c["significant"]), key=lambda c: c["size"])
    related = cluster["related_proposals"]
    assert related and related[0]["status"] == "pending" and related[0]["overlap_share"] >= 0.5
    # rejecting the proposal is reflected in the drift view, which reads review state live
    pid = related[0]["proposal_id"]
    assert client.post(f"/api/v1/taxonomy/proposals/{pid}/reject", json={"note": "test"}).status_code == 200
    again = client.get("/api/v1/monitoring/drift/status").json()
    assert next(p for c in again["emerging_clusters"] for p in c["related_proposals"] if p["proposal_id"] == pid)["status"] == "rejected"


def test_a_reviewed_cluster_no_longer_asks_for_discovery(svc, run, traffic):
    summary = run(svc.drift.run, 24, 14)
    assert summary["discovery"]["recommended"] is False   # the rejected proposal covers it: do not propose again


def test_queue_mode_enqueues_one_discovery_job_for_an_uncovered_cluster(svc, run, traffic, test_settings):
    from app.drift.service import DriftService

    async def scenario():
        await svc.repo._exec("DELETE FROM taxonomy_proposals")
        queue_mode = DriftService(svc.repo, svc.embedder, test_settings.model_copy(update={"job_execution": "queue"}))
        first = await queue_mode.run(24, 14)
        second = await queue_mode.run(24, 14)
        jobs = await svc.repo._fetch("SELECT payload FROM jobs WHERE kind='discover_classes'")
        return first, second, jobs

    first, second, jobs = run(scenario)
    assert first["discovery"]["triggered"] is True and first["discovery"]["job_id"]
    assert second["discovery"]["triggered"] is False and "already" in second["discovery"]["reason"]
    assert len(jobs) == 1 and jobs[0]["payload"]["triggered_by"] == "drift"


def test_api_surface_history_timeline_run_and_status(client, svc, run, traffic):
    hist = client.get("/api/v1/monitoring/drift/history?limit=5").json()["items"]
    assert hist and hist[0]["status"] in ("alert", "ok", "insufficient_data") and hist[0]["emerging_clusters"] >= 1
    tl = client.get("/api/v1/monitoring/drift/timeline?days=14&bucket=day").json()
    assert tl["bucket"] == "day" and sum(p["n"] for p in tl["points"]) == 170 + 95
    assert any("device_issue" in p["intents"] for p in tl["points"])
    assert client.get("/api/v1/monitoring/drift/timeline?bucket=week").status_code == 422
    assert client.get("/api/v1/monitoring/drift/timeline?days=0").status_code == 422

    r = client.post("/api/v1/monitoring/drift/run", json={"window_hours": 24, "baseline_days": 14})
    assert r.status_code == 202
    job = client.get(f"/api/v1/jobs/{r.json()['job_id']}").json()   # inline mode: the job ran after the response
    assert job["status"] == "succeeded" and job["result"]["status"] == "alert" and "snapshot_id" in job["result"]
    assert client.post("/api/v1/monitoring/drift/run", json={"window_hours": 0}).status_code == 422
    assert client.get("/api/v1/monitoring/drift").status_code == 200       # the original label-free endpoint still works


def test_status_before_any_analysis_says_so(client, svc, run, traffic):
    run(svc.repo._exec, "DELETE FROM drift_snapshots")
    body = client.get("/api/v1/monitoring/drift/status").json()
    assert body["status"] == "no_analysis" and "run" in body["note"]
