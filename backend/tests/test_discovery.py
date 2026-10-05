"""Emerging-class discovery: pure clustering logic, then the full propose -> review -> accept/reject flow against Postgres."""
import json
import uuid
from collections import Counter

import numpy as np
import pytest

from app.core.config import REPO_ROOT
from app.discovery.clustering import DiscoveryParams, Neighbors, build_proposals, cluster_embeddings, keywords_per_cluster


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _blob(center, n, noise, rng):
    return [_unit(center + rng.normal(0, noise, len(center))) for _ in range(n)]


# ---------------------------------------------------------------- pure logic
def test_clusters_recurring_topics_and_drops_scattered_noise():
    rng = np.random.default_rng(0)
    d = 16
    a, b = _unit(rng.normal(size=d)), _unit(rng.normal(size=d))
    X = np.array(_blob(a, 6, 0.05, rng) + _blob(b, 5, 0.05, rng) + [_unit(rng.normal(size=d)) for _ in range(5)])
    groups = cluster_embeddings(X, DiscoveryParams(distance_threshold=0.3, min_cluster_size=4))
    assert sorted(len(g) for g in groups) == [5, 6]
    assert set(groups[0]) == set(range(6))


def test_too_few_points_yield_no_clusters():
    assert cluster_embeddings(np.eye(3), DiscoveryParams(min_cluster_size=4)) == []


def test_keywords_prefer_cluster_specific_terms_over_corpus_wide_ones():
    clusters = [["my esim qr code will not activate", "esim activation code fails", "scan the esim qr code, error"],
                ["voicemail pin reset not working", "cannot retrieve voicemail messages"]]
    background = ["my router internet is slow", "internet drops every evening", "router light is red"] * 5
    kws = keywords_per_cluster(clusters, background, 4)
    assert "esim" in " ".join(kws[0]) and "voicemail" in " ".join(kws[1])
    assert not any(k in kws[0] for k in ("internet", "router"))


def _proposals(sim, agreement_intents):
    rng = np.random.default_rng(1)
    c = _unit(rng.normal(size=16))
    X = np.array(_blob(c, 5, 0.03, rng))
    texts = [f"complaint about topic zeta number {i}" for i in range(5)]
    nb = [Neighbors(agreement_intents, ["mobile"] * len(agreement_intents), sim) for _ in range(5)]
    return build_proposals(texts, X, nb, [0.5] * 5, ["background text"], DiscoveryParams(), {"billing_dispute": "billing team", "slow_speed": "net"})


def test_far_from_existing_tickets_is_a_new_class():
    p = _proposals(0.45, ["billing_dispute", "slow_speed"])
    assert len(p) == 1 and p[0]["recommendation"] == "new_class" and p[0]["size"] == 5


def test_close_to_one_existing_class_recommends_extending_it():
    p = _proposals(0.72, ["billing_dispute", "billing_dispute"])
    assert p[0]["recommendation"] == "extend_existing" and p[0]["label_id"] == "billing_dispute" and p[0]["team"] == "billing team"


def test_close_to_existing_tickets_without_a_single_owner_is_dropped_as_noise():
    rng = np.random.default_rng(1)
    c = _unit(rng.normal(size=16))
    X = np.array(_blob(c, 5, 0.03, rng))
    mixed = [Neighbors([i], ["mobile"], 0.7) for i in ("billing_dispute", "slow_speed", "device_issue", "plan_change", "service_outage")]
    assert build_proposals(["t"] * 5, X, mixed, [0.5] * 5, ["bg"], DiscoveryParams(), {}) == []


def test_proposed_label_id_never_collides_with_an_existing_label():
    p = _proposals(0.45, ["slow_speed"])
    base = p[0]["label_id"]
    again = build_proposals([f"complaint about topic zeta number {i}" for i in range(5)],
                            np.array(_blob(_unit(np.random.default_rng(1).normal(size=16)), 5, 0.03, np.random.default_rng(1))),
                            [Neighbors(["slow_speed"], ["mobile"], 0.45)] * 5, [0.5] * 5, ["bg"], DiscoveryParams(), {base: None})
    assert again[0]["label_id"] != base


# ---------------------------------------------------------------- end to end (Postgres + real embeddings)
def _novel(classes):
    rows = [json.loads(l) for l in (REPO_ROOT / "data" / "eval" / "novel_stream.jsonl").read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r["intent"] in classes]


@pytest.fixture()
def abstained_requests(svc, run):
    ids = []

    async def insert(rows):
        for r in rows:
            rid = str(uuid.uuid4())
            ids.append(rid)
            await svc.repo.save_request(rid, "t", r["text"], {}, [], {"evidence": {"confidence": 0.3}}, "abstained", 0.3, 5)
        return {rid: r for rid, r in zip(ids, rows)}

    yield lambda rows: run(insert, rows)
    run(svc.repo._exec, "DELETE FROM resolution_requests WHERE request_id = ANY(%s::uuid[])", (ids,))
    run(svc.repo._exec, "DELETE FROM taxonomy_proposals")
    run(svc.repo._exec, "DELETE FROM taxonomy_labels WHERE introduced_in > 3")
    run(svc.taxonomy.refresh)


def test_discovery_finds_two_novel_classes_and_review_flow_works(client, svc, run, abstained_requests):
    owner = abstained_requests(_novel({"voicemail_issue", "tv_streaming_app"}))
    result = run(svc.discovery.run)
    assert result["proposals"] >= 2

    items = client.get("/api/v1/taxonomy/proposals").json()["items"]
    assert items and all(p["status"] == "pending" for p in items)
    top = [p for p in items if p["recommendation"] == "new_class"]
    assert len(top) >= 2
    kw = " ".join(" ".join(p["keywords"]) for p in top)
    assert "voicemail" in kw or "app" in kw

    # each proposal must be dominated by one true class (clusters follow topics, not noise)
    for p in top:
        prop = run(svc.repo.get_proposal, p["proposal_id"])
        majority = Counter(owner[m]["intent"] for m in prop["member_request_ids"] if m in owner).most_common(1)
        assert majority and majority[0][1] / len(prop["member_request_ids"]) >= 0.8

    v0 = client.get("/api/v1/taxonomy").json()["version"]
    chosen = top[0]
    r = client.post(f"/api/v1/taxonomy/proposals/{chosen['proposal_id']}/accept", json={"label_id": "zz_emerging_a"})
    assert r.status_code == 200 and r.json()["action"] == "created"
    assert client.get("/api/v1/taxonomy").json()["version"] > v0
    assert svc.taxonomy.current.has("intent", "zz_emerging_a")
    assert client.post(f"/api/v1/taxonomy/proposals/{chosen['proposal_id']}/accept", json={}).status_code == 409

    other = top[1]
    assert client.post(f"/api/v1/taxonomy/proposals/{other['proposal_id']}/reject", json={"note": "duplicate of existing process"}).status_code == 200
    # rejected requests must not be proposed again on the next run
    run(svc.discovery.run)
    pending = client.get("/api/v1/taxonomy/proposals").json()["items"]
    rejected_members = set(run(svc.repo.get_proposal, other["proposal_id"])["member_request_ids"])
    for p in pending:
        assert not rejected_members & set(run(svc.repo.get_proposal, p["proposal_id"])["member_request_ids"])


def test_accepting_with_existing_label_id_conflicts_and_unknown_proposal_404(client):
    assert client.post(f"/api/v1/taxonomy/proposals/{uuid.uuid4()}/accept", json={}).status_code == 404
    assert client.post("/api/v1/taxonomy/proposals/not-a-uuid/reject", json={}).status_code == 404
