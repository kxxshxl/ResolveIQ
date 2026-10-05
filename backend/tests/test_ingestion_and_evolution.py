"""Ingestion pipeline, evolving data (new documents + brand-new classes) and embedding re-index."""
import time
import uuid

import pytest

ESIM_Q = "I scanned the QR code for my eSIM but the phone says the plan cannot be added."


def _ticket(**kw):
    base = {"complaint_text": "Smart doorbell camera loses its link to the base station whenever the microwave runs.",
            "intent": "device_issue", "product": "wifi_router", "severity": "low", "sentiment": "neutral",
            "resolution_steps": ["Move the base station away from the microwave.", "Switch the camera to the 5 GHz band."],
            "resolution_summary": "Interference from microwave; relocated base station."}
    base.update(kw)
    return base


@pytest.fixture()
def cleanup(svc, run):
    created = {"ticket": [], "article": [], "labels": []}
    yield created
    if created["ticket"]:
        run(svc.repo.delete_by_ids, "ticket", created["ticket"])
    if created["article"]:
        run(svc.repo.delete_by_ids, "article", created["article"])
    if created["labels"]:
        run(svc.repo.delete_taxonomy_labels, "intent", created["labels"])
        run(svc.taxonomy.refresh)
    run(svc.repo.bump_corpus_version)


# ---------------------------------------------------------------- single ticket
def test_new_ticket_is_searchable_immediately_without_restart(client, cleanup):
    tid = f"TKT-T{uuid.uuid4().hex[:6]}"
    q = "Doorbell camera drops off the base station each time someone heats food in the microwave"
    before = client.get("/api/v1/search", params={"q": q, "strategy": "dense", "source": "ticket", "limit": 3}).json()["tickets"]
    assert tid not in {t["source_id"] for t in before}
    v0 = client.get("/api/v1/stats").json()["corpus_version"]

    t0 = time.perf_counter()
    r = client.post("/api/v1/ingest/ticket", json=_ticket(ticket_id=tid))
    cleanup["ticket"].append(tid)
    assert r.status_code == 201 and r.json()["created"] is True and r.json()["corpus_version"] > v0
    after = client.get("/api/v1/search", params={"q": q, "strategy": "dense", "source": "ticket", "limit": 3}).json()["tickets"]
    assert after[0]["source_id"] == tid                      # top-1 right away: no rebuild, no restart
    assert time.perf_counter() - t0 < 5
    assert client.get("/api/v1/search", params={"q": q, "strategy": "lexical", "source": "ticket"}).status_code == 200


def test_ingest_redacts_pii_before_storage_and_upserts_by_id(client, cleanup):
    tid = f"TKT-T{uuid.uuid4().hex[:6]}"
    body = _ticket(ticket_id=tid, complaint_text="Doorbell camera keeps offline. Email me at bob@example.com or call 07911 123456.")
    r = client.post("/api/v1/ingest/ticket", json=body)
    cleanup["ticket"].append(tid)
    assert r.json()["pii_redactions"] == {"EMAIL": 1, "PHONE": 1}
    stored = next(t for t in client.get("/api/v1/tickets", params={"limit": 100}).json()["items"] if t["source_id"] == tid)
    assert "bob@example.com" not in stored["text"] and "07911" not in stored["text"] and "[EMAIL]" in stored["text"]
    again = client.post("/api/v1/ingest/ticket", json=body)
    assert again.status_code == 201 and again.json()["created"] is False


def test_ingest_without_labels_autolabels_from_corpus(client, cleanup):
    tid = f"TKT-T{uuid.uuid4().hex[:6]}"
    body = {"ticket_id": tid, "complaint_text": "I was billed twice for the same month and want the duplicate payment returned.",
            "resolution_steps": ["Refunded the duplicate payment."], "resolution_summary": "Duplicate payment refunded."}
    assert client.post("/api/v1/ingest/ticket", json=body).status_code == 201
    cleanup["ticket"].append(tid)
    hit = next(t for t in client.get("/api/v1/search", params={"q": body["complaint_text"], "source": "ticket", "strategy": "dense"}).json()["tickets"]
               if t["source_id"] == tid)
    assert hit["metadata"]["intent"] == "billing_dispute" and hit["metadata"]["product"] == "account"


@pytest.mark.parametrize("patch,expected", [
    ({"intent": "not_a_real_intent"}, "unknown intent"),
    ({"severity": "apocalyptic"}, "unknown severity"),
])
def test_ingest_unknown_label_is_rejected_with_actionable_message(client, patch, expected):
    r = client.post("/api/v1/ingest/ticket", json=_ticket(**patch))
    assert r.status_code == 422 and expected in r.json()["error"]["message"] and "/api/v1/taxonomy/labels" in r.json()["error"]["message"]


@pytest.mark.parametrize("patch", [{"resolution_steps": []}, {"complaint_text": "x"}, {"resolution_summary": ""}])
def test_ingest_invalid_payload_is_422(client, patch):
    assert client.post("/api/v1/ingest/ticket", json=_ticket(**patch)).status_code == 422


# ---------------------------------------------------------------- async batch
def test_batch_ingestion_runs_as_background_job(client, cleanup):
    ids = [f"TKT-B{uuid.uuid4().hex[:6]}" for _ in range(3)]
    cleanup["ticket"] += ids
    payload = {"tickets": [_ticket(ticket_id=i, complaint_text=f"Garage door opener loses wifi after every power cut number {n}.")
                           for n, i in enumerate(ids)]}
    r = client.post("/api/v1/ingest/tickets/batch", json=payload)
    assert r.status_code == 202
    job = r.json()["job_id"]
    for _ in range(50):
        status = client.get(f"/api/v1/jobs/{job}").json()
        if status["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.2)
    assert status["status"] == "succeeded" and status["result"]["created"] == 3 and status["result"]["failed"] == 0


# ---------------------------------------------------------------- stale KB handling
def test_deprecating_an_article_removes_it_from_retrieval(client, cleanup):
    aid = f"KB-T{uuid.uuid4().hex[:5]}"
    art = {"article_id": aid, "title": "Garage door opener loses Wi-Fi after power cut", "category": "device_issue", "product": "wifi_router",
           "content": "Garage door openers reconnect slowly after a power cut because the router assigns a new address. Reserve a fixed address for the opener.",
           "steps": ["Reserve a fixed IP address for the opener in the router."]}
    assert client.post("/api/v1/ingest/article", json=art).status_code == 201
    cleanup["article"].append(aid)
    q = {"q": "garage door opener will not reconnect to wifi after the power went out", "source": "article", "strategy": "dense", "limit": 3}
    assert client.get("/api/v1/search", params=q).json()["articles"][0]["source_id"] == aid
    assert client.post(f"/api/v1/articles/{aid}/deprecate", json={"reason": "superseded"}).status_code == 200
    assert aid not in {a["source_id"] for a in client.get("/api/v1/search", params=q).json()["articles"]}
    assert client.post("/api/v1/articles/KB-NOPE/deprecate", json={"reason": "x y z"}).status_code == 404


# ---------------------------------------------------------------- evolving taxonomy (new class, no code change)
def test_new_intent_class_flows_through_classification_retrieval_and_resolution(client, svc, cleanup):
    v0 = client.get("/api/v1/taxonomy").json()["version"]
    before = client.post("/api/v1/resolve", json={"complaint": ESIM_Q}).json()
    assert before["classification"]["intent"] != "esim_activation" and before["status"] == "abstained"   # unknown class -> abstain

    label = {"dimension": "intent", "label_id": "esim_activation", "team": "eSIM provisioning team",
             "description": "Problems activating, downloading or transferring an eSIM profile using a QR code",
             "keywords": ["esim", "qr code", "embedded sim"], "examples": ["My eSIM QR code will not activate"]}
    assert client.post("/api/v1/taxonomy/labels", json=label).status_code == 201
    cleanup["labels"].append("esim_activation")
    assert client.post("/api/v1/taxonomy/labels", json=label).status_code == 409                       # duplicate
    assert client.post("/api/v1/taxonomy/labels", json={**label, "label_id": "Bad Id"}).status_code == 422
    assert client.get("/api/v1/taxonomy").json()["version"] == v0 + 1

    art = {"article_id": f"KB-E{uuid.uuid4().hex[:5]}", "title": "eSIM QR code activation fails", "category": "esim_activation", "product": "mobile",
           "content": "eSIM activation fails when the QR code was already used or the phone has no internet during activation. Regenerate the eSIM profile and scan the new QR code over Wi-Fi.",
           "steps": ["Regenerate the eSIM profile.", "Scan the new QR code while connected to Wi-Fi."]}
    assert client.post("/api/v1/ingest/article", json=art).status_code == 201
    cleanup["article"].append(art["article_id"])
    for i, text in enumerate(["eSIM QR code says invalid or already used.", "Digital SIM profile will not download after scanning the code.",
                              "Cannot activate my eSIM, phone stays on the old line."]):
        tid = f"TKT-E{uuid.uuid4().hex[:5]}{i}"
        r = client.post("/api/v1/ingest/ticket", json={"ticket_id": tid, "complaint_text": text, "intent": "esim_activation", "product": "mobile",
                                                       "severity": "low", "sentiment": "neutral", "resolution_steps": ["Regenerated the eSIM profile.", "Activated over Wi-Fi."],
                                                       "resolution_summary": "eSIM profile regenerated."})
        assert r.status_code == 201
        cleanup["ticket"].append(tid)

    after = client.post("/api/v1/resolve", json={"complaint": ESIM_Q}).json()
    assert after["classification"]["intent"] == "esim_activation"
    assert after["status"] in ("resolved", "degraded") and after["resolution"]["steps"]       # response cache was invalidated by ingestion
    assert art["article_id"] in {c["source_id"] for c in after["citations"]} or any(c["source_id"].startswith("TKT-E") for c in after["citations"])


# ---------------------------------------------------------------- embedding re-index (model migration path)
def test_reindex_backfills_missing_embeddings_idempotently(client, svc, run, cleanup):
    tid = f"TKT-R{uuid.uuid4().hex[:6]}"
    assert client.post("/api/v1/ingest/ticket", json=_ticket(ticket_id=tid, complaint_text="Baby monitor camera feed freezes when the dishwasher starts.")).status_code == 201
    cleanup["ticket"].append(tid)
    run(svc.repo._exec, "DELETE FROM ticket_embeddings WHERE model=%s AND ticket_pk=(SELECT id FROM tickets WHERE ticket_id=%s)", (svc.embedder.model_name, tid))
    q = {"q": "Baby monitor camera feed freezes when the dishwasher starts.", "source": "ticket", "strategy": "dense", "limit": 3}
    assert tid not in {t["source_id"] for t in client.get("/api/v1/search", params=q).json()["tickets"]}
    assert run(svc.ingestion.reindex)["ticket"] == 1
    assert client.get("/api/v1/search", params=q).json()["tickets"][0]["source_id"] == tid
    assert run(svc.ingestion.reindex) == {"ticket": 0, "article": 0}                        # idempotent


# ---------------------------------------------------------------- auth + rate limiting
def test_api_key_auth_and_rate_limit(test_settings, test_database):
    from fastapi.testclient import TestClient

    from app.main import create_app

    secured = test_settings.model_copy(update={"api_keys": "key-one,key-two", "rate_limit_per_minute": 4, "cache_namespace": "rl-" + uuid.uuid4().hex[:6]})
    with TestClient(create_app(secured)) as c:
        assert c.get("/health").status_code == 200                                         # probes stay open
        assert c.get("/api/v1/stats").status_code == 401
        assert c.get("/api/v1/stats", headers={"X-API-Key": "wrong"}).status_code == 401
        codes = [c.get("/api/v1/stats", headers={"X-API-Key": "key-one"}).status_code for _ in range(6)]
        assert codes[:4] == [200] * 4 and codes[4:] == [429, 429]
        assert c.get("/api/v1/stats", headers={"X-API-Key": "key-two"}).status_code == 200      # limits are per key
