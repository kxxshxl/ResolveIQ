"""Postgres job queue semantics: exclusive claim, retry with backoff, failure after max attempts, stale-job recovery."""
import asyncio

import pytest

from app import jobs
from app.worker import run_one

KIND = "qtest"


@pytest.fixture()
def handlers(svc, run):
    calls = {"n": 0}

    async def ok(s, payload, job_id):
        return {"echo": payload.get("x")}

    async def flaky(s, payload, job_id):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return {"attempt": calls["n"]}

    async def boom(s, payload, job_id):
        raise ValueError("always fails")

    jobs.HANDLERS.update({KIND + "_ok": ok, KIND + "_flaky": flaky, KIND + "_boom": boom})
    old = svc.settings.worker_retry_backoff_seconds
    svc.settings.worker_retry_backoff_seconds = 0.0
    yield calls
    svc.settings.worker_retry_backoff_seconds = old
    for k in list(jobs.HANDLERS):
        if k.startswith(KIND):
            del jobs.HANDLERS[k]
    run(svc.repo._exec, "DELETE FROM jobs WHERE kind LIKE %s", (KIND + "%",))


def _status(run, svc, job_id):
    return run(svc.repo.get_job, job_id)


def test_success_path_records_result(svc, run, handlers):
    jid = run(svc.repo.create_job, KIND + "_ok", {"x": 7})
    job = run(svc.repo.claim_job, "w1", [KIND + "_ok"])
    assert job["job_id"] == jid and job["attempts"] == 1
    assert run(run_one, svc, job) == "succeeded"
    row = _status(run, svc, jid)
    assert row["status"] == "succeeded" and row["result"] == {"echo": 7} and row["finished_at"] is not None


def test_a_job_is_claimed_by_exactly_one_of_many_concurrent_workers(svc, run, handlers):
    run(svc.repo.create_job, KIND + "_ok", {})

    async def race():
        return await asyncio.gather(*[svc.repo.claim_job(f"w{i}", [KIND + "_ok"]) for i in range(8)])

    got = [j for j in run(race) if j]
    assert len(got) == 1


def test_distinct_jobs_go_to_distinct_workers(svc, run, handlers):
    ids = {run(svc.repo.create_job, KIND + "_ok", {}) for _ in range(3)}

    async def race():
        return await asyncio.gather(*[svc.repo.claim_job(f"w{i}", [KIND + "_ok"]) for i in range(3)])

    assert {j["job_id"] for j in run(race)} == ids


def test_transient_failure_is_retried_then_succeeds(svc, run, handlers):
    jid = run(svc.repo.create_job, KIND + "_flaky", {})
    assert run(run_one, svc, run(svc.repo.claim_job, "w", [KIND + "_flaky"])) == "queued"      # attempt 1 failed -> back in the queue
    assert _status(run, svc, jid)["error"].startswith("RuntimeError")
    assert run(run_one, svc, run(svc.repo.claim_job, "w", [KIND + "_flaky"])) == "succeeded"   # attempt 2 ok
    assert _status(run, svc, jid)["attempts"] == 2


def test_permanent_failure_stops_after_max_attempts(svc, run, handlers):
    jid = run(svc.repo.create_job, KIND + "_boom", {})
    run(svc.repo._exec, "UPDATE jobs SET max_attempts=2 WHERE job_id=%s", (jid,))
    assert run(run_one, svc, run(svc.repo.claim_job, "w", [KIND + "_boom"])) == "queued"
    assert run(run_one, svc, run(svc.repo.claim_job, "w", [KIND + "_boom"])) == "failed"
    assert run(svc.repo.claim_job, "w", [KIND + "_boom"]) is None
    assert _status(run, svc, jid)["status"] == "failed"


def test_backoff_delays_the_retry(svc, run, handlers):
    svc.settings.worker_retry_backoff_seconds = 3600.0
    jid = run(svc.repo.create_job, KIND + "_boom", {})
    run(run_one, svc, run(svc.repo.claim_job, "w", [KIND + "_boom"]))
    assert run(svc.repo.claim_job, "w", [KIND + "_boom"]) is None  # not runnable until run_after
    assert _status(run, svc, jid)["status"] == "queued"


def test_jobs_of_a_dead_worker_are_recovered(svc, run, handlers):
    jid = run(svc.repo.create_job, KIND + "_ok", {})
    run(svc.repo.claim_job, "dead-worker", [KIND + "_ok"])
    run(svc.repo._exec, "UPDATE jobs SET locked_at = now() - interval '1 hour' WHERE job_id=%s", (jid,))
    assert run(svc.repo.requeue_stale_jobs, 90) >= 1
    assert _status(run, svc, jid)["status"] == "queued"
    assert run(svc.repo.claim_job, "w2", [KIND + "_ok"])["job_id"] == jid


def test_a_live_heartbeat_prevents_recovery(svc, run, handlers):
    jid = run(svc.repo.create_job, KIND + "_ok", {})
    run(svc.repo.claim_job, "alive", [KIND + "_ok"])
    run(svc.repo.heartbeat_job, jid)
    run(svc.repo.requeue_stale_jobs, 90)
    assert _status(run, svc, jid)["status"] == "running"
