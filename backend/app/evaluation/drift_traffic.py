"""Synthetic production traffic for trying out drift monitoring: logged /resolve requests placed in time.

Baseline: labelled complaints from the evaluation sets, 2 to 12 days old. Recent: other labelled complaints from the last day, plus one synthetic
incident (distinct complaints about a smart-home hub the corpus has never seen, all abstained). Every row is marked trace_id = MARKER so it can be
removed without touching real traffic. Used by the tests and by scripts/seed_drift_traffic.py.
"""
from __future__ import annotations

import itertools
import uuid

import numpy as np
from psycopg.types.json import Jsonb

from app.core.config import REPO_ROOT
from app.db.repository import Repository
from app.evaluation.datasets import load_jsonl

MARKER = "synthetic-drift-demo"
ISSUES = ["keeps disconnecting", "will not pair", "shows offline", "went dark", "blinks red", "stopped responding"]
DEVICES = ["thermostat", "door lock", "ceiling lights", "garden camera", "blinds", "smoke alarm"]
WHEN = ["since the firmware update", "after the power cut", "ever since last Tuesday", "since we moved the router", "after a factory reset"]


def incident(n: int) -> list[str]:
    combos = list(itertools.product(ISSUES, DEVICES, WHEN))
    return [f"My smart home hub {i} {w} and I cannot control my {d} from the app." for i, d, w in (combos[(j * 7) % len(combos)] for j in range(n))]


def eval_pool(data_dir=None) -> list[dict]:
    d = (data_dir or REPO_ROOT / "data") / "eval"
    seen, out = set(), []
    for name in ("queries", "gold", "gold_v2", "gold_blind"):
        for r in load_jsonl(d / f"{name}.jsonl"):
            if r["text"] not in seen:
                seen.add(r["text"])
                out.append(r)
    return out


async def _insert(repo: Repository, text: str, cls: dict, status: str, evidence: float, hours_ago: float) -> str:
    rid = str(uuid.uuid4())
    await repo._exec(
        """INSERT INTO resolution_requests (request_id,trace_id,complaint,classification,retrieved,result,status,confidence,latency_ms,created_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,10, now() - make_interval(hours => %s))""",
        (rid, MARKER, text, Jsonb(cls), Jsonb([]), Jsonb({"evidence": {"confidence": evidence}}), status, evidence, hours_ago))
    return rid


def _cls(r: dict) -> dict:
    return {"intent": r["intent"], "product": r["product"], "severity": r["severity"], "sentiment": r["sentiment"]}


async def seed_traffic(repo: Repository, baseline: int = 170, recent: int = 70, incident_n: int = 25, seed: int = 3) -> dict[str, list[str]]:
    """Insert the traffic; returns the request ids of the incident complaints."""
    pool = eval_pool()
    order = np.random.default_rng(seed).permutation(len(pool))
    base, rec = [pool[i] for i in order[:baseline]], [pool[i] for i in order[baseline:baseline + recent]]
    for i, r in enumerate(base):
        await _insert(repo, r["text"], _cls(r), "resolved", 0.88, 48 + (i % 240))
    for i, r in enumerate(rec):
        await _insert(repo, r["text"], _cls(r), "resolved", 0.88, 1 + (i % 22))
    ids = []
    for i, text in enumerate(incident(incident_n)):
        cls = {"intent": "device_issue", "product": "wifi_router", "severity": "medium", "sentiment": "frustrated"}
        ids.append(await _insert(repo, text, cls, "abstained", 0.3, 1 + (i % 22)))
    return {"incident": ids}


async def clear_traffic(repo: Repository) -> int:
    rows = await repo._fetch("DELETE FROM resolution_requests WHERE trace_id = %s RETURNING 1", (MARKER,))
    return len(rows)
