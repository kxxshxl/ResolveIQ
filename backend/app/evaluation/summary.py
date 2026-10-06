"""Recorded evaluation results as one JSON document: what the console's Evaluation view and the README renderer read.

Only files that exist are included; every section says which file it came from and when that file was written, so nothing on screen is presented as live
when it is a recording. Nothing is computed here beyond reading and trimming.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import REPO_ROOT

FILES = {"latest": "data/eval/results/latest.json", "drift_demo": "data/eval/results/drift_demo.json", "pgvector_scale": "data/eval/results/pgvector_scale.json",
         "db_query_plans": "data/eval/results/db_query_plans.json", "failure_analysis": "data/eval/results/failure_analysis.json", "ingest_benchmark": "data/eval/results/ingest_benchmark.json", "load_test": "loadtest/results/latest_summary.json"}


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_results(root: Path | None = None) -> dict:
    root = root or REPO_ROOT
    out: dict = {"files": {}}
    for key, rel in FILES.items():
        p = root / rel
        data = _read(p) if p.exists() else None
        if data is None:
            continue
        if key == "latest":
            data.pop("meta_by_run", None)
        out[key] = data
        out["files"][key] = {"path": rel, "written": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")}
    return out
