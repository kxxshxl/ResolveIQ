"""Evaluation regression gate: fail CI when a recorded metric drops below its floor (or rises above its ceiling).

    python -m app.evaluation.run --suites classification retrieval robustness discovery
    python -m app.evaluation.gate            # exit code 1 on any failed or unmeasured check

Floors live in data/eval/thresholds.json as dotted paths into data/eval/results/latest.json. They are set a few points
below the recorded run so noise does not flap CI but real regressions (a bad embedding swap, a broken taxonomy change,
a worse severity model) do. Raise a floor deliberately when a change improves the system.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.config import REPO_ROOT


@dataclass
class CheckResult:
    path: str
    value: float | None
    bound: str
    ok: bool


def lookup(data: dict, path: str):
    cur = data
    for part in path.split("."):
        if isinstance(cur, list) and part.isdigit() and int(part) < len(cur):      # "sweep.1" = second entry of a list
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur if isinstance(cur, (int, float)) and not isinstance(cur, bool) else None


def evaluate(results: dict, checks: list[dict]) -> list[CheckResult]:
    out = []
    for c in checks:
        v = lookup(results, c["path"])
        if v is None:
            out.append(CheckResult(c["path"], None, "measured", False))
            continue
        ok, bound = True, []
        if "min" in c:
            ok &= v >= c["min"]
            bound.append(f">= {c['min']}")
        if "max" in c:
            ok &= v <= c["max"]
            bound.append(f"<= {c['max']}")
        out.append(CheckResult(c["path"], v, " and ".join(bound), bool(ok)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO_ROOT / "data" / "eval" / "results" / "latest.json"))
    ap.add_argument("--thresholds", default=str(REPO_ROOT / "data" / "eval" / "thresholds.json"))
    args = ap.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    checks = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))["checks"]
    rows = evaluate(results, checks)
    width = max(len(r.path) for r in rows)
    for r in rows:
        print(f"{'PASS' if r.ok else 'FAIL'}  {r.path:<{width}}  {('%.4f' % r.value) if r.value is not None else 'missing':>8}  ({r.bound})")
    failed = [r for r in rows if not r.ok]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
