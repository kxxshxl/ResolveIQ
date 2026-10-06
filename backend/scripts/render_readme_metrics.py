"""Regenerate (or verify with --check) the result tables in README.md and docs/evaluation.md from data/eval/results/*.json and loadtest/results/latest_summary.json.

    python scripts/render_readme_metrics.py
    python scripts/render_readme_metrics.py --check
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation.readme_metrics import check_or_write  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 when a generated block is out of date")
    a = ap.parse_args()
    stale = check_or_write(write=not a.check)
    if a.check:
        print("out of date: " + ", ".join(stale) if stale else "generated blocks are up to date")
        sys.exit(1 if stale else 0)
    print("rewrote: " + ", ".join(stale) if stale else "nothing to change")
