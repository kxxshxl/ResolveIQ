"""Every number in the README's and docs/evaluation.md's result tables comes from the recorded result files; if either side changes without the other, this fails.

Regenerate with:  python scripts/render_readme_metrics.py
"""
import re

from app.core.config import REPO_ROOT
from app.evaluation.readme_metrics import END, START, TARGETS, check_or_write, render


def test_generated_blocks_are_in_sync_with_the_result_files():
    stale = check_or_write(write=False)
    assert stale == [], f"out of date: {stale}. Run: python backend/scripts/render_readme_metrics.py"


def test_both_documents_carry_a_generated_block():
    for rel in TARGETS:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert text.count(START) == 1 and text.count(END) == 1, rel


def test_nothing_is_rendered_for_missing_results():
    out = render("readme", {"files": {}})
    assert "not recorded" in out.lower() or "is missing" in out
    assert not re.search(r"\d\.\d\d", out), "no metric may appear when no result file exists"


def test_partial_results_render_without_inventing_the_rest():
    res = {"files": {}, "latest": {"meta": {"timestamp": "2026-01-01T00:00:00+00:00", "corpus": {"tickets": 3, "articles": 1}, "eval_sets": {"gold": 2}, "llm": ["x"]},
                                   "adaptive": {"config": {"margin_ticket": 0.0, "margin_article": 0.1, "held_out": ["gold"]},
                                                "results": {"gold": {"ticket": {"dense": {"hit@1": 0.5, "hit@5": 0.75, "mrr": 0.6, "ndcg@10": 0.4, "n": 2, "latency": {"p50_ms": 1.0, "p95_ms": 2.0}}}}}}}}
    out = render("evaluation", res)
    assert "0.50" in out and "0.75" in out and "dense (pgvector)" in out
    assert "not measured" in out      # classification, generation and the rest are absent
