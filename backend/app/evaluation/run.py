"""CLI: python -m app.evaluation.run --suites classification retrieval rag e2e evolving [--max-queries N]

Needs the database seeded (python scripts/seed.py). Writes data/eval/results/latest.json and docs/evaluation_results.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging

from app.core.config import REPO_ROOT, get_settings
from app.core.logging import configure_logging
from app.evaluation.report import render_markdown
from app.evaluation.runner import run_suites
from app.services.container import Services

ALL = ["classification", "retrieval", "robustness", "adaptive", "rag", "e2e", "evolving", "discovery"]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=ALL, choices=ALL + ["all"])
    ap.add_argument("--max-queries", type=int, default=None, help="cap test queries per suite (default: all)")
    ap.add_argument("--judge", type=int, default=0, help="also run the optional LLM judge on N answers")
    ap.add_argument("--llm-classifier", action="store_true", help="include the LLM zero-shot classifier (slow)")
    ap.add_argument("--alt-openai-model", default=None, help="extra RAG variant via the OpenAI-compatible provider (Ollama /v1)")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "eval" / "results" / "latest.json"))
    ap.add_argument("--report", default=str(REPO_ROOT / "docs" / "evaluation_results.md"))
    args = ap.parse_args()
    suites = ALL if "all" in args.suites else args.suites

    configure_logging("WARNING")
    logging.getLogger("app").setLevel(logging.WARNING)
    svc = await Services.build(get_settings())
    try:
        results = await run_suites(svc, suites, args.max_queries, include_llm_classifier=args.llm_classifier,
                                   judge_n=args.judge, alt_openai_model=args.alt_openai_model)
    finally:
        await svc.close()
    from pathlib import Path

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # merge with previous results so suites can be (re)run independently
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev.update({k: v for k, v in results.items() if k != "meta"})
    prev.setdefault("meta_by_run", []).append(results["meta"])
    prev["meta"] = results["meta"]
    out.write_text(json.dumps(prev, indent=2, default=str), encoding="utf-8")
    Path(args.report).write_text(render_markdown(prev), encoding="utf-8")
    print(f"wrote {out} and {args.report} (elapsed {results['meta']['elapsed_s']}s)")


if __name__ == "__main__":
    asyncio.run(main())
